from medsam_cancer_detection_corewise_simple import get_dataloaders
import torch
from torch import nn
from segment_anything.modeling.common import LayerNorm2d
from src.masking_generator import MaskingGenerator
from copy import deepcopy
import wandb
from tqdm import tqdm
from pathlib import Path
from medAI.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
import os
from rich_argparse import RichHelpFormatter, ArgumentDefaultsRichHelpFormatter
import typing as tp
import logging 
from abc import ABC, abstractmethod
import einops
logging.basicConfig(level=logging.INFO)


torch.autograd.set_detect_anomaly(True)

class ModelFactory:
    @staticmethod
    def _medsam_image_encoder():
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](
            checkpoint="/ssd005/projects/exactvu_pca/checkpoint_store/medsam_vit_b_cpu.pth"
        )
        image_encoder = sam_model.image_encoder
        return image_encoder


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsRichHelpFormatter, description="Performs SSL training on MedSAM backbone")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--benign-to-cancer-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-knopp-centering-mode", type=str, default="in_mask", choices=["in_mask", "whole_image"], 
                        help="Whether to center the sinkhorn-knopp algorithm on the tokens that are in the mask, or across the whole image")
    return parser.parse_args()


def main(config):
    logging.info("===========================================")
    logging.info("Starting experiment")
    logging.info("===========================================")
    
    logging.info(f"Config: {config}")

    state = None
    if config.checkpoint_dir is not None:
        config.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        if "experiment.ckpt" in os.listdir(config.checkpoint_dir):
            logging.info(f"Loading from checkpoint found in {config.checkpoint_dir}")
            state = torch.load(config.checkpoint_dir / "experiment.ckpt")

    logging.info("Loading data")
    train_loader, val_loader, test_loader = get_dataloaders(
        config.fold,
        config.n_folds,
        config.benign_to_cancer_ratio,
        debug=config.debug,
        augmentation=None,
    )
    logging.info(f"Train: {len(train_loader)} batches, {len(train_loader.dataset)} images")
    logging.info(f"Val: {len(val_loader)} batches, {len(val_loader.dataset)} images")
    logging.info(f"Test: {len(test_loader)} batches, {len(test_loader.dataset)} images")

    logging.info("Building model")
    model = IBotStyleModel()
    torch.compile(model)
    model.cuda()
    logging.info(f"Model <{model.__class__.__name__}> built: ")
    logging.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if state is not None:
        model.load_state_dict(state["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        5 * len(train_loader),
        config.epochs * len(train_loader),
        warmup_start_lr=1e-9,
        eta_min=1e-7,
    )

    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    wandb.init(
        project="sam_ssl_pretraining",
        config=config,
        dir=config.checkpoint_dir,
        id=state["wandb_id"] if state is not None else None,
    )
    wandb.watch(model, log_freq=100)

    start_epoch = state["epoch"] if state is not None else 0

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch}")

        if config.checkpoint_dir is not None:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "wandb_id": wandb.run.id,
                    "epoch": epoch,
                },
                config.checkpoint_dir / "experiment.ckpt",
            )

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            
            optimizer.zero_grad()
            loss = model(batch[0].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            model.ema_update()
            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        if config.checkpoint_dir is not None:
            torch.save(
                model.student.encoder.state_dict(),
                config.checkpoint_dir / f"encoder_{epoch}.pth",
            )


@torch.no_grad()
def do_ema_update(teacher, student, alpha=0.999):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)


class SinkhornKnoppCenteringModule(nn.Module): 
    """Does the sinkhorn-knopp algorithm for optimal transport.
    
    Details of the sinkhorn-knopp algorithm can be found in the method
    ``sinkhorn`` below. This module is a wrapper around that method that
    allows us to do the sinkhorn-knopp algorithm on minibatches of scores, 
    while maintaining a queue of the old minibatch scores, in order to prevent the 
    chance of too small minibatches.
    """

    def __init__(self, use_cache=False, cache_size: int = 1000, eps: float = 0.05, niters: int = 3, n_classes=1024):
        super().__init__()
        self.cache_size = cache_size
        self.eps = eps
        self.niters = niters
        self.use_cache = use_cache

        self.register_buffer('queue', torch.zeros(cache_size, n_classes))
        self.register_buffer('queue_size', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def forward(self, scores): 
        B, N = scores.shape
        
        if self.use_cache: 
            # roll the queue elements to make room
            torch.roll(self.queue, shifts=B, dims=0)
            # overwrite the oldest elements with the new scores in the first rows
            self.queue[:B] = scores 
            # do the sinkhorn-knopp algorithm using the whole queue 
            self.queue_size[:] = min(self.queue_size + B, self.cache_size)
            scores = self.queue[:self.queue_size[0]]
            Q = self.sinkhorn(scores, eps=self.eps, niters=self.niters)
            # return the first rows of the queue, which are the scores for the current batch
            return Q[:B]
        
        else: 
            return self.sinkhorn(scores, eps=self.eps, niters=self.niters)

    def sinkhorn(self, scores, eps=0.05, niters=3):
        """Does the sinkhorn-knopp algorithm for optimal transport. 

        Informally, it receives a NxM matrix of "scores". N is the minibatch 
        dimension and M is the number of tokens. The i,j'th entry of the matrix
        tells us the "score" for how much the i'th minibatch element should be
        assigned to the j'th token. In principle this kind of input could strongly
        prefer one token over all others, but the sinkhorn-knopp algorithm will 
        prevent this from happening by ensuring each row and column of the matrix is 
        assigned more uniformly. In particular, each token has to be assigned to 
        the same number of minibatch elements, and each minibatch element has to be
        assigned to the same number of tokens.

        Args:
            scores (torch.Tensor): NxM matrix of scores
            eps (float, optional): Epsilon for the sinkhorn-knopp algorithm. Defaults to 0.05.
            niters (int, optional): Number of iterations to run the sinkhorn-knopp algorithm. Defaults to 3.    
        """
        Q = torch.exp(scores / eps).T
        Q /= torch.sum(Q)
        K, B = Q.shape
        u, r, c = (
            torch.zeros(K, device=scores.device),
            torch.ones(K, device=scores.device) / K,
            torch.ones(B, device=scores.device) / B,
        )
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).T


class IBotStyleModel(nn.Module):
    """Runs IBot-style SSL training on the medsam backbone. 
    
    IBot does student-teacher training, with the student receiving masked 
    inputs and the teacher receiving unmasked inputs. The student is trained
    to predict the correct 

    enc
    """

    def __init__(
        self,
        encoder_transformer_dim=768,
        proj_dim=512,
        num_classes=1024,
        feature_map_size=64,
        min_num_patches=16,
        max_num_patches=100,
        mask_ratio=0.3,
        ema_alpha=0.999,
        sinkhorn_knopp_centering_mode: tp.Literal["in_mask", "whole_image"] = "in_mask",
        token_matching_mode: tp.Literal["hard", "soft"] = "hard",
        temp=1,
    ):
        super().__init__()
        self.mask_gen = MaskingGenerator(
            (feature_map_size, feature_map_size),
            int(feature_map_size * feature_map_size * mask_ratio),
            min_num_patches=min_num_patches,
            max_num_patches=max_num_patches,
        )
        self.student = MaskableMedSAMWithProjection(
            encoder_transformer_dim, proj_dim, num_classes
        )
        self.teacher = deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.ema_alpha = ema_alpha
        self.temp = temp
        self.sinkhorn_knopp_centering_mode = sinkhorn_knopp_centering_mode
        self.token_matching_mode = token_matching_mode

        self.sinkhorn = SinkhornKnoppCenteringModule(n_classes=num_classes)

    def generate_masks(self, image):
        masks = []
        for _ in range(image.shape[0]):
            masks.append(torch.from_numpy(self.mask_gen()).bool().to(image.device))
        return torch.stack(masks)

    def forward(self, image):
        mask = self.generate_masks(image)
        with torch.no_grad():
            target_token_scores = self.teacher(image, mask=None) # B, self.num_classes, 64, 64 
            target_token_scores = target_token_scores.permute(0, 2, 3, 1) # B, 64, 64, self.num_classes

            if self.sinkhorn_knopp_centering_mode == 'in_mask': 
                # mask out the tokens that are not in the mask - the output is of shape 
                # N, self.num_classes where N is sum of number of tokens in each mask across
                # the whole batch. Note that this means we are centering the sinkhorn-knopp
                # algorithm on the tokens that are actually in the mask, and across the batch. 
                target_token_scores = target_token_scores[mask] 
                target_token_scores = self.sinkhorn(target_token_scores)
                target_dist = (target_token_scores / self.temp).softmax(-1)
            
            elif self.sinkhorn_knopp_centering_mode == 'whole_image':
                # this time, we should do the sinkhorn knopp centering first. We 
                # need to collapse the batch and spatial dimensions to do this.
                B, H, W, C = target_token_scores.shape
                B1, H1, W1 = mask.shape
                assert B == B1 and H == H1 and W == W1, "mask shape must match image shape"
                target_token_scores = einops.rearrange(
                    target_token_scores, 'b h w c -> (b h w) c'
                )
                mask = einops.rearrange(mask, 'b h w -> (b h w)')
                target_token_scores = self.sinkhorn(target_token_scores)
                target_dist = (target_token_scores / self.temp).softmax(-1)
                target_dist = target_dist[mask]

            else: 
                raise ValueError(f"Unknown sinkhorn-knopp centering mode {self.sinkhorn_knopp_centering_mode}")

        # now compute the student token scores
        student_token_scores = self.student(image, mask=mask)
        student_token_scores = student_token_scores.permute(0, 2, 3, 1)
        student_token_scores = student_token_scores[mask]

        if self.token_matching_mode == 'soft': 
            # compute the cross entropy loss between the student and teacher token scores
            loss = torch.sum(
                -target_dist * torch.log_softmax(student_token_scores, dim=-1), dim=-1
            ).mean()
        elif self.token_matching_mode == 'hard': 
            # discretize the target distribution to integer labels 
            target_labels = torch.argmax(target_dist, dim=-1)
            loss = torch.nn.functional.cross_entropy(student_token_scores, target_labels)
        else: 
            raise ValueError(f"Unknown token matching mode {self.token_matching_mode}")

        return loss

    def ema_update(self):
        do_ema_update(self.teacher, self.student, self.ema_alpha)

    @property
    def image_encoder(self):
        return self.student.encoder


class MaskableMedSAMWithProjection(nn.Module):
    """Wraps the medsam image encoder with a projection head and tokenization head. 
    
    The input of this model will be a batch of images of shape (B, 3, 1024, 1024). The intermediate
    output of the model will be a batch of patch embeddings of shape (B, 256, 64, 64). The final output
    will be a batch of token embeddings of shape (B, 1024, 64, 64), where each position (i, :, j, k) 
    is the vector of token ``scores`` for the patch at position (i, j, k).
    """

    def __init__(self, encoder_transformer_dim=768, proj_dim=512, ntokens=1024):
        super().__init__()
        self.encoder = ModelFactory._medsam_image_encoder()

        self.proj = nn.Sequential(
            nn.Conv2d(256, proj_dim, 1),
            LayerNorm2d(proj_dim),
            nn.Conv2d(512, ntokens, 1),
        )

        self.mask_token = torch.nn.Parameter(torch.randn(encoder_transformer_dim))

    def forward(self, image, mask=None):
        embed = self.encoder.patch_embed(image)  # B, N, H, W

        if mask is not None:
            embed[mask] = self.mask_token

        # do the rest of the forward pass
        x = embed

        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))
        x = self.proj(x)
        return x


if __name__ == "__main__":
    config = parse_args()
    main(config)
