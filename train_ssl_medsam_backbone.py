from medsam_cancer_detection_corewise_simple import get_dataloaders
import torch
from torch import nn
from segment_anything.modeling.common import LayerNorm2d
from medAI.utils.masking_generator import MaskingGenerator
from medAI.modeling.swav import sinkhorn_knopp
from copy import deepcopy
import wandb
from tqdm import tqdm
from pathlib import Path
from medAI.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
import os
from rich_argparse import RichHelpFormatter
import logging 
logging.basicConfig(level=logging.INFO)


torch.autograd.set_detect_anomaly(True)


class ModelFactory:
    @staticmethod
    def _medsam_image_encoder():
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](
            checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
        )
        image_encoder = sam_model.image_encoder
        return image_encoder


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(formatter_class=RichHelpFormatter, description="Performs SSL training on MedSAM backbone")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--benign-to-cancer-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-6)
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


def sinkhorn(scores, eps=0.05, niters=3):
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

    def generate_masks(self, image):
        masks = []
        for _ in range(image.shape[0]):
            masks.append(torch.from_numpy(self.mask_gen()).bool().to(image.device))
        return torch.stack(masks)

    def forward(self, image):
        mask = self.generate_masks(image)
        with torch.no_grad():
            target_token_scores = self.teacher(image, mask=None)
            target_token_scores = target_token_scores.permute(0, 2, 3, 1)
            target_token_scores = target_token_scores[mask]
            target_token_scores = sinkhorn(target_token_scores)
            target_dist = (target_token_scores / self.temp).softmax(-1)

        student_token_scores = self.student(image, mask=mask)
        student_token_scores = student_token_scores.permute(0, 2, 3, 1)
        student_token_scores = student_token_scores[mask]

        loss = torch.sum(
            -target_dist * torch.log_softmax(student_token_scores, dim=-1), dim=-1
        ).mean()

        return loss

    def ema_update(self):
        do_ema_update(self.teacher, self.student, self.ema_alpha)

    @property
    def image_encoder(self):
        return self.student.encoder


class MaskableMedSAMWithProjection(nn.Module):
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
