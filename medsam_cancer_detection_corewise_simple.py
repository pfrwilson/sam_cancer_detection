from src.datasets import ExactNCT2013BModeImages, CohortSelectionOptions
import torch
from torch import nn
from einops import rearrange, repeat
from dataclasses import dataclass
from src.utils import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
import wandb
import numpy as np
from rich_argparse import RichHelpFormatter
import argparse
import random 



class ModelRegistry:
    encoder_checkpoint = None

    @classmethod
    def sam_backbone_average_linear(cls):
        """
        Uses the sam model backbone, but replaces the segmentation head with average pooling and a linear layer.
        """
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](
            checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
        )
        image_encoder = sam_model.image_encoder

        if cls.encoder_checkpoint is not None:
            print("Loading encoder checkpoint")
            image_encoder.load_state_dict(torch.load(cls.encoder_checkpoint))

        pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        flatten = torch.nn.Flatten()
        fc = torch.nn.Linear(256, 1)
        model_ = torch.nn.Sequential(image_encoder, pool, flatten, fc)

        class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                self.model = model_

            def forward(self, X, mask):
                return model_(X)

        model = Model()
        criterion = nn.BCEWithLogitsLoss()
        return model, criterion

    @classmethod
    def _v1(cls, freeze_encoder=False):
        model = MedSAMClassifierBackboneOnly(freeze_encoder=freeze_encoder)

        if cls.encoder_checkpoint is not None:
            print("Loading encoder checkpoint")
            model.image_encoder.load_state_dict(torch.load(cls.encoder_checkpoint))

        criterion = nn.BCEWithLogitsLoss()
        return model, criterion

    @classmethod
    def finetune_backbone_linear_head_needle_region(cls):
        return cls._v1(False)

    @classmethod
    def frozen_backbone_linear_head(cls):
        return cls._v1(True)

    @classmethod
    def v0_adapters_64(cls):
        from models import wrap_image_encoder_with_adapter, freeze_non_adapter_layers

        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](
            checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
        )
        image_encoder = sam_model.image_encoder
        image_encoder = wrap_image_encoder_with_adapter(image_encoder, adapter_dim=64)
        image_encoder = freeze_non_adapter_layers(image_encoder)
        pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        flatten = torch.nn.Flatten()
        fc = torch.nn.Linear(256, 1)
        model_ = torch.nn.Sequential(image_encoder, pool, flatten, fc)

        class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                self.model = model_

            def forward(self, X, mask):
                return model_(X)

        model = Model()
        criterion = nn.BCEWithLogitsLoss()
        return model, criterion

    @classmethod
    def v1_adapters_64(cls):
        from models import wrap_image_encoder_with_adapter, freeze_non_adapter_layers

        model = MedSAMClassifierBackboneOnly()
        model.image_encoder = wrap_image_encoder_with_adapter(
            model.image_encoder, adapter_dim=64
        )
        freeze_non_adapter_layers(model.image_encoder)
        criterion = nn.BCEWithLogitsLoss()
        return model, criterion

    @classmethod
    def _v1_adapters_thaw_patch_embed(cls, dim):
        from models import wrap_image_encoder_with_adapter, freeze_non_adapter_layers

        model = MedSAMClassifierBackboneOnly()
        model.image_encoder = wrap_image_encoder_with_adapter(
            model.image_encoder, adapter_dim=dim
        )
        freeze_non_adapter_layers(model.image_encoder)
        criterion = nn.BCEWithLogitsLoss()

        for param in model.image_encoder.patch_embed.parameters():
            param.requires_grad = True

        return model, criterion

    @classmethod
    def v1_adapters_64_thaw_patch_embed(cls):
        return ModelRegistry._v1_adapters_thaw_patch_embed(64)

    @classmethod
    def v1_adapters_256_thaw_patch_embed(cls):
        return ModelRegistry._v1_adapters_thaw_patch_embed(256)

    @classmethod 
    def backbone_image_wise_attention_pool(cls): 
        from src.sam import build_medsam
        sam_model = build_medsam()
        image_encoder = sam_model.image_encoder

        if cls.encoder_checkpoint is not None:
            print("Loading encoder checkpoint")
            image_encoder.load_state_dict(torch.load(cls.encoder_checkpoint))

        pool = SimpleAttentionPooling(256)
        flatten = torch.nn.Flatten()
        fc = torch.nn.Linear(256, 1)
        model_ = torch.nn.Sequential(image_encoder, pool, flatten, fc)

        class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__()
                self.model = model_

            def forward(self, X, mask):
                return model_(X)

        model = Model()
        criterion = nn.BCEWithLogitsLoss()
        return model, criterion



# all class methods that don't start with _
_MODEL_NAMES = [
    k
    for k in dir(ModelRegistry)
    if not k.startswith("_") and callable(getattr(ModelRegistry, k))
]


def parse_args() -> argparse.Namespace:
    from argparse import ArgumentParser

    parser = ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("--name", type=str, required=False, default=None)
    parser.add_argument("--project", type=str, default="medsam_cancer_detection_corewise_simple", help="Wandb project name")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug mode"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--model-name", type=str, choices=_MODEL_NAMES, default=_MODEL_NAMES[0]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--benign-to-cancer-ratio", type=float, default=None)
    parser.add_argument(
        "--augmentation", type=str, default="none", choices=("none", "v1", "v2")
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--load-encoder-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main(config):

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        config.fold,
        config.n_folds,
        config.benign_to_cancer_ratio,
        config.debug,
        augmentation=config.augmentation,
    )

    ModelRegistry.encoder_checkpoint = config.load_encoder_checkpoint
    model, criterion = getattr(ModelRegistry, config.model_name)()
    model = model.cuda()
    torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        5 * len(train_loader),
        config.epochs * len(train_loader),
        warmup_start_lr=1e-9,
        eta_min=1e-7,
    )
    scaler = torch.cuda.amp.GradScaler()

    best_score = 0

    wandb.init(
        project="medsam_cancer_detection_corewise_simple",
        config=config,
        name=config.name,
    )

    for epoch in range(config.epochs):
        print(f"Epoch {epoch}", flush=True)
        print(f"Training...")
        pred_list, label_list, involvement_list = run_epoch(
            config,
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            desc=f"Epoch {epoch}",
        )

        metrics = compute_metrics(pred_list, label_list, involvement_list)
        metrics = {f"train/{k}": v for k, v in metrics.items()}
        wandb.log(metrics)

        print(f"Validation...", flush=True)
        pred_list, label_list, involvement_list = run_epoch(
            config, model, val_loader, criterion, desc="Validation"
        )
        metrics = compute_metrics(pred_list, label_list, involvement_list)
        metrics = {f"val/{k}": v for k, v in metrics.items()}

        wandb.log(metrics)
        if metrics["val/auc_high_involvement"] > best_score:
            print(f"New best score: {metrics['val/auc_high_involvement']}", flush=True)
            best_score = metrics["val/auc_high_involvement"]

            print(f"Testing...", flush=True)
            pred_list, label_list, involvement_list = run_epoch(
                config, model, test_loader, criterion, desc="Test"
            )
            metrics = compute_metrics(pred_list, label_list, involvement_list)
            metrics = {f"test/{k}": v for k, v in metrics.items()}
            wandb.log(metrics)

        scheduler.step()


def run_epoch(
    config,
    model,
    loader,
    criterion,
    optimizer=None,
    scheduler=None,
    scaler=None,
    desc=None,
):
    training = optimizer is not None
    desc = desc or ("training" if training else "validation")

    model.train() if training else model.eval()
    with torch.set_grad_enabled(training):
        pred_list = []
        label_list = []
        involvement_list = []

        for i, (image, needle_mask, involvement, label) in enumerate(
            tqdm(loader, desc=desc)
        ):
            image = image.cuda()
            involvement = involvement.cuda()
            label = label.cuda()

            with torch.cuda.amp.autocast():
                pred = model(image, needle_mask)
                pred = pred.squeeze(-1)
                loss_val = criterion(pred, label.float())

            if training:
                scaler.scale(loss_val).backward()

                if i % config.batch_size // 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                scheduler.step()

                wandb.log(
                    {
                        "loss": loss_val.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            pred_list.append(pred.sigmoid().detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            involvement_list.append(involvement.detach().cpu().numpy())

        pred_list = np.concatenate(pred_list)
        label_list = np.concatenate(label_list)
        involvement_list = np.concatenate(involvement_list)

    return pred_list, label_list, involvement_list


def compute_metrics(pred_list, label_list, involvement_list):
    metrics = {}

    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    try:
        auc = roc_auc_score(label_list, pred_list)
    except ValueError:
        auc = np.nan
    metrics["auc"] = auc
    metrics["acc"] = balanced_accuracy_score(label_list, pred_list > 0.5)

    high_involvement = (involvement_list > 0.4) | (involvement_list == 0)
    pred_list = pred_list[high_involvement]
    label_list = label_list[high_involvement]

    try:
        auc = roc_auc_score(label_list, pred_list)
    except ValueError:
        auc = np.nan
    metrics["auc_high_involvement"] = auc
    metrics["acc_high_involvement"] = balanced_accuracy_score(
        label_list, pred_list > 0.5
    )

    return metrics


def get_dataloaders(
    fold=0, n_folds=5, benign_to_cancer_ratio=None, debug=False, augmentation="none", batch_size=4
):
    class Transform:
        def __init__(self, augmentation="none"):
            self.augmentation = augmentation

        def __call__(self, item):
            image = item["bmode"]
            image = torch.from_numpy(image.copy()).float()
            image = image.unsqueeze(0)
            image = (image - image.min()) / (image.max() - image.min())
            from torchvision.transforms import v2 as T
            from torchvision.tv_tensors import Image, Mask

            image = T.Resize((1024, 1024), antialias=True)(image)

            needle_mask = item["needle_mask"]
            needle_mask = torch.from_numpy(needle_mask.copy()).float()
            needle_mask = needle_mask.unsqueeze(0)
            needle_mask = T.Resize((1024, 1024), antialias=True)(needle_mask)

            if self.augmentation == "v1":
                image, needle_mask = T.RandomResizedCrop(
                    1024, scale=(0.7, 1.0), antialias=True
                )(Image(image), Mask(needle_mask))
            if self.augmentation == "v2":
                image, needle_mask = T.RandomAffine(degrees=0, translate=(0.2, 0.2))(
                    Image(image), Mask(needle_mask)
                )

            image = image.repeat(3, 1, 1)

            involvement = torch.tensor(item["pct_cancer"]).float() / 100.0
            if torch.isnan(involvement).item():
                involvement = torch.tensor(0.0).float()

            label = torch.tensor(item["grade"] != "Benign").float()
            return image, needle_mask, involvement, label

    train_ds = ExactNCT2013BModeImages(
        split="train",
        transform=Transform(augmentation=augmentation),
        cohort_selection_options=CohortSelectionOptions(
            fold=fold,
            n_folds=n_folds,
            min_involvement=40,
            remove_benign_from_positive_patients=True,
            benign_to_cancer_ratio=benign_to_cancer_ratio,
        ),
    )

    val_ds = ExactNCT2013BModeImages(
        split="val",
        transform=Transform(),
        cohort_selection_options=CohortSelectionOptions(
            fold=fold,
            n_folds=n_folds,
            min_involvement=None,
            remove_benign_from_positive_patients=False,
            benign_to_cancer_ratio=None,
        ),
    )

    test_ds = ExactNCT2013BModeImages(
        split="test",
        transform=Transform(),
        cohort_selection_options=CohortSelectionOptions(
            fold=fold,
            n_folds=n_folds,
            min_involvement=None,
            remove_benign_from_positive_patients=False,
            benign_to_cancer_ratio=None,
        ),
    )

    if debug:
        train_ds = torch.utils.data.Subset(train_ds, np.arange(0, 16))
        val_ds = torch.utils.data.Subset(val_ds, np.arange(0, 16))
        test_ds = torch.utils.data.Subset(test_ds, np.arange(0, 16))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
    )

    return train_loader, val_loader, test_loader


def get_model(name):
    match name:
        case "v0":
            from segment_anything import sam_model_registry

            sam_model = sam_model_registry["vit_b"](
                checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
            )
            image_encoder = sam_model.image_encoder
            pool = torch.nn.AdaptiveMaxPool2d((1, 1))
            flatten = torch.nn.Flatten()
            fc = torch.nn.Linear(256, 1)
            model_ = torch.nn.Sequential(image_encoder, pool, flatten, fc)

            class Model(nn.Module):
                def __init__(self, *args, **kwargs) -> None:
                    super().__init__()
                    self.model = model_

                def forward(self, X, mask):
                    return model_(X)

            model = Model()
            model = model.cuda()
            model = torch.compile(model)
            criterion = nn.BCEWithLogitsLoss()

        case "v1":
            model = MedSAMClassifierBackboneOnly()
            model = model.cuda()
            model = torch.compile(model)
            criterion = nn.BCEWithLogitsLoss()

        case "v0_adapters_64":
            from models import (
                wrap_image_encoder_with_adapter,
                freeze_non_adapter_layers,
            )

            from segment_anything import sam_model_registry

            sam_model = sam_model_registry["vit_b"](
                checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
            )
            image_encoder = sam_model.image_encoder
            image_encoder = wrap_image_encoder_with_adapter(
                image_encoder, adapter_dim=64
            )
            image_encoder = freeze_non_adapter_layers(image_encoder)
            pool = torch.nn.AdaptiveMaxPool2d((1, 1))
            flatten = torch.nn.Flatten()
            fc = torch.nn.Linear(256, 1)
            model_ = torch.nn.Sequential(image_encoder, pool, flatten, fc)

            class Model(nn.Module):
                def __init__(self, *args, **kwargs) -> None:
                    super().__init__()
                    self.model = model_

                def forward(self, X, mask):
                    return model_(X)

            model = Model()
            model = model.cuda()
            model = torch.compile(model)
            criterion = nn.BCEWithLogitsLoss()

        case "v1_adapters_128":
            from models import (
                wrap_image_encoder_with_adapter,
                freeze_non_adapter_layers,
            )

            model = MedSAMClassifierBackboneOnly(freeze_encoder=True)
            model.image_encoder = wrap_image_encoder_with_adapter(
                model.image_encoder, adapter_dim=128
            )
            model = torch.compile(model)
            criterion = nn.BCEWithLogitsLoss()

        case _:
            raise NotImplementedError

    return model, criterion


class MaskedFeatureSelector(nn.Module):
    def forward(self, X, mask, label=None):
        B, C, H, W = X.shape

        batch_idx = torch.arange(B, device=X.device)
        batch_idx = repeat(batch_idx, "b -> b c h w", c=C, h=H, w=W)
        if label is not None:
            label = repeat(label, "b -> b c h w", c=C, h=H, w=W)

        mask = torch.nn.functional.interpolate(
            mask, size=(H, W), mode="bilinear", align_corners=False
        )
        mask_flat = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        valid_indices = mask_flat > 0.5

        batch_idx_flat = rearrange(batch_idx, "b c h w -> (b h w) c")[..., 0]
        X_flat = rearrange(X, "b c h w -> (b h w) c")

        return X_flat[valid_indices], batch_idx_flat[valid_indices]


class SelectedFeaturesLoss(nn.Module):
    def forward(self, X, batch_idx, label):
        label_broadcast = torch.zeros_like(X[:, 0])
        for i in batch_idx.unique():
            label_broadcast[batch_idx == i] = label[i]
        return torch.nn.functional.binary_cross_entropy_with_logits(X, label_broadcast)


class MedSAMClassifierBackboneOnly(nn.Module):
    def __init__(self, freeze_encoder=False):
        super().__init__()
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](
            checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
        )
        self.image_encoder = sam_model.image_encoder
        self.masked_feature_selector = MaskedFeatureSelector()
        self.linear_layer = nn.Linear(256, 1)

        self.freeze_encoder = freeze_encoder

    def forward(self, X, mask):
        with torch.set_grad_enabled(not self.freeze_encoder):
            feats = self.image_encoder(X)

        feats, batch_idx = self.masked_feature_selector(feats, mask)

        outputs = self.linear_layer(feats)

        pooled_outputs = []
        for i in batch_idx.unique():
            pooled_outputs.append(outputs[batch_idx == i].mean(dim=0))
        pooled_outputs = torch.stack(pooled_outputs).squeeze(-1)
        return pooled_outputs


class SimpleAttentionPooling(nn.Module): 
    def __init__(self, features_dim):
        super().__init__()
        self.features_dim = features_dim
        self.attention = nn.Linear(features_dim, 1)

    def forward(self, X): 
        B, C, H, W = X.shape 
        X = rearrange(X, "b c h w -> b (h w) c")
        attention_weights = self.attention(X)
        attention_weights = torch.softmax(attention_weights, dim=1)
        X = X * attention_weights
        X = X.sum(dim=1)
        return X
    



# class MaskedAttentionPooling(nn.Module): 
#     def __init__(self, features_dim, mask_threshold=0.5):
#         super().__init__()
#         self.features_dim = features_dim
#         self.attention = nn.Linear(features_dim, 1)
#         self.pad_token = nn.Parameter(torch.zeros(features_dim))
#         self.mask_threshold = mask_threshold
# 
#     def forward(self, X, mask):
#         B, C, H, W = X.shape 
# 
#         X = X.permute(0, 2, 3, 1)
#         mask = torch.nn.functional.interpolate(
#             mask, size=(H, W), mode="bilinear", align_corners=False
#         )
#         mask = mask > self.mask_threshold
#         mask = mask.permute(0, 2, 3, 1)
# 
#         outputs = []
#         for i in range(B): 
#             x = X[i]
#             m = mask[i]
#             x = x[m]
#             masked_features = x # masked_features.shape = (N, C)
#             attention_weights = self.attention(masked_features) #(N, 1)
#             attention_weights = torch.softmax(attention_weights, dim=0) #(N, 1)
#             outputs.append(torch.sum(attention_weights * masked_features, dim=0))
# 
#         outputs = torch.stack(outputs)
#         return outputs
# 
# 




if __name__ == "__main__":
    args = parse_args()
    main(args)
