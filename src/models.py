import torch
from torch import nn
from segment_anything.modeling.common import MLPBlock
from segment_anything.modeling.image_encoder import ImageEncoderViT, Attention, Block


class Adapter(nn.Module):
    def __init__(self, feature_dim, adapter_dim, init_scale=1e-3):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(feature_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, feature_dim)
        self.act = nn.GELU()

        # initializations to make it close to identity function
        nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
        nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.act(self.down_project(x))) + x


class AdapterAttn(nn.Module):
    def __init__(self, attn: Attention, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()
        self.attn = attn
        embedding_dim = attn.proj.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.adapter(x)
        return x


class AdapterMLPBlock(nn.Module):
    def __init__(self, mlp: MLPBlock, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()

        self.mlp = mlp
        embedding_dim = mlp.lin1.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.adapter(x)
        return x


def wrap_block_with_adapter(block: Block, adapter_dim: int, init_scale: float = 1e-3):
    block.attn = AdapterAttn(block.attn, adapter_dim, init_scale=init_scale)
    block.mlp = AdapterMLPBlock(block.mlp, adapter_dim, init_scale=init_scale)
    return block


def wrap_image_encoder_with_adapter(
    image_encoder: ImageEncoderViT, adapter_dim: int, init_scale: float = 1e-3
):
    new_blocks = torch.nn.ModuleList()
    for block in image_encoder.blocks:
        new_block = wrap_block_with_adapter(block, adapter_dim, init_scale=init_scale)
        new_blocks.append(new_block)

    image_encoder.blocks = new_blocks

    return image_encoder


def freeze_non_adapter_layers(model: nn.Module):
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False

    return model
