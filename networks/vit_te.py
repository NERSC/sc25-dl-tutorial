import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial
from networks.helpers import DropPath, trunc_normal_

# mp stuff
from utils import comm
from distributed.layers import (
    DistributedMatmul,
    DistributedMLP,
    DistributedAttention,
    DistributedLayerNorm,
)
from distributed.helpers import compute_split_shapes
from distributed.mappings import scatter_to_parallel_region, gather_from_parallel_region

import transformer_engine.pytorch as te


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if comm.get_size("tp-cp") > 1:
            tp_group = comm.get_group("tp")
            tp_size = comm.get_size("tp")
        else:
            tp_group = None
            tp_size = 1

        self.fc1 = te.Linear(
            in_features,
            hidden_features,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.act = act_layer()
        self.fc2 = te.Linear(
            hidden_features,
            out_features,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="row" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.drop = nn.Dropout(drop)

        # mark weights
        self.fc1.weight.is_shared_mp = ["cp"]
        self.fc1.weight.mark_for_reduction = ["cp"]
        self.fc1.bias.is_shared_mp = ["cp"]
        self.fc1.bias.mark_for_reduction = ["cp"]
        self.fc2.weight.is_shared_mp = ["cp"]
        self.fc2.weight.mark_for_reduction = ["cp"]
        self.fc2.bias.is_shared_mp = [
            "tp-cp"
        ]  # careful : output has joined, so shared across tp-cp
        self.fc2.bias.mark_for_reduction = ["cp"]

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # do we need this? will te default to sequential from the size?
        if comm.get_size("tp-cp") > 1:
            tp_group = comm.get_group("tp")
            cp_group = comm.get_group("cp")
            tp_size = comm.get_size("tp")
            cp_size = comm.get_size("cp")
            cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
            cp_stream = torch.cuda.Stream()
        else:
            tp_group = None
            cp_group = None
            tp_size = 1
            cp_size = 1
            cp_ranks = []
            cp_stream = None

        # tp shards heads
        self.num_heads_local = num_heads // tp_size

        # should we just use MHA and ditch QKV proj?
        self.q = te.Linear(
            dim,
            dim,
            bias=qkv_bias,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.k = te.Linear(
            dim,
            dim,
            bias=qkv_bias,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.v = te.Linear(
            dim,
            dim,
            bias=qkv_bias,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )

        # TE's attention
        self.attention = te.DotProductAttention(
            num_attention_heads=self.num_heads,
            kv_channels=self.head_dim,
            attention_dropout=attn_drop,
            attn_mask_type="no_mask",
            attention_type="self",
            qkv_format="bshd",
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            cp_group=cp_group,
            cp_global_ranks=cp_ranks,
            cp_comm_type="p2p",
            cp_stream=cp_stream,
        )

        self.proj = te.Linear(
            dim,
            dim,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="row" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.proj_drop = nn.Dropout(proj_drop)

        # mark weights
        self.q.weight.is_shared_mp = ["cp"]
        self.q.weight.mark_for_reduction = ["cp"]
        self.q.bias.is_shared_mp = ["cp"]
        self.q.bias.mark_for_reduction = ["cp"]
        self.k.weight.is_shared_mp = ["cp"]
        self.k.weight.mark_for_reduction = ["cp"]
        self.k.bias.is_shared_mp = ["cp"]
        self.k.bias.mark_for_reduction = ["cp"]
        self.v.weight.is_shared_mp = ["cp"]
        self.v.weight.mark_for_reduction = ["cp"]
        self.v.bias.is_shared_mp = ["cp"]
        self.v.bias.mark_for_reduction = ["cp"]
        self.proj.weight.is_shared_mp = ["cp"]
        self.proj.weight.mark_for_reduction = ["cp"]
        self.proj.bias.is_shared_mp = [
            "tp-cp"
        ]  # careful : output has joined, so shared across tp-cp
        self.proj.bias.mark_for_reduction = ["cp"]

    def _pad(self, x):
        B, N, C = x.shape
        if comm.get_size("tp-cp") > 1:
            N_tensor = torch.tensor([N], dtype=torch.long, device=x.device)
            N_list = [torch.zeros_like(N_tensor) for _ in range(comm.get_size("tp-cp"))]
            torch.distributed.all_gather(N_list, N_tensor, group=comm.get_group("tp-cp"))
            max_N = max([n.item() for n in N_list])
            if max_N % 2 != 0:
                max_N += 1
            if N < max_N:
                pad_size = max_N - N
                padding = torch.zeros(B, pad_size, C, dtype=x.dtype, device=x.device)
                x = torch.cat([x, padding], dim=1)
                N = max_N
        return x

    def forward(self, x):
        B, N, C = x.shape

        x = self._pad(x) # pad if needed for TE to work with odd N or uneven splits
        N_max = x.shape[1]
        # Compute q, k, v projections
        q = self.q(x).reshape(B, N_max, self.num_heads_local, self.head_dim).contiguous()
        k = self.k(x).reshape(B, N_max, self.num_heads_local, self.head_dim).contiguous()
        v = self.v(x).reshape(B, N_max, self.num_heads_local, self.head_dim).contiguous()

        x = self.attention(
                q, k, v,
            )
        x = x[:, :N, :] # unpad if needed
        x = x.reshape(B, N, self.num_heads_local * self.head_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # grid of patches
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        num_patches = self.h * self.w
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=[224, 224],
        patch_size=16,
        in_chans=3,
        out_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_ch = out_chans
        self.drop_rate = drop_rate

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # if context parallel, split the sequence/context
        self.cp_shapes = compute_split_shapes(num_patches, comm.get_size("cp"))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.out_size = self.out_ch * self.patch_size * self.patch_size

        self.head = nn.Linear(embed_dim, self.out_size, bias=False)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        # add positional encoding to each token
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_head(self, x):
        B, _, _ = x.shape  # B x N x embed_dim
        x = x.reshape(B, self.patch_embed.h, self.patch_embed.w, self.embed_dim)
        B, h, w, _ = x.shape

        # apply head
        x = self.head(x)
        x = x.reshape(shape=(B, h, w, self.patch_size, self.patch_size, self.out_ch))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_ch, self.img_size[0], self.img_size[1]))

        return x

    def forward(self, x):
        x = self.prepare_tokens(x)

        # split sequence if cp is on (shape of x is (batch, seq, embed))
        x = scatter_to_parallel_region(x, dim=1, comm_name="cp")

        # if cp is on, each block operates on a sequence shard
        for blk in self.blocks:
            x = blk(x)

        # gather sequence if cp is on
        x = gather_from_parallel_region(x, dim=1, shapes=self.cp_shapes, comm_name="cp")

        x = self.forward_head(x)
        return x


def ViT(params, **kwargs):
    model = VisionTransformer(
        img_size=params.img_size,
        in_chans=params.n_in_channels,
        out_chans=params.n_out_channels,
        patch_size=params.patch_size,
        embed_dim=params.embed_dim,
        depth=params.depth,
        num_heads=params.num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=params.dropout,
        drop_rate=params.dropout,
        attn_drop_rate=params.dropout,
        **kwargs
    )
    return model
