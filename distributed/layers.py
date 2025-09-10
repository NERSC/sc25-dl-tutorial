import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import comm

from torch.cuda import amp

from networks.helpers import trunc_normal_

# matmul parallel
from distributed.mappings import (
    copy_to_parallel_region,
    gather_from_parallel_region,
    all_gather_from_parallel_region,
    reduce_from_parallel_region,
    reduce_scatter_to_parallel_region
)
from typing import Tuple


class DistributedMatmul(nn.Module):
    """Distributed Matrix Multiply
    Y = XW
    W is sharded in a 1D fashion: either row or col parallel
    W is a (in_dim, out_dim) size matrix when unsharded
    So shape of W is either (in_dim/n, out_dim) or (in_dim, out_dim/n)
    X is assumed sharded similarly to match the dimensions
    comm_act_name is an orthogonal comm used for sharding the activation 
    X using m procs (batch_seq/m, in_dim)
    """

    def __init__(
        self,
        inp_dim,
        out_dim,
        comm_inp_name,
        comm_out_name,
        comm_act_name='cp',
        bias=True,
    ):
        super(DistributedMatmul, self).__init__()

        # get sizes
        self.comm_inp_name = comm_inp_name
        self.comm_out_name = comm_out_name
        comm_inp_size = comm.get_size(self.comm_inp_name)
        comm_out_size = comm.get_size(self.comm_out_name)

        assert (
            not (comm_inp_size > 1 and comm_out_size > 1)
        ), "Error, weights are sharded in a 2D fashion, not supported currently"
        assert (
            inp_dim % comm_inp_size == 0
        ), f"Error, the size of input feature dim ({inp_dim}) has to be evenly divisible by the input feature comm dim ({comm_inp_size})"
        assert (
            out_dim % comm_out_size == 0
        ), f"Error, the size of output feature dim ({out_dim}) has to be evenly divisible by the output feature comm dim ({comm_out_size})"

        # compute reduced dims
        inp_dim_local = inp_dim // comm_inp_size
        out_dim_local = out_dim // comm_out_size

        # parameters
        self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local))
        self.weight.is_shared_mp = [comm_act_name] # weights are sharded in tp but shared across cp
        self.weight.mark_for_reduction = [comm_act_name] # shared weights must be additionally reduced
        if bias:
            self.bias = nn.Parameter(torch.ones(1, 1, out_dim_local))
            # if inp dim of W is sharded, then the bias is shared across this group and also
            # shared in cp grp
            self.bias.is_shared_mp = [self.comm_inp_name, comm_act_name]
            self.bias.mark_for_reduction = [comm_act_name] # shared bias must be additionally reduced

        # init weights
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.weight, std=0.02)
        if hasattr(self, "bias"):
            nn.init.constant_(self.bias, 0.0)

    # since this method is full of custom autograd, it cannot be jitted from torch frontend.
    @torch.jit.ignore
    def forward(self, x):
        x_cp = copy_to_parallel_region(x, self.comm_out_name)
        # don't add bias (else allreduce will add it too often)
        x_loc = F.linear(x_cp, self.weight, bias=None) 
        x_out = reduce_from_parallel_region(x_loc, self.comm_inp_name)
        if hasattr(self, "bias"):
            x_out = x_out + self.bias
        return x_out


class DistributedMLP(nn.Module):
    """Distributed MLP layer
    Currently implements 1D tensor parallelism
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        comm_tp_name='tp',
        comm_cp_name='cp',
        act_layer=nn.GELU,
        drop=0.0,
    ):

        super(DistributedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = DistributedMatmul(
            in_features,
            hidden_features,
            comm_inp_name=None,
            comm_out_name=comm_tp_name,
            comm_act_name=comm_cp_name,
            bias=True,
        )

        self.fc2 = DistributedMatmul(
            hidden_features,
            out_features,
            comm_inp_name=comm_tp_name,
            comm_out_name=None,
            comm_act_name=comm_cp_name,
            bias=True,
        )

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DistributedAttention(nn.Module):
    """Distributed Attention layer"""

    def __init__(
        self,
        dim,
        comm_tp_name='tp',
        comm_cp_name='cp',
        cp_shapes=None,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0
    ):

        super(DistributedAttention, self).__init__()

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        assert (
            num_heads % comm.get_size(comm_tp_name) == 0
        ), "heads are not evenly split across TP model ranks"

        self.num_heads_local = num_heads // comm.get_size(comm_tp_name)
        self.head_dim = dim // self.num_heads
        self.scale = (dim // self.num_heads) ** -0.5
        self.fused_attn = True

        self.comm_tp_name = comm_tp_name
        self.comm_cp_name = comm_cp_name
        self.cp_shapes = cp_shapes

        # qkv is col parallel in the weights
        self.q = DistributedMatmul(dim, dim,
            comm_inp_name=None, 
            comm_out_name=comm_tp_name, 
            bias=qkv_bias,
            comm_act_name=comm_cp_name
        )
        self.k = DistributedMatmul(dim, dim,
            comm_inp_name=None, 
            comm_out_name=comm_tp_name, 
            bias=qkv_bias,
            comm_act_name=comm_cp_name
        )
        self.v = DistributedMatmul(dim, dim,
            comm_inp_name=None, 
            comm_out_name=comm_tp_name, 
            bias=qkv_bias,
            comm_act_name=comm_cp_name
        )
        self.attn_drop = nn.Dropout(attn_drop)

        # proj is row parallel in the weights
        self.proj = DistributedMatmul(
            dim, dim, comm_inp_name=comm_tp_name, comm_out_name=None,
            comm_act_name=comm_cp_name
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # note: N is local sequence shard if CP is on
        B, N, C = x.shape
    
        q = self.q(x).reshape(B, N, self.num_heads_local, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads_local, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads_local, self.head_dim).permute(0, 2, 1, 3)

        k = all_gather_from_parallel_region(k, dim=2, shapes=self.cp_shapes, comm_name=self.comm_cp_name)
        v = all_gather_from_parallel_region(v, dim=2, shapes=self.cp_shapes, comm_name=self.comm_cp_name)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # transpose back
        x = x.transpose(1, 2).reshape(B, N, self.num_heads_local * self.head_dim)

        # this is distributed again
        x = self.proj(x)

        # generally we have to be super careful with dropout layers, since
        # those are normalized over the dropouts. That would need to be reduced across nodes
        x = self.proj_drop(x)

        return x

class DistributedLayerNorm(nn.Module):
    """ 
    Distributed layer norm layer 
    Sequence parallel only
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 bias=True,
                 device=None,
                 dtype=None,
                 comm_tp_name='tp',
                 comm_cp_name='cp'):
        super(DistributedLayerNorm, self).__init__()
         
        self.norm = nn.LayerNorm(normalized_shape,
                                 eps=eps,
                                 elementwise_affine=elementwise_affine,
                                 bias=bias, device=device, dtype=dtype)

        if elementwise_affine:
            # affine weights need additional allreduce and are shared 
            # across all groups
            self.norm.weight.is_shared_mp = [comm_tp_name, comm_cp_name]
            self.norm.weight.mark_for_reduction = [comm_cp_name]
            if bias:
                self.norm.bias.is_shared_mp = [comm_tp_name, comm_cp_name]
                self.norm.bias.mark_for_reduction = [comm_cp_name]


    def forward(self, x):
        return self.norm(x)
