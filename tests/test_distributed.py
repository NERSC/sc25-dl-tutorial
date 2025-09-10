import os
import torch
import torch.distributed as dist
import unittest
import datetime as dt
from utils.rank_generator import RankGenerator
from utils import comm

from networks.vit import MLP, Attention
from parameterized import parameterized

# distributed
from distributed.layers import (
    DistributedMatmul,
    DistributedMLP,
    DistributedAttention,
    DistributedLayerNorm,
)
from distributed.helpers import compute_split_shapes
from distributed.mappings import scatter_to_parallel_region, gather_from_parallel_region


class TestDistributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_size = int(os.getenv("WORLD_SIZE", 1))
        cls.world_rank = int(os.getenv("RANK", 0))
        port = int(os.getenv("MASTER_PORT", 0))
        master_address = os.getenv("MASTER_ADDR")

        # get model parallel sizes
        tp = int(os.getenv("TP", 1))
        cp = int(os.getenv("CP", 1))
        pp = 1
        order = "cp-tp-dp-pp"
        model_parallel_size = tp * cp * pp
        dp = cls.world_size // model_parallel_size
        assert dp >= 1, "ERROR: data parallel wireup failed since dp = {}".format(dp)

        cls.print_to_screen = cls.world_rank == 0
        if cls.print_to_screen:
            print(
                "Distributed unit tests with DP = {}, TP = {}, CP = {}, PP = {}".format(
                    dp, tp, cp, pp
                )
            )

        if torch.cuda.is_available():
            if cls.print_to_screen:
                print("Running test on GPU")
            local_rank = cls.world_rank % torch.cuda.device_count()
            cls.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.manual_seed(333)
            comm_backend = "nccl"
        else:
            if cls.print_to_screen:
                print("Running test on CPU")
            cls.device = torch.device("cpu")
            comm_backend = "gloo"
        torch.manual_seed(333)

        if cls.world_size > 1:
            # create tcp store
            store = dist.TCPStore(
                host_name=master_address,
                port=port,
                world_size=cls.world_size,
                is_master=(cls.world_rank == 0),
                timeout=dt.timedelta(seconds=900),
            )

            # initialize process groups
            dist.init_process_group(
                backend=comm_backend,
                rank=cls.world_rank,
                world_size=cls.world_size,
                store=store,
            )
        else:
            assert False, "Running distributed tests on single GPU"

        # init model + dp groups individually
        comm.init_model_parallel_info(tp=tp, cp=cp, dp=dp, pp=pp, order=order)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)

    def _copy_mlp_weights(self, mlp_layer, mlp_layer_distributed):
        """copy the weights, bias of mlp into the correct shard of mlp_dist"""
        tp = comm.get_size("tp")
        # fc1 is col sharded, fc2 is row sharded (careful: PyT does AW^T)
        embed_local = mlp_layer.fc1.weight.shape[0] // tp
        rank_tp = comm.get_rank("tp")  # which tp rank

        with torch.no_grad():
            # copy sharded weights and biases for fc1
            start = rank_tp * embed_local
            end = start + embed_local
            mlp_layer_distributed.fc1.weight.copy_(mlp_layer.fc1.weight[start:end, :])
            mlp_layer_distributed.fc1.bias.copy_(
                mlp_layer.fc1.bias[start:end].view(1, 1, -1)
            )
            # copy sharded weights for fc2
            mlp_layer_distributed.fc2.weight.copy_(mlp_layer.fc2.weight[:, start:end])
            # copy shared bias for fc2 across all shards
            mlp_layer_distributed.fc2.bias.copy_(mlp_layer.fc2.bias.view(1, 1, -1))

    # tests to run with input parameterization
    # inputs are batch, seq, embed, tolerance
    @parameterized.expand([[4, 1024, 2048, 1e-4], [4, 4050, 2048, 1e-4]])
    def test_distributed_mlp(self, batch, seq, embed, tolerance):
        # set the ops
        mlp_layer = MLP(in_features=embed, hidden_features=4 * embed).to(self.device)
        mlp_layer_distributed = DistributedMLP(
            in_features=embed,
            hidden_features=4 * embed,
            comm_tp_name="tp",
            comm_cp_name="cp",
        ).to(self.device)

        # sync the local and distributed weights
        self._copy_mlp_weights(mlp_layer, mlp_layer_distributed)

        #############################################################
        # non-distributed op
        #############################################################
        # create tensor
        inp = torch.randn((batch, seq, embed), dtype=torch.float32, device=self.device)
        inp.requires_grad = True

        # forward pass
        out = mlp_layer(inp)

        # backward pass
        with torch.no_grad():
            out_grad = torch.randn_like(out)
        out.backward(out_grad)  # vjp with random vector
        inp_grad = inp.grad.clone()

        #############################################################
        # distributed op
        #############################################################
        cp_shapes = compute_split_shapes(seq, comm.get_size("cp"))
        # split the input tensor to get local tensor
        with torch.no_grad():
            inp_local = scatter_to_parallel_region(inp, dim=1, comm_name="cp")
        inp_local.requires_grad = True

        # forward pass local
        out_local = mlp_layer_distributed(inp_local)

        # backward pass local
        with torch.no_grad():
            out_grad_local = scatter_to_parallel_region(out_grad, dim=1, comm_name="cp")
        out_local.backward(out_grad_local)  # vjp with same random local vector
        inp_grad_local = inp_local.grad.clone()

        #############################################################
        # evaluate forward pass
        #############################################################
        with torch.no_grad():
            out_gather = gather_from_parallel_region(
                out_local, dim=1, shapes=cp_shapes, comm_name="cp"
            )
            err = torch.mean(
                torch.norm(out - out_gather, p="fro", dim=(-1, -2))
                / torch.norm(out, p="fro", dim=(-1, -2))
            )
            if self.print_to_screen:
                print(f"final relative error of output in mlp: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        #############################################################
        # evaluate backward pass
        #############################################################
        with torch.no_grad():
            inp_grad_gather = gather_from_parallel_region(
                inp_grad_local, dim=1, shapes=cp_shapes, comm_name="cp"
            )
            err = torch.mean(
                torch.norm(inp_grad - inp_grad_gather, p="fro", dim=(-1, -2))
                / torch.norm(inp_grad, p="fro", dim=(-1, -2))
            )
            if self.print_to_screen:
                print(f"final relative error of gradients in mlp: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

    def _copy_attn_weights(self, attn_layer, attn_layer_distributed):
        """copy the weights, bias of attn into the correct shard of attn_dist"""
        tp = comm.get_size("tp")
        embed = attn_layer.proj.weight.shape[1]
        embed_local = embed // tp
        rank_tp = comm.get_rank("tp")  # which tp rank

        with torch.no_grad():
            # copy sharded weights and biases for qkv
            start = rank_tp * embed_local
            end = start + embed_local
            attn_layer_distributed.q.weight.copy_(attn_layer.q.weight[start:end, :])
            attn_layer_distributed.q.bias.copy_(
                attn_layer.q.bias[start:end].view(1, 1, -1)
            )
            attn_layer_distributed.k.weight.copy_(attn_layer.k.weight[start:end, :])
            attn_layer_distributed.k.bias.copy_(
                attn_layer.k.bias[start:end].view(1, 1, -1)
            )
            attn_layer_distributed.v.weight.copy_(attn_layer.v.weight[start:end, :])
            attn_layer_distributed.v.bias.copy_(
                attn_layer.v.bias[start:end].view(1, 1, -1)
            )
            # copy sharded weights for proj
            start = rank_tp * embed_local
            end = start + embed_local
            attn_layer_distributed.proj.weight.copy_(
                attn_layer.proj.weight[:, start:end]
            )
            attn_layer_distributed.proj.bias.copy_(attn_layer.proj.bias.view(1, 1, -1))

    # tests to run with input parameterization
    # inputs are batch, seq, embed, num_heads, tolerance
    @parameterized.expand([[4, 1024, 2048, 8, 1e-4], [4, 4050, 2048, 8, 1e-4]])
    def test_distributed_attention(self, batch, seq, embed, num_heads, tolerance):
        # set the ops
        attn_layer = Attention(dim=embed, num_heads=num_heads, qkv_bias=True).to(
            self.device
        )
        cp_shapes = compute_split_shapes(seq, comm.get_size("cp"))
        attn_layer_distributed = DistributedAttention(
            dim=embed,
            num_heads=num_heads,
            qkv_bias=True,
            comm_tp_name="tp",
            comm_cp_name="cp",
            cp_shapes=cp_shapes,
        ).to(self.device)

        # sync the local and distributed weights
        self._copy_attn_weights(attn_layer, attn_layer_distributed)

        #############################################################
        # non-distributed op
        #############################################################
        # create tensor
        inp = torch.randn((batch, seq, embed), dtype=torch.float32, device=self.device)
        inp.requires_grad = True

        # forward pass
        out = attn_layer(inp)

        # backward pass
        with torch.no_grad():
            out_grad = torch.randn_like(out)
        out.backward(out_grad)  # vjp with random vector
        inp_grad = inp.grad.clone()

        #############################################################
        # distributed op
        #############################################################
        # split the input tensor to get local tensor
        with torch.no_grad():
            inp_local = scatter_to_parallel_region(inp, dim=1, comm_name="cp")
        inp_local.requires_grad = True

        # forward pass local
        out_local = attn_layer_distributed(inp_local)

        # backward pass local
        with torch.no_grad():
            out_grad_local = scatter_to_parallel_region(out_grad, dim=1, comm_name="cp")
        out_local.backward(out_grad_local)  # vjp with same random local vector
        inp_grad_local = inp_local.grad.clone()

        #############################################################
        # evaluate forward pass
        #############################################################
        with torch.no_grad():
            out_gather = gather_from_parallel_region(
                out_local, dim=1, shapes=cp_shapes, comm_name="cp"
            )
            err = torch.mean(
                torch.norm(out - out_gather, p="fro", dim=(-1, -2))
                / torch.norm(out, p="fro", dim=(-1, -2))
            )
            if self.print_to_screen:
                print(f"final relative error of output in sa: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        #############################################################
        # evaluate backward pass
        #############################################################
        with torch.no_grad():
            inp_grad_gather = gather_from_parallel_region(
                inp_grad_local, dim=1, shapes=cp_shapes, comm_name="cp"
            )
            err = torch.mean(
                torch.norm(inp_grad - inp_grad_gather, p="fro", dim=(-1, -2))
                / torch.norm(inp_grad, p="fro", dim=(-1, -2))
            )
            if self.print_to_screen:
                print(f"final relative error of gradients in sa: {err.item()}")
        self.assertTrue(err.item() <= tolerance)


if __name__ == "__main__":
    unittest.main()
