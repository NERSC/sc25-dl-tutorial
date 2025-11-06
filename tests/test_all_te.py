import os
import torch
import torch.distributed as dist
import torch.nn as nn
import unittest
import datetime as dt
from utils.rank_generator import RankGenerator
from utils import comm

from networks.vit_te import VisionTransformer
from parameterized import parameterized
from distributed.helpers import (
    compute_split_shapes,
    init_params_for_shared_weights,
)
from distributed.mappings import (
    scatter_to_parallel_region,
    gather_from_parallel_region,
    init_ddp_model_and_reduction_hooks,
)


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
            cls.local_rank = cls.world_rank % torch.cuda.device_count()
            cls.device = torch.device(f"cuda:{cls.local_rank}")
            torch.cuda.set_device(cls.device)
            torch.cuda.manual_seed(333)
            cls.comm_backend = "nccl"
        else:
            if cls.print_to_screen:
                print("Running test on CPU")
            cls.device = torch.device("cpu")
            cls.comm_backend = "gloo"
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
                backend=cls.comm_backend,
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

    def _get_flattened_grads(self, model):
        """Get flattened gradients from a model"""
        grads = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([], device=self.device)

    def _get_reconstructed_grads(self, model):
        """Get reconstructed gradients from a distributed model, handling shared parameters"""
        tp_size = comm.get_size("tp")
        cp_size = comm.get_size("cp")
        tp_rank = comm.get_rank("tp")
        tp_cp_size = comm.get_size("tp-cp")

        all_grads = []
        col_sharded_layers = ["fc2", "attn.proj"]  # need to gather differently

        if model.pos_embed.grad is not None:
            all_grads.append(model.pos_embed.grad.view(-1))

        # Handle patch embed gradients (shared in both tp and cp)
        if model.patch_embed.proj.weight.grad is not None:
            all_grads.append(model.patch_embed.proj.weight.grad.view(-1))
        if (
            hasattr(model.patch_embed.proj, "bias")
            and model.patch_embed.proj.bias.grad is not None
        ):
            all_grads.append(model.patch_embed.proj.bias.grad.view(-1))

        # Handle transformer block gradients
        for name, param in model.named_parameters():
            if "blocks" in name and param.grad is not None:
                is_shared_tp = hasattr(param, "is_shared_mp") and any(
                    "tp" in group for group in param.is_shared_mp
                )

                if is_shared_tp:
                    all_grads.append(param.grad.view(-1))
                else:
                    # sharded params, so gather them
                    grad = param.grad
                    gathered_grads = [torch.zeros_like(grad) for _ in range(tp_size)]
                    dist.all_gather(gathered_grads, grad, group=comm.get_group("tp"))
                    if any(nm in name for nm in col_sharded_layers):
                        grad = torch.cat(gathered_grads, dim=1).view(-1)
                    else:
                        grad = torch.cat([g.view(-1) for g in gathered_grads])
                    all_grads.append(grad)

        # Handle norm gradients (shared in both tp-cp)
        if model.norm.norm.weight.grad is not None:
            all_grads.append(model.norm.norm.weight.grad.view(-1))
            all_grads.append(model.norm.norm.bias.grad.view(-1))
        # Handle head gradients (shared in both tp-cp)
        if model.head.weight.grad is not None:
            all_grads.append(model.head.weight.grad.view(-1))

        return (
            torch.cat(all_grads) if all_grads else torch.tensor([], device=self.device)
        )

    def _copy_mlp_weights(self, mlp_layer, mlp_layer_distributed):
        """copy the weights, bias of mlp into the correct shard of mlp_dist"""
        tp = comm.get_size("tp")
        # fc1 is col sharded, fc2 is row sharded (careful: PyT does AW^T)
        embed_local = mlp_layer.fc1.weight.shape[0] // tp
        rank_tp = comm.get_rank("tp")  # which tp rank

        # copy sharded weights and biases for fc1
        start = rank_tp * embed_local
        end = start + embed_local
        mlp_layer_distributed.fc1.weight.copy_(mlp_layer.fc1.weight[start:end, :])
        mlp_layer_distributed.fc1.bias.copy_(mlp_layer.fc1.bias[start:end])
        # copy sharded weights for fc2
        mlp_layer_distributed.fc2.weight.copy_(mlp_layer.fc2.weight[:, start:end])
        # copy shared bias for fc2 across all shards
        mlp_layer_distributed.fc2.bias.copy_(mlp_layer.fc2.bias)

    def _copy_attn_weights(self, attn_layer, attn_layer_distributed):
        """copy the weights, bias of attn into the correct shard of attn_dist"""
        tp = comm.get_size("tp")
        embed = attn_layer.proj.weight.shape[1]
        embed_local = embed // tp
        rank_tp = comm.get_rank("tp")  # which tp rank

        # copy sharded weights and biases for qkv
        start = rank_tp * embed_local
        end = start + embed_local
        attn_layer_distributed.q.weight.copy_(attn_layer.q.weight[start:end, :])
        attn_layer_distributed.q.bias.copy_(attn_layer.q.bias[start:end])
        attn_layer_distributed.k.weight.copy_(attn_layer.k.weight[start:end, :])
        attn_layer_distributed.k.bias.copy_(attn_layer.k.bias[start:end])
        attn_layer_distributed.v.weight.copy_(attn_layer.v.weight[start:end, :])
        attn_layer_distributed.v.bias.copy_(attn_layer.v.bias[start:end])
        # copy sharded weights for proj
        start = rank_tp * embed_local
        end = start + embed_local
        attn_layer_distributed.proj.weight.copy_(attn_layer.proj.weight[:, start:end])
        attn_layer_distributed.proj.bias.copy_(attn_layer.proj.bias)

    def _copy_vit_weights(self, model, model_distributed):
        """copy the weights of the model into the correct shard of model_distributed"""
        with torch.no_grad():
            # copy patch embed weights
            model_distributed.patch_embed.proj.weight.copy_(
                model.patch_embed.proj.weight
            )
            model_distributed.patch_embed.proj.bias.copy_(model.patch_embed.proj.bias)
            # copy pos embed weights
            model_distributed.pos_embed.copy_(model.pos_embed)
            for block, block_distributed in zip(model.blocks, model_distributed.blocks):
                self._copy_mlp_weights(block.mlp, block_distributed.mlp)
                self._copy_attn_weights(block.attn, block_distributed.attn)
                # copy norm weights
                block_distributed.norm1.norm.weight.copy_(block.norm1.norm.weight)
                block_distributed.norm1.norm.bias.copy_(block.norm1.norm.bias)
                block_distributed.norm2.norm.weight.copy_(block.norm2.norm.weight)
                block_distributed.norm2.norm.bias.copy_(block.norm2.norm.bias)
            # copy head weights
            model_distributed.norm.norm.weight.copy_(model.norm.norm.weight)
            model_distributed.norm.norm.bias.copy_(model.norm.norm.bias)
            model_distributed.head.weight.copy_(model.head.weight)

    @parameterized.expand(
        [
            [4, 128, 128, 8, 4, 512, 8, 2, 1e-2],
            [4, 360, 720, 8, 20, 512, 8, 2, 1e-2],
        ]
    )
    def test_distributed_model(
        self,
        batch,
        H,
        W,
        patch_size,
        chans,
        embed,
        heads,
        depth,
        tolerance,
    ):
        #############################################################
        # non-distributed op
        #############################################################
        # temporarily remove tp and tp-sp groups for non-distributed version
        old_groups = {}
        comm_groups = list(comm._COMM_GROUPS.keys())
        for old_group in comm_groups:
            old_groups[old_group] = comm._COMM_GROUPS.pop(old_group, None)
        model = VisionTransformer(
            img_size=[H, W],
            patch_size=patch_size,
            in_chans=chans,
            out_chans=chans,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            qkv_bias=True,
        ).to(self.device)
        model.zero_grad(set_to_none=True)

        # create tensor in BHWC format
        inp = torch.randn(
            (batch, chans, H, W),
            dtype=torch.float32,
            device=self.device,
        )
        tar = torch.randn(
            (batch, chans, H, W),
            dtype=torch.float32,
            device=self.device,
        )
        inp.requires_grad = True
        # if you use fp16, don't forget about the gradscalers (not impl in this test)
        autocast_dtype = torch.bfloat16
        # forward pass
        with torch.autocast(device_type=inp.device.type, dtype=autocast_dtype):
            out = model(inp)
            loss = torch.mean((out - tar) ** 2)
        loss.backward()
        non_dist_grads = self._get_flattened_grads(model)

        inp_grad = inp.grad.clone()

        #############################################################
        # distributed op
        #############################################################
        # restore groups for distributed version
        for group_name, group in old_groups.items():
            if group is not None:
                comm._COMM_GROUPS[group_name] = group

        model_distributed = VisionTransformer(
            img_size=[H, W],
            patch_size=patch_size,
            in_chans=chans,
            out_chans=chans,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            qkv_bias=True,
        ).to(self.device)
        model_distributed.zero_grad(set_to_none=True)

        # wrap in DDP
        init_params_for_shared_weights(model_distributed)  # mark shared params
        model_distributed = init_ddp_model_and_reduction_hooks(
            model_distributed,
            device_ids=[self.local_rank],
            output_device=self.device,
        )

        # sync the weights
        self._copy_vit_weights(model, model_distributed.module)

        # forward pass
        with torch.no_grad():
            # dp split that dataloaders take care of usually
            inp_local = scatter_to_parallel_region(inp, dim=0, comm_name="dp")
            tar_local = scatter_to_parallel_region(tar, dim=0, comm_name="dp")
        inp_local.requires_grad = True

        with torch.autocast(device_type=inp.device.type, dtype=autocast_dtype):
            out_local = model_distributed(inp_local)
            loss = torch.mean((out_local - tar_local) ** 2)

        loss.backward()
        inp_grad_local = inp_local.grad.clone()

        ############################################################
        # evaluate forward pass
        ############################################################
        with torch.no_grad():
            # compute error over spatial dimensions
            out_gather = gather_from_parallel_region(
                out_local, dim=0, shapes=None, comm_name="dp"
            )
            err = torch.mean(
                torch.norm(out - out_gather, p=2, dim=(1, 2, 3))
                / torch.norm(out, p=2, dim=(1, 2, 3))
            )
            if self.print_to_screen:
                print(f"final relative error of output in model: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        #############################################################
        # evaluate backward pass
        #############################################################
        with torch.no_grad():
            inp_grad_gather = gather_from_parallel_region(
                inp_grad_local, dim=0, shapes=None, comm_name="dp"
            ) / comm.get_size("dp")
            err = torch.mean(
                torch.norm(inp_grad - inp_grad_gather, p=2, dim=(1, 2, 3))
                / torch.norm(inp_grad, p=2, dim=(1, 2, 3))
            )
            if self.print_to_screen:
                print(f"final relative error of input gradients in model: {err.item()}")
        self.assertTrue(err.item() <= tolerance)
        #############################################################
        # evaluate wgrads
        #############################################################
        dist_grads_full = self._get_reconstructed_grads(model_distributed.module)
        # Compare with the non-distributed gradients
        grad_diff = non_dist_grads - dist_grads_full
        err = grad_diff.norm().item() / non_dist_grads.norm().item()
        if self.print_to_screen:
            print(f"final relative error of weight gradients: {err}")

        self.assertTrue(err <= tolerance)


if __name__ == "__main__":
    unittest.main()
