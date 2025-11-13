import torch
import torch.nn as nn
import argparse
from utils import comm
from tests.helpers import copy_mlp_weights
import os


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
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        if comm.get_world_rank == 0:
            print(f"after fc1: output shape {x.shape}")
        x = self.act(x)
        x = self.fc2(x)
        if comm.get_world_rank == 0:
            print(f"after fc2: output shape {x.shape}")
        return x


class DistributedMLP(nn.Module):
    """Distributed MLP layer
    Currently implements 1D tensor parallelism
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):

        super(DistributedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # TODO: split the hidden_features evenly across the TP ranks
        hidden_features_local = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features_local, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features_local, out_features, bias=False)

    def forward(self, x):
        """ forward is incomplete 
            what syncs do you need? Take a look at distributed/mappings.py for the syncs available.
        """
        # TODO: take care of comms
        x = self.fc1(x)
        if comm.get_world_rank == 0:
            print(f"after fc1: output shape {x.shape}")
        x = self.act(x)
        x = self.fc2(x)
        if comm.get_world_rank == 0:
            print(f"after fc2: output shape {x.shape}")
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    params = {"tp": args.tp}
    
    # do some setup
    comm.init(params)
    local_rank = comm.get_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.manual_seed(333)
    torch.manual_seed(333)

    # create model
    in_features = 1024
    hidden_features = 4096
    model = MLP(in_features=in_features, hidden_features=hidden_features).to(device)
    model_distributed = DistributedMLP(
        in_features=in_features, hidden_features=hidden_features
    ).to(device)
    # make sure the weights are the same for both models
    copy_mlp_weights(model, model_distributed)

    if comm.get_world_rank() == 0:
        print("model:")
        print(model)
        print("model_distributed:")
        print(model_distributed)

    # create input
    x = torch.randn(512, in_features).to(device)
    x_for_distributed = x.clone()
    x.requires_grad = True
    x_for_distributed.requires_grad = True
    if comm.get_world_rank() == 0:
        print(f"input shape {x.shape}")

    # forward passes
    out = model(x)
    out_distributed = model_distributed(x_for_distributed)
    
    # check error
    err = (out - out_distributed).detach()
    out_norm = torch.norm(out.detach())
    if comm.get_world_rank() == 0:
        print(f"forward error: {torch.norm(err) / out_norm}")

    # # backward pass
    with torch.no_grad():
        out_grad = torch.randn_like(out)
    out.backward(out_grad)
    inp_grad = x.grad.clone()
    if comm.get_world_rank() == 0:
        print(f"input grad shape {inp_grad.shape}")

    out_distributed.backward(out_grad)
    inp_grad_distributed = x_for_distributed.grad.clone()
    if comm.get_world_rank() == 0:
        print(f"input grad shape {inp_grad_distributed.shape}")

    # check error
    err = (inp_grad - inp_grad_distributed).detach()
    inp_grad_norm = torch.norm(inp_grad.detach())
    if comm.get_world_rank() == 0:
        print(f"backward error: {torch.norm(err) / inp_grad_norm}")

    torch.distributed.barrier(device_ids=[local_rank])
    torch.distributed.destroy_process_group(None)