import torch
from utils import comm

def copy_mlp_weights(mlp_layer, mlp_layer_distributed):
    """copy the weights of mlp into the correct shard of mlp_dist"""
    tp = comm.get_size("tp")
    rank_tp = comm.get_rank("tp")  # which tp rank
    embed_local = mlp_layer.fc1.weight.shape[0] // tp
    start = rank_tp * embed_local
    end = start + embed_local

    if embed_local == mlp_layer_distributed.fc1.weight.shape[0]:
        # model is sharded
        if comm.get_world_rank() == 0:
            print("copying weights for sharded model")
        with torch.no_grad():
            mlp_layer_distributed.fc1.weight.copy_(mlp_layer.fc1.weight[start:end, :])
            mlp_layer_distributed.fc2.weight.copy_(mlp_layer.fc2.weight[:, start:end])

