import torch
import torch.distributed as dist
from utils import comm

def init_params_for_shared_weights(model):
    """Helper routine to ensure shared weights are the same after initialization"""
    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():
            if not hasattr(param, "is_shared_mp"):
                # all sharded weights manually annotate this field
                # if weight doesnt have annotation, then it is a shared weight
                # layers like patch-embed, decoder head, pos-embed are fully
                # shared (and not sharded) in this example
                param.is_shared_mp = ["tp-cp"] # only TP-CP implemented for now
                # careful about this stuff..
                param.mark_for_reduction = [] # not all params need special handling

            for comm_group in param.is_shared_mp:
                if comm.get_size(comm_group) > 1:
                    tlist = [
                        torch.empty_like(param)
                        for x in range(comm.get_size(comm_group))
                    ]
                    tlist[comm.get_rank(comm_group)] = param
                    # gather all weights in the comm group
                    dist.all_gather(tlist, param, group=comm.get_group(comm_group))
                    # use weight of rank 0
                    # important to use copy here otherwise the handle gets detaches from the optimizer
                    param.copy_(tlist[0])

# distributed primitives
# helper routine to compute uneven splitting in balanced way:
def compute_split_shapes(size, num_chunks):
    # treat trivial case first
    if num_chunks == 1:
        return [size]
    
    # first, check if we can split using div-up to balance the load: 
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks-1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections

def _reduce(input_, comm_name):
    """All-reduce the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU or if 
    # communicator is not initialized
    if comm.get_size(comm_name) == 1:
        return input_

    # All-reduce.
    dist.all_reduce(input_.contiguous(), group=comm.get_group(comm_name))

    return input_


def split_tensor_along_dim(tensor, dim, num_chunks):
    """Helper routine to split a tensor along a given dimension"""
    if dim >= tensor.dim(): # scattering from dim that doesnt exist
        raise ValueError(f"Error: Scattering along {dim} for a tensor of size {tensor.dim()}")
    if tensor.shape[dim] < num_chunks:
        raise ValueError(f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into {num_chunks} chunks")

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = list(torch.split(tensor, sections, dim=dim))

    return tensor_list


def _split(input_, dim_, comm_name):
    """Split the tensor along dim."""
    # Bypass the function if we are using only 1 GPU or if 
    # communicator is not initialized
    comm_size = comm.get_size(comm_name)
    if comm_size == 1:
        return input_

    # Split along  dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    comm_rank = comm.get_rank(comm_name)
    output = input_list[comm_rank].contiguous()

    return output


def _gather(input_, dim_, shapes_, comm_name):
    """
    Gather tensors and concatinate along the dimension dim_.
    """
    comm_size = comm.get_size(comm_name)
    if (shapes_ is not None) and (len(shapes_) != comm_size):
        raise ValueError(f"Error: passed shapes of size not equal to {comm_size}")
    if dim_ >= input_.dim(): # gathering along dim that doesnt exist
        raise ValueError(f"Error: Gathering along {dim} for a tensor of size {tensor.dim()}")

    # Bypass the function if we are using only 1 GPU or if 
    # communicator is not initialized
    if comm_size == 1:
        return input_

    comm_rank = comm.get_rank(comm_name)
    input_ = input_.contiguous()
    input_shape = list(input_.shape)
    if shapes_ is not None:
        input_list = []
        for src in range(comm_size):
            input_shape[dim_] = shapes_[src]
            input_list.append(torch.empty(input_shape, dtype=input_.dtype, device=input_.device))
    else:
        # assume equal shape on all ranks
        input_list = [torch.empty_like(input_) for _ in range(comm_size)]

    dist.all_gather(input_list, input_, group=comm.get_group(comm_name))
    output = torch.cat(input_list, dim=dim_).contiguous()

    return output

def _reduce_scatter(input_, dim_, comm_name):
    """
    Reduces and scatters along dim_ 
    """
    comm_size = comm.get_size(comm_name)
    if dim_ >= input_.dim(): # RS along dim that doesnt exist
        raise ValueError(f"Error: Reduce-scatter along {dim} for a tensor of size {tensor.dim()}")

    # Bypass the function if we are using only 1 GPU or if 
    # communicator is not initialized
    if comm_size == 1:
        return input_

    comm_rank = comm.get_rank(comm_name)
    input_ = input_.contiguous()

    # Split along  dimension. Make sure the individual tensors are contiguous!
    input_list = [t.contiguous() for t in split_tensor_along_dim(input_, dim_, comm_size)]

    output = torch.empty_like(input_list[comm_rank].contiguous())
    dist.reduce_scatter(output, input_list, group=comm.get_group(comm_name))

    return output
