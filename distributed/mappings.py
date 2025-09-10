import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import comm
from functools import partial

# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# helper functions
from distributed.helpers import (
     _reduce,
     _split,
     _gather,
     _reduce_scatter,
     compute_split_shapes
)


class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_name_):
        """symbolic method"""
        return input_

    @staticmethod
    def forward(ctx, input_, comm_name_): 
        ctx.comm_name = comm_name_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, comm_name=ctx.comm_name), None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_name_):
        """symbolic method"""
        return _reduce(input_, comm_name=comm_name_)

    @staticmethod
    def forward(ctx, input_, comm_name_):  
        return _reduce(input_, comm_name=comm_name_)

    @staticmethod
    def backward(ctx, grad_output): 
        return grad_output, None

class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input and keep it on the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_, comm_name_):
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, ctx.comm_name), None, None, None

        
class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_name_):
        return _split(input_, dim_, comm_name_)

    @staticmethod
    def forward(ctx, input_, dim_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        ctx.split_shapes = compute_split_shapes(
            input_.shape[dim_], comm.get_size(comm_name_)
        )
        return _split(input_, dim_, comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, ctx.split_shapes, ctx.comm_name), None, None

class _ReduceScatterToParallelRegion(torch.autograd.Function):
    """Reduce the inputs and scatter to ranks."""

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_, comm_name_):
        return _reduce_scatter(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def forward(ctx, input_, dim_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        ctx.split_shapes = compute_split_shapes(
            input_.shape[dim_], comm.get_size(comm_name_)
        )
        return _reduce_scatter(input_, dim_, comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, ctx.split_shapes, ctx.comm_name), None, None

class _AllGatherFromParallelRegion(torch.autograd.Function):
    """Reduce the inputs and scatter to ranks."""

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_, comm_name_):
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter(grad_output, ctx.dim, ctx.comm_name), None, None, None

     
# matmul parallel
def copy_to_parallel_region(input_, comm_name):
    """Parallel copy helper"""
    return _CopyToParallelRegion.apply(input_, comm_name)

def reduce_from_parallel_region(input_, comm_name):
    """Parallel reduction helper"""
    return _ReduceFromParallelRegion.apply(input_, comm_name)

def gather_from_parallel_region(input_, dim, shapes, comm_name):
    """Parallel gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, shapes, comm_name)

def all_gather_from_parallel_region(input_, dim, shapes, comm_name):
    """
    Parallel allgather helper that combines reduce-scatter
    in the bwd pass
    """
    return _AllGatherFromParallelRegion.apply(input_, dim, shapes, comm_name)

def reduce_scatter_to_parallel_region(input_, dim, shapes, comm_name):
    """Parallel reduce scatter helper"""
    return _ReduceScatterToParallelRegion.apply(input_, dim, shapes, comm_name)

def scatter_to_parallel_region(input_, dim, comm_name):
    """Parallel scatter helper"""
    return _ScatterToParallelRegion.apply(input_, dim, comm_name)

def init_ddp_model_and_reduction_hooks(model,
                                       device_ids,
                                       output_device,
                                       bucket_cap_mb = 25,
                                       broadcast_buffers = True,
                                       find_unused_parameters = False,
                                       gradient_as_bucket_view = True,
                                       static_graph = False):
    # early exit if we are not in a distributed setting:
    if not dist.is_initialized():
        return model

    need_hooks = False
    if comm.get_size("tp-cp") == 1:
        # no model parallel, just use DDP with 
        # the full world size
        ddp_group = None
    elif comm.get_size("cp") == 1:
        # only cp requires additional allreduce
        # if no cp, use DDP 
        ddp_group = comm.get_group("dp")
    else:
        broadcast_buffers = False
        ddp_group = comm.get_group("dp")
        need_hooks = True # need a grad hook for additional reduce

    model = DistributedDataParallel(model,
                                    device_ids = device_ids,
                                    output_device = output_device,
                                    bucket_cap_mb = bucket_cap_mb,
                                    broadcast_buffers = broadcast_buffers,
                                    find_unused_parameters = find_unused_parameters,
                                    gradient_as_bucket_view = gradient_as_bucket_view,
                                    static_graph = static_graph,
                                    process_group = ddp_group)
    if not need_hooks:
        return model

    # define comm hook because some params need additional allreduce
    def reduction_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        # allreduce everything first
        buff = bucket.buffer()
        # get future for allreduce

        # do the normal DDP all reduce 
        fut = dist.all_reduce(buff, op=dist.ReduceOp.AVG, group=comm.get_group("dp"), async_op=True).get_future()

        # get grads for shared weights
        params = bucket.parameters()
        def grad_reduction(fut, grads, group):
            # reduce remaining gradients
            coalesced = _flatten_dense_tensors(grads)
            # extra allreduce for param wgrads that need it
            dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=comm.get_group(group), async_op=False)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
            return bucket.buffer()

        append_hooks = False
        for group in comm.get_names():
            if group == "dp":
                continue
            grads = []
            for p in params:
                # p needs an allreduce in group
                if group in p.mark_for_reduction:
                    if p.grad is not None:
                        grads.append(p.grad.data)
            if not grads:
                continue
            # append the new reduction functions
            append_hooks = True
            fut = fut.then(partial(grad_reduction, grads=grads, group=group))

        if not append_hooks:
            # this bucket's params only needed the DP allreduce
            # return the bucket directly
            return fut.then(lambda fut: fut.value()[0])
        else:
            # got some additional allreduce chained to fut
            # the grad_reduction will return the bucket
            return fut

    # register model comm hook
    model.register_comm_hook(state=None, hook=reduction_comm_hook)
    return model
