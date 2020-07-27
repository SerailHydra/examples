import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from collections import OrderedDict
import time
import os
import concurrent.futures

def flat_dist_call(tensors, call, log_dir=None, iteration=0, msg=None, extra_args=None):
    pid = os.getpid()
    if log_dir:
        stdlog = open(os.path.join(log_dir, "stdout.{}.txt".format(pid)), "a+")

    flat_dist_call.warn_on_half = True
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)

    if flat_dist_call.warn_on_half:
        if torch.cuda.HalfTensor in buckets:
            print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                  " It is recommended to use the NCCL backend in this case.")
            flat_dist_call.warn_on_half = False

    if msg and log_dir:
        stdlog.write("start:node[flat_dist_call_{}],pid[{}],idx[{}],time[{:.2f}]\n\n".format(msg, pid, iteration, time.time() * 1000 * 1000))

    for tp in buckets:
        bucket = buckets[tp]
        coalesced = _flatten_dense_tensors(bucket)
        print("flat_dist_call {} bytes".format(coalesced.numel() * coalesced.element_size()))
        torch.cuda.synchronize()
        if extra_args is not None:
            call(coalesced, *extra_args)
        else:
            call(coalesced)
        if call is dist.all_reduce:
            coalesced /= dist.get_world_size()

        for buf, synced in zip(bucket, _unflatten_dense_tensors(coalesced, bucket)):
            buf.copy_(synced)
    print("\n")
    if msg and log_dir:
        stdlog.write("end:node[flat_dist_call_{}],pid[{}],idx[{}],time[{:.2f}]\n\n".format(msg, pid, iteration, time.time() * 1000 * 1000))

class PSDataParallel(Module):
    """
    :class:`apex.parallel.DistributedDataParallel` is a module wrapper that enables
    easy multiprocess distributed data parallel training, similar to ``torch.nn.parallel.DistributedDataParallel``.

    :class:`DistributedDataParallel` is designed to work with
    the launch utility script ``apex.parallel.multiproc.py``.
    When used with ``multiproc.py``, :class:`DistributedDataParallel`
    assigns 1 process to each of the available (visible) GPUs on the node.
    Parameters are broadcast across participating processes on initialization, and gradients are
    allreduced and averaged over processes during ``backward()``.

    :class:`DistributedDataParallel` is optimized for use with NCCL.  It achieves high performance by
    overlapping communication with computation during ``backward()`` and bucketing smaller gradient
    transfers to reduce the total number of transfers required.

    :class:`DistributedDataParallel` assumes that your script accepts the command line
    arguments "rank" and "world-size."  It also assumes that your script calls
    ``torch.cuda.set_device(args.rank)`` before creating the model.

    https://github.com/NVIDIA/apex/tree/master/examples/distributed shows detailed usage.
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet shows another example
    that combines :class:`DistributedDataParallel` with mixed precision training.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (Default = 1e7): Minimum number of elements in a communication bucket.
        shared_param (Default = False): If your model uses shared parameters this must be True.  It will disable bucketing of parameters to avoid race conditions.

    """

    def __init__(self, module, world_size, rank, log_dir=None, message_size=10000000, shared_param=False):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False
        self.shared_param = shared_param
        self.rank = rank
        self.world_size = world_size
        self.message_size = message_size
        self.log_dir = log_dir
        self.iteration = -1

        #reference to last iterations parameters to see if anything has changed
        self.param_refs = []

        self.reduction_stream = torch.cuda.Stream()

        self.module = module
        self.param_list = list(self.module.parameters())
        self.recv_grad = []

        print("initializing buffer")
        for param in self.param_list:
            self.recv_grad.append(torch.Tensor(param.data.size()))
        print("recv buffer initialized")

        if dist._backend == dist.dist_backend.NCCL:
            for param in self.param_list:
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."

        self.dist_call_id = 0
        self.create_hooks()

        flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, log_dir=self.log_dir, iteration=0, msg="broadcast", extra_args=(0,) )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


    def create_hooks(self):
        #all reduce gradient hook

        for param_i, param in enumerate(list(self.module.parameters())):
            def wrapper(param_i, param):

                def comm_hook(*unused):
                    if rank == 0:
                        reqs = []
                        for j in range(1, self.world_size):
                            self.executor.submit(dist.recv, self.recv_params[param_i], j)
                            self.executor.submit(torch.add, param.grad.data. self.recv_params[param_i])
                        self.executor.submit(torch.div, param.grad.data. self.world_size)
                        for j in range(1, self.world_size):
                            self.executor.submit(dist.send, param.grad.data, j)
                    else:
                        self.executor.submit(dist.send, param.grad.data, 0)
                        self.executor.submit(dist.recv, param.grad.data, 0)

                if param.requires_grad:
                    param.register_hook(comm_hook)
            wrapper(param_i, param)


    def forward(self, *inputs, **kwargs):
        self.iteration += 1

        return self.module(*inputs, **kwargs)
