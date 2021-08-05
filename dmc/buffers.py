import torch
import typing


# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_buffers(flags, ram=False):
    """
    We create buffers for different devices (i.e., GPU).
    That is, each device has a buffer.
    """
    T = flags.unroll_length
    buffers = []
    for device in range(torch.cuda.device_count()):
        buffers.append({})
        if ram:
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_s=dict(size=(T, 128), dtype=torch.float32),
                obs_a=dict(size=(T, 6), dtype=torch.int8),
            )
        else:
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_s=dict(size=(T, 3, 210, 160), dtype=torch.float32),
                obs_a=dict(size=(T, 6), dtype=torch.int8),
            )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                # share_memory_(): Moves the storage to shared memory.
                _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                _buffers[key].append(_buffer)
        buffers[device] = _buffers
    return buffers


def get_batch(free_queue, full_queue, buffers, flags, lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        # different device has a sole lock
        # multiple threads might have to acquire/wait a lock
        # print("----- {} -- {} : {}".format(os.getpid(), threading.currentThread().ident, id(lock)))
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch
