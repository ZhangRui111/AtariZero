import os
import threading
import time
import timeit
import torch
from torch import multiprocessing as mp

from .models import Model
from .buffers import create_buffers, get_batch
from .utils import create_optimizers, act, learn


def train(flags):
    # Initialize actor models
    models = []
    assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), \
        'The number of actor devices can not exceed the number of available devices'
    for device in range(flags.num_actor_devices):
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models.append(model)

    # Initialize buffers
    buffers = create_buffers(flags)  # each device has three buffers for the three positions.

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = []  # record indices of free/used entries in the buffer
    full_queue = []  # record indices of full/unused entries in the buffer
    for device in range(flags.num_actor_devices):
        # SimpleQueue: an unbounded FIFO queue.
        _free_queue = ctx.SimpleQueue()
        _full_queue = ctx.SimpleQueue()
        free_queue.append(_free_queue)
        full_queue.append(_full_queue)

    # Make all entries in the buffer available for storage
    for device in range(flags.num_actor_devices):
        for m in range(flags.num_buffers):
            free_queue[device].put(m)
            free_queue[device].put(m)
            free_queue[device].put(m)

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizer = create_optimizers(flags, learner_model)  # each position has a sole optimizer.

    T = flags.unroll_length
    B = flags.batch_size
    frames = 0

    # Load models (and more information) if any
    checkpoint_path = os.path.expandvars(os.path.expanduser('%s/%s' % (flags.save_dir, 'model.tar')))
    if flags.load_model and os.path.exists(checkpoint_path):
        checkpoint_states = torch.load(
            checkpoint_path, map_location="cuda:" + str(flags.training_device)
        )
        learner_model.get_model().load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        for device in range(flags.num_actor_devices):
            models[device].get_model().load_state_dict(learner_model.get_model().state_dict())
        frames = checkpoint_states["frames"]
        print("Resuming job from {}".format(checkpoint_path))

    # Starting actor processes to collect transitions for learning
    # There are [flags.num_actor_devices * flags.num_actors] actors in total.
    for device in range(flags.num_actor_devices):
        num_actors = flags.num_actors
        for i in range(num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, local_lock, learn_lock, lock=threading.Lock()):
        """ Thread target for the learning process. """
        # Use the keyword nonlocal to declare that the variable is not local.
        # The nonlocal keyword is used to work with variables inside nested functions,
        # where the variable should not belong to the inner function.
        nonlocal frames  # nonlocal variables
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device], full_queue[device], buffers[device], flags, local_lock)
            _stats = learn(models, learner_model.get_model(), batch, optimizer, flags, learn_lock)

            with lock:
                # Here, all threads share a same lock
                # all threads have to acquire/wait this lock
                # print("----- {} -- {} : {}".format(os.getpid(), threading.currentThread().ident, id(lock)))
                frames += T * B

    threads = []
    local_locks = [threading.Lock() for _ in range(flags.num_actor_devices)]
    learn_lock = threading.Lock()

    # Starting learner processes
    # There are [flags.num_actor_devices * flags.num_threads] actors in total.
    for device in range(flags.num_actor_devices):
        for i in range(flags.num_threads):
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-{}'.format(i),
                args=(i, device, local_locks[device], learn_lock))
            thread.start()
            threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        print("Saving checkpoint to {}".format(checkpoint_path))
        _model = learner_model.get_model()
        # Save more information besides model_state_dict
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save({
            'model_state_dict': _model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'flags': vars(flags),
            'frames': frames,
        }, checkpoint_path)

        # Save the weights for evaluation purpose
        model_weights_dir = os.path.expandvars(os.path.expanduser(
            '%s/%s' % (flags.save_dir, 'weights_' + str(frames) + '.ckpt')))
        torch.save(learner_model.get_model().state_dict(), model_weights_dir)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            time.sleep(5)
            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
    except KeyboardInterrupt:
        pass
    else:
        for thread in threads:
            thread.join()
        print("Learning finished after {} frames.".format(frames))

    checkpoint(frames)
