import numpy as np
import traceback
import torch

from env.atari_env import create_env
from dmc.env_utils import Environment


Action2Onehot = {0: np.array([1, 0, 0, 0, 0, 0]),
                 1: np.array([0, 1, 0, 0, 0, 0]),
                 2: np.array([0, 0, 1, 0, 0, 0]),
                 3: np.array([0, 0, 0, 1, 0, 0]),
                 4: np.array([0, 0, 0, 0, 1, 0]),
                 5: np.array([0, 0, 0, 0, 0, 1])}


def action2onehot(action):
    return torch.from_numpy(Action2Onehot[action])


def create_optimizers(flags, learner_model):
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha
    )
    return optimizer


def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to synchronize with the main process.
    """
    try:
        print("Device {} Actor {} started.".format(device, i))

        T = flags.unroll_length

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = []
        episode_return_buf = []
        target_buf = []
        obs_s_buf = []
        obs_a_buf = []
        buf_size = 0

        env_output = env.reset()

        while True:
            while True:
                obs_s_buf.append(env_output['obs_s'][0, ...])
                with torch.no_grad():
                    agent_output = model.forward(env_output['obs_s'], env_output['obs_a'], flags=flags)
                action = int(agent_output['action'].cpu().detach().numpy())
                obs_a_buf.append(action2onehot(action))
                env_output = env.step(action)
                buf_size += 1
                if env_output['done']:
                    diff = buf_size - len(target_buf)
                    if diff > 0:
                        done_buf.extend([False for _ in range(diff - 1)])
                        done_buf.append(True)

                        episode_return = env_output['episode_return']
                        episode_return_buf.extend([0.0 for _ in range(diff - 1)])
                        episode_return_buf.append(episode_return)
                        # discount factor is 0, i.e., target_buf is filled with the episode_return
                        target_buf.extend([episode_return for _ in range(diff)])
                    break

            if buf_size > T:
                index = free_queue.get()  # get(): Remove and return thr index of a free entry from the queue.
                if index is None:
                    break
                for t in range(T):
                    buffers['done'][index][t, ...] = done_buf[t]
                    buffers['episode_return'][index][t, ...] = episode_return_buf[t]
                    buffers['target'][index][t, ...] = target_buf[t]
                    buffers['obs_s'][index][t, ...] = obs_s_buf[t]
                    buffers['obs_a'][index][t, ...] = obs_a_buf[t]
                full_queue.put(index)  # put(): Put the index of a full entry into the queue.

                # Free transitions that have been buffered.
                done_buf = done_buf[T:]
                episode_return_buf = episode_return_buf[T:]
                target_buf = target_buf[T:]
                obs_s_buf = obs_s_buf[T:]
                obs_a_buf = obs_a_buf[T:]
                buf_size -= T
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Exception in worker process {}".format(i))
        traceback.print_exc()
        raise e


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def learn(actor_models, model, batch, optimizer, flags, lock):
    """ Perform a learning (optimization) step. """
    device = torch.device('cuda:' + str(flags.training_device))
    obs_s = batch['obs_s'].to(device)
    obs_a = batch['obs_a'].to(device)
    target = batch['target'].to(device)
    episode_returns = batch['episode_return'][batch['done']]

    with lock:
        # multiple threads might have to acquire/wait a lock
        # print("----- {} -- {} : {}".format(os.getpid(), threading.currentThread().ident, id(lock)))
        learner_outputs = model(obs_s, obs_a, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return': torch.mean(episode_returns).item(),
            'loss': loss.item(),
        }
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update actor models from the learn model
        for actor_model in actor_models:
            actor_model.get_model().load_state_dict(model.state_dict())

        return stats
