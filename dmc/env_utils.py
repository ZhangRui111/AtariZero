import numpy as np
import torch


def _format_observation(obs_s, device, env_ram):
    """ A utility function to process observations and move them to CUDA. """
    device = torch.device('cuda:' + str(device))
    obs_s = np.stack([obs_s] * 6, axis=0).astype(np.float32)
    if env_ram:
        obs_s = torch.from_numpy(obs_s).to(device)  # RAM observation
    else:
        obs_s = torch.from_numpy(obs_s).permute(0, 3, 1, 2).to(device)  # RGB observation
    obs_a = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
    obs_a = torch.from_numpy(obs_a.astype(np.float32)).to(device)
    # print(obs_s.shape, obs_a.shape)
    return obs_s, obs_a


class Environment:
    def __init__(self, env, device):
        """ Initialize environment wrapper. """
        self.env = env
        self.device = device
        self.episode_return = None
        self.env_ram = True if 'ram' in env.spec.id else False

    def reset(self):
        obs_s_init = self.env.reset()
        obs_s, obs_a = _format_observation(obs_s_init, self.device, self.env_ram)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        self.episode_return = torch.zeros(1, 1)
        return dict(
            obs_s=obs_s,
            obs_a=obs_a,
            done=initial_done,
            episode_return=self.episode_return,
        )

    def step(self, action):
        obs_s, reward, done, _ = self.env.step(action)

        self.episode_return += reward
        episode_return = self.episode_return

        if done:
            obs_s = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        obs_s, obs_a = _format_observation(obs_s, self.device, self.env_ram)
        done = torch.tensor(done).view(1, 1)

        return dict(
            obs_s=obs_s,
            obs_a=obs_a,
            done=done,
            episode_return=episode_return,
        )

    def close(self):
        self.env.close()
