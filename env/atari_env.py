import gym


def create_env(flags):
    env = gym.make(flags.env)
    return env
