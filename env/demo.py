import gym

env = gym.make('MsPacman-v0')

for i_episode in range(20):
    step_episode = 0
    obs = env.reset()
    while True:
        # env.render()
        action = env.action_space.sample()  # discrete 0~8
        obs_, reward, done, info = env.step(action)
        step_episode += 1
        print("Episode: {:2d} Step: {:4d} -- Action: {} Reward: {}".format(
            i_episode, step_episode, action, reward))
        if done:
            break

env.close()
