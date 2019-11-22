import gym

env = gym.make('MountainCar-v0')
env.reset()
for _ in range(10**6):
    env.render()
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        break
