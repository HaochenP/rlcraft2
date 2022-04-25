from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders
ale = ALEInterface()

from ale_py.roms import Breakout


ale.loadROM(SpaceInvaders)

import gym

env = gym.make('ALE/SpaceInvaders-v5')


observation = env.reset()