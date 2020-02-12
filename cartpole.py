import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
from q_learning_agent import QLearningAgent
from gym_runner import GymRunner
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """ saves a transition """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#class DQN(nn.Module):
#    def __init__(self, h, w, outputs):
#        super(DQN, self).__init__()
#        self.


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.stage1 = nn.Linear(4, 12)
        self.stage2 = nn.Linear(12, 12)
        self.stage2 = nn.Linear(12, 12)
        self.stage4 = nn.Linear(12, 2)

    def forward(self, state):
        output = F.relu(self.stage1(state))
        output = F.relu(self.stage2(output))
        output = F.relu(self.stage3(output))
        output = self.stage4(output)
        return output

out_right = torch.tensor([0.99, 0.01])
out_left  = torch.tensor([0.01, 0.99])

if __name__ == "__main__":
    gym = GymRunner("CartPole-v1", "gymresults/cartpole-v1")
    agent = QLearningAgent(4,2)

    gym.train(agent, 1000)
    gym.run(agent, 500)

    #agent.model.save_weights("models/cartpole-v1.h5", overwrite=True)
    #gym.close_and_upload(os.environ['API_KEY'])


#env = gym.make('CartPole-v1')
#observation_space = env.observation_space.shape[0]
#action_space = env.action_space.n
#
#episode = 0
#
#while episode < 100:
#    episode += 1
#    state = env.reset()
#    step = 0
#    while True:
#        step += 1
#        action = env.action_space.sample()
#        env.render()
#        state_next, reward, done, info = env.step(action)
#        print(state_next)
#        if not done:
#            reward = reward
#        else:
#            reward = -reward
#        state = state_next
#
#        if done:
#            print("episode: " + str(episode) + ", step: " + str(step))
#            break
