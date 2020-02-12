import abc
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QLearningAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(QLearningAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # hyperparameters
        self.gamma = 0.95  # discount rate on future rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995  # the decay of epsilon after each training batch
        self.epsilon_min = 0.1  # the minimum exploration rate permissible
        self.batch_size = 32  # maximum size of the batches sampled from memory

        # agent state
        #self.model = self.build_model()
        self.memory = deque(maxlen=50000)

        # Pytorch network
        self.stage1 = nn.Linear(state_size, 12)
        self.stage2 = nn.Linear(12, 12)
        self.stage3 = nn.Linear(12, 12)
        self.stage4 = nn.Linear(12, action_size)
        
        # optimizer
        self.optimizer = optim.RMSprop(self.parameters())

    def forward(self, state):
        output = F.relu(self.stage1(state))
        output = F.relu(self.stage2(output))
        output = F.relu(self.stage3(output))
        output = self.stage4(output)
        return output

    def select_action(self, state, do_train=True):
        if do_train and np.random.rand() <= self.epsilon:
            # We are doing a random action for "exploration"
            return random.randrange(self.action_size)
        # We are picking the best action based on the model
        #TODO this needs to return the one hot output of the network for the state
        return np.argmax(self.model.predict(state)[0])

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        #TODO it looks like this is where the training actually happens
        #TODO start here
        #TODO self.model.fit is the training part
        minibatch = random.sample(self.memory, self.batch_size)

        # State is the current state
        # Action is the actrion picked for this state
        # Reward is the calculated reward for this action at this state
        # Next_state is the next state as a result of picking this action at this state
        # done is done
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Target is reward plus the time-discounted best reward of our prediction (best reward is the max of network output)
                next_state = torch.from_numpy(next_state)
                next_state = next_state.type(torch.FloatTensor)
                print("Next state")
                print(next_state)
                print(type(next_state))
                intermediate = self.forward(next_state)
                # This is a double and needs to be a float....
                print("intermediate: ")
                print(intermediate)
                target = (reward + self.gamma *
                          # forward(state) should be a tuple of [reward-left, reward-right]
                          # So this is really just the max of the output, forget the axis nonsense...
                          #np.amax(self.forward(next_state)[0]))
                          np.amax(intermediate))

            #target_f = self.model.predict(state)
            #target_f[0][action] = target
            #self.model.fit(state, target_f, epochs=1, verbose=0)
            #That means we only want to update value of THE ACTION that we choose in the experience, target is estimate value (reward and future value) of THE ACTION, and target_f is the current value, notice that loss function is 'mse', and target_f is the output of state, so in loss calculation, only the index of THE ACTION is not 0 because it's value is covered by target in target_f[0][action] = target, may this help you.

            # Get the predicted output of this state
            target_f = self.forward(state)

            # We have a better prediction now:
            # Actual reward + disounted(predicted(reward)) vs
            # Predicted reward + discounted(predicted(reward))
            #TODO for this to work, action needs to be an index into the action space
            target_f[action] = target

            # Train the network with this new data
            # i.e. compute loss between prediction and actual
            loss = F.smooth_l1_loss(self.forward(state), target_f)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in self.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
