import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """
    Define a Critic(value) network
    
    """
    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize the final layer weights and biases of the critic from a uniform 
        distribution [-3e-3, 3e-3]
        
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state, action):
        """
        Define a forward process that maps state-action pairs to Q-values

        """
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output

    
class Actor(nn.Module):
    """
    Define a Actor(policy) network
    
    """
    
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """
        Initialize the network and parameters.
        
        Parameters
        ==========
        state_size(int): dimension of state space
        action_size(int): dimension of action space
        fc1_unit(int): number of hidden units of hidden layer 1
        fc2_unit(int): number of hidden units of hidden layer 2

        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize the final layer weights and biases of the actor from a uniform 
        distribution [-3e-3, 3e-3]
        
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Define a forward process that maps state to actions
        
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        output = torch.tanh(self.fc3(x))
        
        return output

        
        