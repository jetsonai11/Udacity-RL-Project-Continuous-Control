import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from Model import *
from Utils import *

batch_size = 128
gamma = 0.99
tau = 1e-3
critic_lr = 2e-4
actor_lr = 2e-4
buffer_size = int(1e5)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGagent:
    def __init__(self, state_size, action_size, random_seed):
        
        """
        initialize a network agent
        
        Arguments
        =========
        state_size(int): dimension of state space
        action_size(int): dimension of action space
            
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # Initialize networks and target networks
        self.actor = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.critic = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
     
        # Initialize target networks as copies of original networks            
        self.hard_copy_weights(self.actor_target, self.actor)
        self.hard_copy_weights(self.critic_target, self.critic)
        
        #OU Noise
        self.noise = OUNoise(action_size, random_seed)
        
        # Experienced replay
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)  
        
        # Set up criterion and optimizers
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def hard_copy_weights(self, target, original):
        """ 
        copy weights from the original to the target network when initializing the agent
        """
        for target_param, param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_(param.data)
        
        
    def step(self, state, action, reward, next_state, done):
        """
        Save experience in the replay buffer, and sample random experience tuples to learn.
        """
        # Store experience tuples
        self.memory.add(state, action, reward, next_state, done)

        # Learn (if enough samples are available in memory)
        if len(self.memory) > batch_size:
            experiences = self.memory.sample()
            self.update(experiences, gamma)
            
    
    def act(self, state, add_noise=True):
        """
        Return actions for given state under the policy that is currently following
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    

    def reset(self):
        self.noise.reset()
        
    
    def update(self, experiences, gamma):
        """
        update policy and value using random batch of experience tuples
        
        Arguments
        =========
        experiences(Tuple[torch.Tensor]): tuple of state, action, reward, next_state and dones
        gamma(float): the discount factor
        
        """
        states, actions, rewards, next_states, dones = experiences
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # -------------------------------------- Update Critic --------------------------------------#
        ##############################################################################################
        # Define critic loss (Note: the original Q values are calculated with the original networks, #
        # while the next-state Q values are calculated with the target network instead. We then use  #
        # minimize the MSE between the original Q-values and the updated Q-values. The updated Q-val #
        # is calculated by using the Bellman Equation.                                               #
        ##############################################################################################

        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime.detach())
        
        # -------------------------------------- Update Actor ---------------------------------------#
        ##############################################################################################
        # Define actor loss. Since we are taking updating the policy with mini-batches of experience # 
        # sampled from the replay buffer, we take the mean of the sum of gradients. And since we are #
        # using gradient ascent, we add the negative sign in front of the equation because of that   #
        # the default in PyTorch is gradient descent.                                                #
        ##############################################################################################
        actions_pred = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, actions_pred).mean()
        
        # Update original networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
   
        # Update target networks using soft update
        self.soft_update(self.critic_target, self.critic, tau)
        self.soft_update(self.actor_target, self.actor, tau)
        
    def soft_update(self, target_network, original_network, tau):
        """
        Define a soft update process for updating the target networks.
        formula: θ_target = τ * θ_original + (1 - τ) * θ_target
        
        Arguments
        =========
        original_network: model where weights will be copied from
        target_network: model where weights will be copied to
        tau(float): the interpolation parameter
        
        """
        for target_param, param in zip(target_network.parameters(), original_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)