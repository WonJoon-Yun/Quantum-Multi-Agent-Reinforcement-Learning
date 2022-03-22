import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.optim.lr_scheduler import CosineAnnealingLR
from  torch.distributions import Categorical
from Environment import ENV
from network import QActor, QCritic,ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

n_agents = 4 
s_dim = 4 
a_dim = 4 
gamma = 0.99
n_epochs = 1000
wires_per_block = 2 
a_lr = 1e-3
c_lr = 1e-3
static = 'store_true'

def train_actor(Experience, Pi, td_error, device, optimizer):
    S, A, R, S_Prime = Experience
    for i in range(n_agents):
        log_dist      = torch.log(Pi[i](S[:,i]))
        log_pi_a      = log_dist.gather(-1, A[:,:,i])
        loss          = - (td_error.detach() * log_pi_a).sum()
        
        optimizer[i].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Pi[i].parameters(), 10)
        optimizer[i].step()
        
def train_critic(t, Experience, V, VTarget, optimizer):
    V.train()
    VTarget.eval()
    S, A, R, S_Prime = Experience
    State       = S.reshape(S.shape[0], -1)
    State_Prime = S_Prime.reshape(S_Prime.shape[0], -1)
    R = R.sum(-1)
    v             = V(State)
    vtarget       = VTarget(State_Prime)
    targets       = R + gamma * vtarget 
    td_error      = targets.detach() - v
    loss          = (td_error ** 2).sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(V.parameters(), 10)
    optimizer.step()
    if t > 0 and t % 10 == 0:
        VTarget.load_state_dict(V.state_dict())
    return td_error
        
def train():    
    env  = ENV()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    actors = [QActor().to(device) for _ in range(n_agents)]
    critic = QCritic().to(device)    
    critic_target = copy.deepcopy(critic)
    critic_target.load_state_dict(critic.state_dict())
    replay     = ReplayBuffer(device)
    Qoptimizer = [optim.Adam(actors[i].parameters(), lr=a_lr) for i in range(n_agents)]
    Scheduler  = [CosineAnnealingLR(Qoptimizer[i], T_max=n_epochs) for i in range(n_agents)]
    Voptimizer = optim.Adam(critic.parameters(), lr = c_lr)

    if static:
        for i in range(n_agents):
            
            actors[i].q_layer.static_on(wires_per_block=wires_per_block)

    for epoch in range(1, n_epochs + 1):
        
        s = env.reset()
        s_prime = np.copy(s)
        
        for t in range(env.T_MAX):
            
            s = s_prime.copy()
            a = []
            for i in range(n_agents):
                action_dist  = actors[i](torch.from_numpy(s[i]).unsqueeze(0).to(device, dtype=torch.float))
                a.append(Categorical(action_dist).sample().item())
            a = np.array(a)
            r, s_prime, done = env.step(a)
            transition = [s,a,r,s_prime]
            replay.put_data(transition)
            
        experience = replay.make_batch()
        td_error = train_critic(t=epoch, 
                                Experience= experience, 
                                V = critic, 
                                VTarget= critic_target, 
                                optimizer= Voptimizer)
        
        train_actor(Experience=experience, 
                    Pi= actors, 
                    td_error= td_error, 
                    device= device, 
                    optimizer= Qoptimizer)
        
        for i in range(n_agents):
            Scheduler[i].step()
            
        for i in range(n_agents):
            torch.save(actors[i].state_dict(),'./Qagent.pkl')
        torch.save(critic.state_dict()       ,'./Qcritic.pkl')
            
if __name__ == "__main__":
    train()
