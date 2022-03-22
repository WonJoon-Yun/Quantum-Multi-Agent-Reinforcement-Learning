import numpy as np
import copy

class ENV :
    def __init__(self):
        self.A_EDGE               = 0.3
        self.ACTION_SPACE         = np.array([0.1,0.2])
        self.DEPARTURE_CLOUD      = np.array([1,1]) * 0.3
        self.CAPACITY_EDGE        = np.array([1,1,1,1])
        self.CAPACITY_CLOUD       = np.array([1,1])
        self.N_AGENTS             = 4
        self.N_CLOUD              = 2
        self.T_MAX                = 30
        self.time                 = 0
        self.q_prev_edge          = np.array([0.8,0.8,0.2,0.2])
        self.q_curr_edge          = np.array([0.8,0.8,0.2,0.2])
        self.q_curr_cloud         = np.array([0.5,0.5])
        self.action_dim           = 4
        self.state_dim            = 4
        
    def get_state(self):
        return np.array([ np.hstack([self.q_prev_edge[i],self.q_curr_edge[i],self.q_curr_cloud]) for i in range(self.N_AGENTS)])
    
    def step(self, actions):
        self.q_prev_edge     = np.copy(self.q_curr_edge)
        self.indicator_edge = np.zeros(self.N_AGENTS)
        self.actions        = np.zeros(self.N_AGENTS)
        
        for n in range(self.N_AGENTS):
            if actions[n] == 0:
                self.indicator_edge[n] = 0
                self.actions[n]        = self.ACTION_SPACE[0]
            elif actions[n] == 1:  
                self.indicator_edge[n] = 0
                self.actions[n] = self.ACTION_SPACE[1]
            elif actions[n] ==  2:
                self.indicator_edge[n] = 1
                self.actions[n] = self.ACTION_SPACE[0]
            elif actions[n] == 3:  
                self.indicator_edge[n] = 1
                self.actions[n] = self.ACTION_SPACE[1]
        
        
        self.arrival_edge    = np.random.rand( self.N_AGENTS)* self.A_EDGE
        self.q_curr_edge     = self.q_curr_edge +  self.arrival_edge  - self.actions
        self.q_loss_edge     = ( self.q_curr_edge >  self.CAPACITY_EDGE) * np.abs(self.q_curr_edge- self.CAPACITY_EDGE)
        self.q_stall_edge    = (self.q_curr_edge < 0) * np.abs(self.q_curr_edge)
        self.q_curr_edge     = np.clip(self.q_curr_edge,0,1)

        self.actions         = self.actions - self.q_stall_edge

        self.arrival_cloud   = np.array([ ((self.indicator_edge== i) * self.actions).sum() for i in range(self.N_CLOUD)])
        self.q_curr_cloud    = self.q_curr_cloud + self.arrival_cloud - self.DEPARTURE_CLOUD
        self.q_loss_cloud    = (self.q_curr_cloud > self.CAPACITY_CLOUD) * np.abs(self.q_curr_cloud- self.CAPACITY_CLOUD)
        self.q_stall_cloud   = (self.q_curr_cloud<0) * np.abs(self.q_curr_cloud)
        self.departure_cloud = self.DEPARTURE_CLOUD - self.q_stall_cloud
        self.q_curr_cloud    = np.clip(self.q_curr_cloud, 0, 1)
        
        
        comm_r     =  1*self.q_loss_cloud.sum()  +  4*self.q_stall_cloud.sum()
        rewards      = -comm_r
        
        next_state = self.get_state()
        self.time += 1
        
        if self.time >=self.T_MAX:
            done = True
        else:
            done = False
            
        return rewards,  next_state, done

    def reset(self):
        self.q_prev_edge     = np.array([0.8,0.8,0.2,0.2])
        self.q_curr_edge     = np.array([0.8,0.8,0.2,0.2])
        self.q_curr_cloud    = np.array([0.5,0.5])
        return self.get_state()
