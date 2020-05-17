#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import math
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import pickle
import matplotlib.pyplot as plt
import timeit

import replay_buffer
from replay_buffer import PrioritizedReplayBuffer

REWARD_THRESHOLD = 20
reg_lambda = 5
per_flag = True
beta_start = 0.4

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, use_cuda, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.use_cuda     = use_cuda
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.use_cuda:
            weight_epsilon = self.weight_epsilon.cuda()
            bias_epsilon   = self.bias_epsilon.cuda()
        else:
            weight_epsilon = self.weight_epsilon
            bias_epsilon   = self.bias_epsilon
            
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class DuelingDQNnoise(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQNnoise, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.linear = nn.Linear(num_inputs, 128)
        
        self.noisy_value1 = NoisyLinear(128, 128, use_cuda = USE_CUDA)
        self.noisy_value2 = NoisyLinear(128, 1, use_cuda = USE_CUDA)
        
        self.noisy_advantage1 = NoisyLinear(128, 128, use_cuda = USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(128, self.num_outputs, use_cuda = USE_CUDA)
        
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        x = value + advantage - advantage.mean() 
        return x
    
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
        
    
    def act(self, state):
        with torch.no_grad():
            state   = Variable(torch.FloatTensor(state).unsqueeze(0))
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data.item()
        return action

current_model = DuelingDQNnoise(len(feature_fields), 25).to(device)
current_model.load_state_dict(torch.load('./model_save/model_params_noise.pkl'))
current_model.eval()

def softmax_prob(q_value, beta = 10):
    t = np.exp( beta * (q_value - np.max(q_value)))
    return t/sum(t)

patient_num = np.size(pd.unique(a['icustayid'])) 
gamma = 0.99
sq_id = 0

for i in range(patient_num):
    
    icuid = pd.unique(a['icustayid'])[i]
    maxbloc = pd.value_counts(a['icustayid'])[icuid] 
    patient = a[a['icustayid'] == icuid]
    
    patient_rho = [1]
    patient_reward = [0]
    patient_reward_new = [0]
    patient_gamma = [0]
    patient_q = [0]
    patient_v = []
    
    for j in range(maxbloc):
        
        cur_state = patient.loc[j+sq_id,feature_fields]
        reward = patient.loc[j+sq_id,'reward']
        reward_new = patient.loc[j+sq_id,'reward_new']
        
        if j < (maxbloc-1):
            if patient.loc[j+sq_id+1,'bloc'].item() - patient.loc[j+sq_id,'bloc'].item() > 1:
                continue
            else:
                next_state = patient.loc[j+sq_id+1, feature_fields]
                reward = reward + intermediate_reward(cur_state, next_state)
                reward_new = reward
                
        iv = int(patient.loc[j+sq_id, 'iv_input'])
        vaso = int(patient.loc[j+sq_id, 'vaso_input'])
        action = action_map[iv,vaso]

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        
        q_values = q_values_test.detach().numpy()
        rho = softmax_prob(q_values)[action]
        
        patient_rho.append(rho*patient_rho[-1])
        patient_reward.append(reward)
        patient_reward_new.append(reward_new)
        patient_gamma.append(gamma**j)
        
        q_value = q_values[action]
        patient_q.append(q_value)
        
        value = sum(q_values*softmax_prob(q_values))
        patient_v.append(value)
    
    patient_v.append(0)
    patient_rho = np.array(patient_rho)
    patient_reward = np.array(patient_reward)
    patient_reward_new = np.array(patient_reward_new)
    patient_gamma = np.array(patient_gamma)
    patient_q = np.array(patient_q)
    patient_v = np.array(patient_v)
    
    patient_DR[i] = sum(patient_gamma * patient_rho * patient_reward) - sum(patient_gamma*(patient_rho*patient_q - patient_rho*patient_v))

    sq_id = sq_id + maxbloc

print(np.mean(patient_DR))

