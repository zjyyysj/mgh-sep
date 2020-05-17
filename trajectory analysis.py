#!/usr/bin/env python
# coding: utf-8

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

df=pd.read_csv('single trajectory/patient_individual.csv')

action_map = {}
count = 0
for iv in range(5):
    for vaso in range(5):
        action_map[(iv,vaso)] = count
        count += 1

REWARD_THRESHOLD = 20

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

a = df.copy()
num = np.size(a,0)
patient_num = np.size(pd.unique(a['icustayid'])) # distinct 601 patients in set

action_table = np.zeros([num, 6],dtype=int)
action_table_top = np.zeros([num, 12],dtype=int)

for i in range(num):
    cur_state = a.loc[i,feature_fields]
    iv = int(a.loc[i, 'iv_input'])
    vaso = int(a.loc[i, 'vaso_input'])
    action = action_map[iv,vaso] 
    
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values = clinician_model(state)
    q_values_noise = current_model(state)
    
    score = shock_score(state)
    if phase_cut(score) == 1:
        q_values_phase = current_model1(state)
    else:
        q_values_phase = current_model2(state)
    
    opt_cli = torch.max(q_values,0)[1].item()
    opt_cli_iv = opt_cli // 5
    opt_cli_vaso = opt_cli % 5
    action_table[i,0] = opt_cli_iv
    action_table[i,1] = opt_cli_vaso
    
    opt_noi = torch.max(q_values_noise,0)[1].item()
    opt_noi_iv = opt_noi // 5
    opt_noi_vaso = opt_noi % 5
    action_table[i,2] = opt_noi_iv
    action_table[i,3] = opt_noi_vaso
    
    opt_phase = torch.max(q_values_phase,0)[1].item()
    opt_phase_iv = opt_phase // 5
    opt_phase_vaso = opt_phase % 5
    action_table[i,4] = opt_phase_iv
    action_table[i,5] = opt_phase_vaso
    
    opt_cli_top3 = q_values.sort(descending=True)[1][0:3].cpu().detach().numpy()
    opt_cli_iv_top3max = max(opt_cli_top3 // 5)
    opt_cli_iv_top3min = min(opt_cli_top3 // 5)
    opt_cli_vaso_top3max = max(opt_cli_top3 % 5)
    opt_cli_vaso_top3min = min(opt_cli_top3 % 5)
    action_table_top[i,0] = opt_cli_iv_top3max
    action_table_top[i,1] = opt_cli_iv_top3min
    action_table_top[i,2] = opt_cli_vaso_top3max
    action_table_top[i,3] = opt_cli_vaso_top3min
    
    opt_noi_top3 = q_values_noise.sort(descending=True)[1][0:3].cpu().detach().numpy()
    opt_noi_iv_top3max = max(opt_noi_top3 // 5)
    opt_noi_iv_top3min = min(opt_noi_top3 // 5)
    opt_noi_vaso_top3max = max(opt_noi_top3 % 5)
    opt_noi_vaso_top3min = min(opt_noi_top3 % 5)
    action_table_top[i,4] = opt_noi_iv_top3max
    action_table_top[i,5] = opt_noi_iv_top3min
    action_table_top[i,6] = opt_noi_vaso_top3max
    action_table_top[i,7] = opt_noi_vaso_top3min
    
    opt_phase_top3 = q_values_phase.sort(descending=True)[1][0:3].cpu().detach().numpy()
    opt_phase_iv_top3max = max(opt_phase_top3 // 5)
    opt_phase_iv_top3min = min(opt_phase_top3 // 5)
    opt_phase_vaso_top3max = max(opt_phase_top3 % 5)
    opt_phase_vaso_top3min = min(opt_phase_top3 % 5)
    action_table_top[i,8] = opt_phase_iv_top3max
    action_table_top[i,9] = opt_phase_iv_top3min
    action_table_top[i,10] = opt_phase_vaso_top3max
    action_table_top[i,11] = opt_phase_vaso_top3min

import scipy.io as io
mat_path = 'single trajectory/target_action.mat'
io.savemat(mat_path, {'target_action': action_table})
mat_path = 'single trajectory/target_top_action.mat'
io.savemat(mat_path, {'target_top_action': action_table_top})





