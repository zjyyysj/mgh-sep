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

test_df=pd.read_csv('./csv/rl_test_set.csv')

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

a = test_df.copy()
ind=41301

states_test = None
actions_test = None
patients_test = None
for i in range(ind):
    cur_state = a.loc[i,feature_fields]
    iv = int(a.loc[i, 'iv_input'])
    vaso = int(a.loc[i, 'vaso_input'])
    action = action_map[iv,vaso]
    
    if states_test is None:
        states_test = copy.deepcopy(cur_state)
    else:
        states_test = np.vstack((states_test,cur_state))

    if actions_test is None:
        actions_test = [action]
    else:
        actions_test = np.vstack((actions_test,action))

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
states = Variable(torch.FloatTensor(np.float32(states_test)))
actions = Variable(torch.LongTensor(actions_test))

num_action_same = 0
idx_action_same = []
return_cli_opt = []
return_cli_zero = []
return_cli_random = []
return_opt = []
return_cli =[]
return_zero = []
return_random = []
q_random = 0
for i in range(41301):
    q_values_test = current_model(states[i])
    if actions[i].item() in q_values_test.sort(descending=True)[1][0:5] or abs(actions[i].item() - torch.max(q_values_test,0)[1].item()) == 1 or abs(actions[i].item() - torch.max(q_values_test,0)[1].item()) == 5:
        num_action_same = num_action_same + 1
        idx_action_same.append(i)
    return_opt.append(torch.max(q_values_test,0)[0])
    return_cli.append(q_values_test[actions[i].item()])
    return_zero.append(q_values_test[0])
    for i in range(10):
        q_random+=(q_values_test[random.choice(range(25))])
    return_random.append(q_random/10)
    return_cli_opt.append(torch.max(q_values_test,0)[0] - q_values_test[actions[i].item()]) 
    return_cli_zero.append(q_values_test[actions[i].item()] - q_values_test[0]) 
    return_cli_random.append(q_values_test[actions[i].item()] - q_random/10)
    q_random = 0

patient_num = np.size(pd.unique(a['icustayid']))
death_inhosp = np.size(a.query('reward==-15'),0)
death_90d = np.size(a.query('reward_new==-15'),0)
print(death_inhosp/patient_num,death_90d/patient_num)

patient_num = np.size(pd.unique(a['icustayid']))

patient_close_opt = np.zeros(patient_num)
patient_close_opt_iv = np.zeros(patient_num)
patient_close_opt_vaso = np.zeros(patient_num)

patient_hosp = np.zeros(patient_num)
patient_mor90 = np.zeros(patient_num)

sq_id = 0
for i in range(patient_num):
    
    icuid = pd.unique(a['icustayid'])[i]
    maxbloc = pd.value_counts(a['icustayid'])[icuid] 
    
    close_action_sum = 0
    close_iv_sum = 0
    close_vaso_sum = 0
    
    for j in range(maxbloc):
        
        cur_state = patient.loc[j+sq_id,feature_fields]
        iv = int(patient.loc[j+sq_id, 'iv_input'])
        vaso = int(patient.loc[j+sq_id, 'vaso_input'])
        action = action_map[iv,vaso]

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        action_opt = torch.max(q_values_test,0)[1].item()
        iv_opt = action_opt // 5
        vaso_opt = action_opt % 5
        
        if action in q_values_test.sort(descending=True)[1][0:5] or abs(action - action_opt) == 1 or abs(action - action_opt) == 5:
            close_action_sum = close_action_sum + 1
        
        if abs(iv - iv_opt) <= 1:
            close_iv_sum = close_iv_sum + 1
        
        if abs(vaso - vaso_opt) <= 1:
            close_vaso_sum = close_vaso_sum + 1
     
    if patient.loc[j+sq_id,'reward'] < 0:
        patient_hosp[i] = 1
    if patient.loc[j+sq_id,'reward_new'] < 0:
        patient_mor90[i] = 1
        
    patient_close_opt[i] = close_action_sum/maxbloc
    patient_close_opt_iv[i] = close_iv_sum/maxbloc
    patient_close_opt_vaso[i] = close_vaso_sum/maxbloc
    
    sq_id = sq_id + maxbloc

close_patient = 0
close_patient_hosp = 0
for i in range(patient_num):
    if patient_close_opt[i] >= np.median(patient_close_opt):
        close_patient = close_patient + 1
        close_patient_hosp = close_patient_hosp + patient_hosp[i]
close_patient_mortality_inhosp = close_patient_hosp/close_patient

close_patient = 0
close_patient_mor90 = 0
for i in range(patient_num):
    if patient_close_opt[i] >= np.median(patient_close_opt):
        close_patient = close_patient + 1
        close_patient_mor90 = close_patient_mor90 + patient_mor90[i]
close_patient_mortality_mor90 = close_patient_mor90/close_patient

