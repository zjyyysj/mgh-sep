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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

import replay_buffer
from replay_buffer import PrioritizedReplayBuffer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_dir="./tmp"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

binary_fields = ['gender','mechvent','re_admission']
norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
    'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
    'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
    'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',
    'PaO2_FiO2','cumulated_balance', 'elixhauser', 'Albumin', 'CO2_mEqL', 'Ionised_Ca','time']
log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',
              'input_total','input_1hourly','output_total','output_1hourly']
feature_fields=binary_fields+norm_fields+log_fields

test_df=pd.read_csv('csv_1h/rl_test_set_1h.csv')

action_map = {}
count = 0
for iv in range(5):
    for vaso in range(5):
        action_map[(iv,vaso)] = count
        count += 1

REWARD_THRESHOLD = 20
noise_std = 0.4
dim=128
ac_dim=25

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, use_cuda, std_init=noise_std):
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
        
        self.linear = nn.Linear(num_inputs, dim)
        
        self.noisy_value1 = NoisyLinear(dim, dim, use_cuda = USE_CUDA)
        self.noisy_value2 = NoisyLinear(dim, 1, use_cuda = USE_CUDA)
        
        self.noisy_advantage1 = NoisyLinear(dim, dim, use_cuda = USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(dim, self.num_outputs, use_cuda = USE_CUDA)
        
        
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
    

def intermediate_reward(state, next_state, c0=-0.025, c1=-0.125, c2=-2):
    mediate_reward = 0
    if abs(state[30] - next_state[30])<1e-6 and next_state[30] > 0:
        mediate_reward = mediate_reward + c0
    
    mediate_reward = mediate_reward - c1 * (state[30] - next_state[30]) - c2 * math.tanh(state[29] - next_state[29])
    
    return mediate_reward

current_model = DuelingDQNnoise(len(feature_fields), ac_dim).to(device)
current_model.load_state_dict(torch.load('model_save/model_params_noise.pkl',map_location='cpu'))
current_model.eval()

a = test_df.copy()
num = np.size(a,0)

pd.unique(a['icustayid'])
np.size(pd.unique(a['icustayid']))

num_action_same_top5 = 0
num_action_same_top3 = 0
num_action_same = 0
idx_action_same = np.zeros(num)

for i in range(num):
    
    cur_state = states_test[i]
    action = actions_test[i].item()
    
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    
    if action in q_values_test.sort(descending=True)[1][0:5] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
        num_action_same_top5 = num_action_same_top5 + 1
        idx_action_same[i]=5
        if action in q_values_test.sort(descending=True)[1][0:3] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
            num_action_same_top3 = num_action_same_top3 + 1
            idx_action_same[i]=3
            if abs(action - torch.max(q_values_test,0)[1].item()) == 0:
                num_action_same = num_action_same + 1
                idx_action_same[i]=1

ratio = [num_action_same/num, num_action_same_top3/num, num_action_same_top5/num]
print(ratio)

x_opt_noisetop5 = []
y_opt_noisetop5 = []
x_opt_noisetop3 = []
y_opt_noisetop3 = []
x_opt_noisetop1 = []
y_opt_noisetop1 = []

for i in range(num):
    cur_state = states_test[i]
    
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)

    for j in range(5):
        topaction_opt =  q_values_test.sort(descending=True)[1][j].item()
        if j < 1:
            x_opt_noisetop1.append(topaction_opt // 5)
            y_opt_noisetop1.append(topaction_opt % 5)
        if j < 3:
            x_opt_noisetop3.append(topaction_opt // 5)
            y_opt_noisetop3.append(topaction_opt % 5)
        if j < 5:
            x_opt_noisetop5.append(topaction_opt // 5)
            y_opt_noisetop5.append(topaction_opt % 5)

# noise phase-gated net
phase_model1 = DuelingDQNnoise(len(feature_fields), ac_dim).to(device)
phase_model1.load_state_dict(torch.load('./model_save/model_params_phase1.pkl',map_location='cpu'))
phase_model1.eval()
phase_model2 = DuelingDQNnoise(len(feature_fields), ac_dim).to(device)
phase_model2.load_state_dict(torch.load('./model_save/model_params_phase2.pkl',map_location='cpu'))
phase_model2.eval()

def shock_score(state):
    feature_index = [32, 7, 5, 24, 10, 25, 42] 
    feature_weight= [3.66, -1.36, -0.78, -0.61, 0.57, -0.56, 0.5]
    score = 0
    num = np.size(feature_index)
    for i in range(num):
        score = score + state[feature_index[i]]*feature_weight[i]
    
    return score

score_p = []
for i in range(num):
    score = shock_score(states_test[i])
    score_p.append(score)

score = np.array(score_p)

def phase_cut(score):
    phase = 1
    if score < 0.9: # TREWScore 60 percentile
        phase = 2
    
    return phase        

phase_action_same_top5 = 0
phase_action_same_top3 = 0
phase_action_same = 0
phase1_test = 0
phase_idx_action_same = np.zeros(num)

for i in range(num):
    
    cur_state = a.loc[i,feature_fields]
    iv = int(a.loc[i, 'iv_input'])
    vaso = int(a.loc[i, 'vaso_input'])
    action = action_map[iv,vaso]
    
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    score = shock_score(cur_state)
    if phase_cut(score) == 1:
        phase1_test = phase1_test + 1
        q_values_test = phase_model1(state)
    else:
        q_values_test = phase_model2(state)
    
    if action in q_values_test.sort(descending=True)[1][0:5] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
        phase_action_same_top5 = phase_action_same_top5 + 1
        phase_idx_action_same[i]=5
        if action in q_values_test.sort(descending=True)[1][0:3] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
            phase_action_same_top3 = phase_action_same_top3 + 1
            phase_idx_action_same[i]=3
            if abs(action - torch.max(q_values_test,0)[1].item()) == 0:
                phase_action_same = phase_action_same + 1
                phase_idx_action_same[i]=1

print(phase1_test/num)

phase1_action_same_top5 = 0
phase1_action_same_top3 = 0
phase1_action_same = 0
phase1_test = 0 
phase1_idx_action_same = np.zeros(num)
phase2_action_same_top5 = 0
phase2_action_same_top3 = 0
phase2_action_same = 0
phase2_test = 0
phase2_idx_action_same = np.zeros(num)


for i in range(num):
    
    cur_state = a.loc[i,feature_fields]
    iv = int(a.loc[i, 'iv_input'])
    vaso = int(a.loc[i, 'vaso_input'])
    action = action_map[iv,vaso]
    
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    score = shock_score(cur_state)
    if phase_cut(score) == 1:
        phase1_test = phase1_test + 1
        q_values_test = phase_model1(state)
        
        if action in q_values_test.sort(descending=True)[1][0:5] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
            phase1_action_same_top5 = phase1_action_same_top5 + 1
            phase1_idx_action_same[i]=5
            if action in q_values_test.sort(descending=True)[1][0:3] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
                phase1_action_same_top3 = phase1_action_same_top3 + 1
                phase1_idx_action_same[i]=3
                if abs(action - torch.max(q_values_test,0)[1].item()) == 0:
                    phase1_action_same = phase1_action_same + 1
                    phase1_idx_action_same[i]=1
        
    else:
        q_values_test = phase_model2(state)
        phase2_test = phase2_test + 1
        
        if action in q_values_test.sort(descending=True)[1][0:5] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
            phase2_action_same_top5 = phase2_action_same_top5 + 1
            phase2_idx_action_same[i]=5
            if action in q_values_test.sort(descending=True)[1][0:3] or abs(action - torch.max(q_values_test,0)[1].item()) == 1 or abs(action - torch.max(q_values_test,0)[1].item()) == 5:
                phase2_action_same_top3 = phase2_action_same_top3 + 1
                phase2_idx_action_same[i]=3
                if abs(action - torch.max(q_values_test,0)[1].item()) == 0:
                    phase2_action_same = phase2_action_same + 1
                    phase2_idx_action_same[i]=1

phase_ratio1 = [phase1_action_same/phase1_test, phase1_action_same_top3/phase1_test, phase1_action_same_top5/phase1_test]
phase_ratio2 = [phase2_action_same/phase2_test, phase2_action_same_top3/phase2_test, phase2_action_same_top5/phase2_test]
print(phase_ratio1,phase_ratio2)
















