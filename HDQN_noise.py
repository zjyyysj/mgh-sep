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

df = pd.read_csv('./rl_train_set_1h.csv')
test_df=pd.read_csv('./rl_test_set_1h.csv')

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
lr=0.0001
alpha=1
beta_start = 0.2
per_epsilon = 1e-5
buffer_size=50000
max_iters = 300000
sample = 2
batch_size = 32
gamma = 0.99

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

def process_sample(sample_size=1, add_reward=True, train=True, eval_type = None):
    if not train:
        if eval_type is None:
            raise Exception('Provide eval_type to process_batch')
        elif eval_type == 'train':
            a = df.copy()
        elif eval_type == 'val':
            a = val_df.copy()
        elif eval_type == 'test':
            a = test_df.copy()
        else:
            raise Exception('Unknown eval_type')
    else:
        a = df.sample(n=sample_size)
        
    states = None
    actions = None
    rewards = None
    next_states = None
    done_flags = None
    for i in a.index:
        cur_state = a.loc[i,feature_fields]
        iv = int(a.loc[i, 'iv_input'])
        vaso = int(a.loc[i, 'vaso_input'])
        action = action_map[iv,vaso]
        reward = a.loc[i,'reward']
        reward_new = a.loc[i,'reward_new']

        if i != df.index[-1]:
            # if not terminal step in trajectory 
            if df.loc[i+1,'bloc'].item() - df.loc[i,'bloc'].item() > 1:
                return process_sample(add_reward=True, train=True, eval_type = None)
            
            if df.loc[i, 'icustayid'] == df.loc[i+1, 'icustayid']:
                next_state = df.loc[i + 1, feature_fields]
                done_flag = 0
            else:
                # trajectory is finished
                next_state = np.zeros(len(cur_state))
                done_flag = 1
        else:
            # last entry in df is the final state of that trajectory
            next_state = np.zeros(len(cur_state))
            done_flag = 1
                
        if states is None:
            states = copy.deepcopy(cur_state)
        else:
            states = np.vstack((states,cur_state))

        if actions is None:
            actions = [action]
        else:
            actions = np.vstack((actions,action))
        
        if add_reward and done_flag == 0:
            reward = reward + intermediate_reward(cur_state, next_state) # add intermediate reward
        if rewards is None:
            rewards = [reward]
        else:
            rewards = np.vstack((rewards,reward))

        if next_states is None:
            next_states = copy.deepcopy(next_state)
        else:
            next_states = np.vstack((next_states,next_state))

        if done_flags is None:
            done_flags = [done_flag]
        else:
            done_flags = np.vstack((done_flags,done_flag))
    
    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

current_model = DuelingDQNnoise(len(feature_fields), ac_dim).to(device)
target_model = DuelingDQNnoise(len(feature_fields), ac_dim).to(device)
    
optimizer = optim.Adam(current_model.parameters(), lr=lr)
replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model, target_model)

def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, dones, weights, indices = replay_buffer.sample(batch_size, beta)
    
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    dones      = Variable(torch.FloatTensor(np.float32(dones)))
    weights    = Variable(torch.FloatTensor(weights))
    
    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)
    
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value = next_q_state_values.gather(1,torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    next_q_value=next_q_value.clamp( -REWARD_THRESHOLD , REWARD_THRESHOLD )
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    
    loss = (q_value - expected_q_value.detach()).pow(2) *weights
    prios = loss + per_epsilon
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

beta_i= lambda i: min(1.0, beta_start + 1 * i * (1.0 - beta_start)/(max_iters * sample)) 
losses = []
all_losses = []

start=timeit.default_timer()
for i in range(max_iters * sample):
    state, action, reward, next_state, done, sampled_df = process_sample(sample_size=1, add_reward=True)
    replay_buffer.push(state, action, reward, next_state, done)
        
    if len(replay_buffer) > batch_size and i % sample == 0:
        beta = beta_i(i)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.data.item())
        all_losses.append(loss.data.item())
    if i%(100 * sample)==0:
        update_target(current_model, target_model)
    if i % (1000 * sample) == 0 and i>0:
        end=timeit.default_timer()
        av_loss = np.array(losses).mean()
        print("iter:",i)
        print((end-start)/(1000 * sample),"s/iter")
        print("Average loss is ", av_loss)
        losses=[]
        start=timeit.default_timer()
