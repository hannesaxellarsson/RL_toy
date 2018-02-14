#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 22:29:52 2017
A batch actor critic algorithm

very inspired by homework 2 from UC Berkeley's Deep reinforcement learning
course from fall 2017:
http://rll.berkeley.edu/deeprlcourse/

@author: hannes larsson
"""
import numpy as np
import tensorflow as tf
import gym
import itertools
import matplotlib.pyplot as plt
import math
import time

env = gym.make('CartPole-v0')
#learning_rate = 5e-3
#val_learning_rate = 5e-3
max_learning_rate = 1e-2
min_learning_rate = 1e-6
lr_decay = 100  
num_iterations = 200

batch_steps = 1000 #minimum steps in a batch
max_steps = 200 #maximum number of steps for a run, after this many steps it ends no matter the outcome
gamma = 0.9

losses = []
reward_log = []

#%% 
'''
Get dimensions and stuff
'''
obs_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n

#%% helper functions

def return_neural_net(input_placeholder, scope, layers = 2, width = 64, 
                      output_nodes = ac_dim):
    with tf.variable_scope(scope):
        out = input_placeholder
        for i in range(layers):
            out = tf.layers.dense(inputs = out, units = width, 
                                  activation = tf.nn.tanh)
        return tf.layers.dense(inputs = out, units = output_nodes)
    
def discounted_reward(rewards, gamma):
    return sum(r * gamma ** i for i, r in enumerate(rewards))
        
def discounted_rewards(rewards, gamma):
    return [discounted_reward(rewards[i:], gamma) for i in range(len(rewards))]

def normalize(array, ref_mean = 0, ref_std = 1):
    mean = np.mean(array)
    std = np.std(array)
    return ((array - mean) / std) * ref_std + ref_mean

def returnLearningRate(t, min_learning_rate, max_learning_rate, decay_speed):
    return min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-t/decay_speed)
#%% Build the neural net for policy gradient
'''
learning rate
'''    
sy_lr = tf.placeholder(tf.float32)
'''
Placeholders for observations, actions and advantages
'''
sy_ob_no = tf.placeholder(dtype=tf.float32, shape = (None, obs_dim), name = "ob")
sy_ac_na = tf.placeholder(dtype=tf.int32, shape = (None), name = "ac")
sy_adv_n = tf.placeholder(dtype=tf.float32, shape = (None), name = "adv")

'''
get action given observation
'''
sy_act_probs = return_neural_net(sy_ob_no, "act_probs", layers = 1, width = 32)
sy_sampled_ac = tf.multinomial(logits = sy_act_probs, num_samples = 1)
sy_sampled_ac = tf.squeeze(sy_sampled_ac, axis = 1)
'''
get probabilities for all actions at once, fed into act_placeholder
'''
sy_act_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = sy_ac_na, logits = sy_act_probs, name="action_probailities")

'''
optimize: we want to maximise the mean of the policy times the advantage
'''
loss = -tf.reduce_mean(sy_act_logprobs*sy_adv_n, name="policy_loss")
opt_step = tf.train.AdamOptimizer(sy_lr, name = "opt_step").minimize(loss)

'''
value function approximator (critic)
'''
baseline_prediction = tf.squeeze(return_neural_net(
        sy_ob_no, "baseline", output_nodes=1))

baseline_target = tf.placeholder(dtype=tf.float32, shape = (None), 
                                 name = "baseline_target")

baseline_loss = tf.losses.mean_squared_error(labels = baseline_target,
                                     predictions = baseline_prediction)
bl_opt_step = tf.train.AdamOptimizer(sy_lr).minimize(baseline_loss)

#%%
'''
Start a session
'''
sess = tf.Session()
sess.__enter__()
tf.global_variables_initializer().run()

writer = tf.summary.FileWriter('./ACLogDir', sess.graph)
writer.add_graph(tf.get_default_graph())

#%%
'''
1. Sample trajectories from policy
'''
for i_episode in range(num_iterations):
    print("*********Iteration no. {}*************".format(i_episode+1)) 
    
    #collect paths until we have enough timesteps
    paths = []
    totalSteps = 0
    while True:
        animate = len(paths) == 0
        #make lists to store the observations, actions and rewards in
        ob, ac, rew = [], [], []
        #parameter for linear regression of states
        observation = env.reset()
        for t in itertools.count():
#            if animate:
#                env.render()
#                time.sleep(0.02)
            ob.append(observation)
            #get action from NN
            action = sess.run(sy_sampled_ac, feed_dict={
                    sy_ob_no: observation[None]})
            action = action[0]
            
            #run action and get reward and next observation
            observation, reward, done, _ = env.step(action)
            #store stuff
            ac.append(action)
            rew.append(reward)
        
            if done or t > max_steps:
                nTimeSteps = t+1
                if animate:
                    print("Episode {} finished after {} timesteps".format(i_episode+1, nTimeSteps))
                totalSteps += nTimeSteps
                break
        
        path = {"observation" : np.array(ob),
                    "action" : np.array(ac),
                    "reward" : np.array(rew)}
        paths.append(path)
        if totalSteps >= batch_steps:
            break
        
    '''
    Now simulation is done for this batch. Fit value fucntion and policy:
    '''    
    #put all observations and actions in single numpy arrays
    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    '''
    calculate rewards and stuff
    '''
    rtg = np.concatenate([discounted_rewards(path["reward"], 
                                             gamma) for path in paths])
    q_n = rtg.copy()
    
    #meanRew is just so we can write it out in the console
    meanRew = np.concatenate([[discounted_reward(path["reward"], 1
                                                    )] for path in paths])
    meanRew = np.mean(meanRew)
    reward_log.append(meanRew)
    print("Mean reward::: {}\n".format(meanRew))
    
    '''
    get advantages: A(s_i, a_i) = r(s_i,a_i) + V(s_{i+1}) - V(s_i) 
    '''
    
    V_t = sess.run(baseline_prediction, feed_dict = {
            sy_ob_no : ob_no})
    adv_n = []
    for t in range(len(ac_na) - 2):
        adv_n.append(q_n[t] + V_t[t+1] - V_t[t])
    adv_n.append(q_n[totalSteps-1] - V_t[totalSteps-1])
    adv_n.append(0)
    
    #normalize
    adv_n = normalize(adv_n)
    
    '''
    fit value function approx (critic) to rewards
    '''
    
    learning_rate = returnLearningRate(i_episode, min_learning_rate,
                                       max_learning_rate, lr_decay)
    
    numValueIters= 1
    for i in range(numValueIters):
        sess.run(bl_opt_step, feed_dict = {sy_ob_no: ob_no,
                                       baseline_target: normalize(rtg),
                                       sy_lr: learning_rate})

    '''
    Improve policy
    '''
    trainLoss, _ = sess.run([loss, opt_step], feed_dict = {sy_ob_no: ob_no,
                                    sy_ac_na: ac_na,
                                    sy_adv_n: adv_n,
                                    sy_lr: learning_rate})
    
    #save losses so we can plot them later
    losses.append(trainLoss)
    plt.plot(reward_log)
    
sess.close()
env.render(close=True)
