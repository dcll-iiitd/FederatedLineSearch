import numpy as np
import matplotlib.pyplot as plt
import csv

# Define the objective functions
def F1(w):
    return (w[0] + w[1]-3)**2

def F2(w):
    return (w[0] + 2*w[1]-3)**2

# Gradient of the objective functions
def grad_F1(w):
    return np.array([2* ( w[0] + w[1] - 3), 2 * (w[0] + w[1] - 3)])

def grad_F2(w):
    return np.array([2* ( w[0] + 2*w[1] - 3), 4 * ( w[0] + 2*w[1] - 3)])
def F_global(w):
    return 0.5 * (F1(w) + F2(w))
# Gradient descent settings
w_init = np.array([0.24,1])  # initial weights
learning_rate = 0.01
num_steps = 10
Rounds = 50

# Tracking variables
weights = [w_init]
w_global = w_init
w1 = w_init
w2 = w_init
# for client 1
def compute_gradient_client1(w1):
 w_local = w1
 for i in range(num_steps):
    grad = grad_F1(w1) 
    w_local = w_local - learning_rate * grad
 return w1-w_local

    
# for client 2
def compute_gradient_client2(w2):
 w_local = w2
 for i in range(num_steps):
    grad = grad_F2(w2)
    w_local = w_local - learning_rate * grad
 return w2-w_local
    

#For server

for i in range(Rounds):
    temp1 = compute_gradient_client1(w_global)
    temp2 = compute_gradient_client2(w_global)
    numerator = np.linalg.norm(temp1)**2+np.linalg.norm(temp2)**2
    temp = (temp1+temp2)/2
    temp_norm_sq = np.linalg.norm(temp)**2
    denominator = 2.0*2*temp_norm_sq
    ratio = numerator/denominator
    eta_g = max(1,ratio)
    w_global = w_global - eta_g*temp
    loss_val = F_global(w_global)
    print(loss_val)
    

