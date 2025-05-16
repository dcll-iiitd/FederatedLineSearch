import numpy as np
import matplotlib.pyplot as plt
import csv
def reset(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = step_size * gamma**(1. / n_batches_per_epoch)

    elif reset_option == 2:
        step_size = init_step_size

    return step_size

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

w_init = np.array([0.24,1])  # initial weights
# learning_rate = 0.01
num_steps = 10
Rounds = 50


weights = [w_init]
w_global = w_init
w1 = w_init
w2 = w_init

eta_max = 1
eta = 0.2

gamma=4
c=0.01
beta =0.2
def compute_gradient_client1(w1):
 intial_eta=0.9
 w_local = w1
 for i in range(num_steps):
    if i==0:
        step_size = reset(step_size=intial_eta,n_batches_per_epoch=1,
                                    gamma=gamma,
                                    reset_option=1,
                                    init_step_size=intial_eta)
    else: 
        step_size = reset(step_size=step_size,n_batches_per_epoch=1,
                                    gamma=gamma,
                                    reset_option=1,
                                  init_step_size=intial_eta)

    while True:
            step_size *= beta
            w_tilde = w1 - step_size * grad_F1(w1)
            if F1(w_tilde) <= F1(w1) - c * step_size * np.linalg.norm(grad_F1(w1))**2:
                break
    w1 = w_tilde   
 return w1

# for client 2
def compute_gradient_client2(w2):
 intial_eta=0.1
 w_local = w2
 for i in range(num_steps):
    if i==0:
        step_size = reset(step_size=intial_eta,n_batches_per_epoch=1,
                                    gamma=gamma,
                                    reset_option=1,
                                    init_step_size=intial_eta)
    else: 
        step_size = reset(step_size=step_size,n_batches_per_epoch=1,
                                    gamma=gamma,
                                    reset_option=1,
                                    init_step_size=intial_eta)
    while True:
            step_size *= beta
            w_tilde = w2 - step_size * grad_F2(w2)
            if F2(w_tilde) <= F2(w2) - c * step_size * np.linalg.norm(grad_F2(w2))**2:
                break
    w2 = w_tilde   
 return w2
    

for i in range(Rounds):
    temp1 = compute_gradient_client1(w_global)
    temp2 = compute_gradient_client2(w_global)
    w_global = (temp1+temp2)/2
    loss_val = F_global(w_global)
    print(loss_val)
    
    

