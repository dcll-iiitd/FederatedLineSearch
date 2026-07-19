
##################################
### Main: Server side optimization and testing
#################################
# Initiate the NN

from sys import float_info
from utils_data import *
from utils_models import *
from utils_general import *
import time 
import yaml
import csv
import json
import os
from datetime import datetime #added 18 feb

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num_clients', type=int, required=True)
parser.add_argument('--num_participating_clients', type=int, required=True)
parser.add_argument('--num_rounds', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--reset-option', type=int, choices=(0, 1, 2), default=1)
parser.add_argument('--eta-lmax', type=float, default=1.0)
parser.add_argument('--armijo-c', type=float, default=0.1)
parser.add_argument('--measure-kappa', action='store_true', default=False)
parser.add_argument('--kappa-measure-every', type=int, default=10)
parser.add_argument('--deterministic-sls-seed', action='store_true', default=False)

if not 0.0 < args_required.armijo_c < 1.0:
    parser.error("--armijo-c must be strictly between 0 and 1")
args_required = parser.parse_args()

if args_required.kappa_measure_every <= 0:
    parser.error("--kappa-measure-every must be greater than zero")
if args_required.measure_kappa and args_required.algorithm not in ("fedsls", "fedexpsls"):
    parser.error("--measure-kappa is supported only for fedsls and fedexpsls")
if args_required.measure_kappa and args_required.reset_option != 2:
    parser.error("--measure-kappa requires --reset-option 2")
if args_required.deterministic_sls_seed and args_required.algorithm not in ("fedsls", "fedexpsls"):
    parser.error("--deterministic-sls-seed is supported only for fedsls and fedexpsls")



seed = args_required.seed
dataset = args_required.dataset
algorithm = args_required.algorithm
model = args_required.model
num_clients = args_required.num_clients
num_participating_clients = args_required.num_participating_clients
num_rounds = args_required.num_rounds
alpha = args_required.alpha
reset_option = args_required.reset_option
eta_lmax = args_required.eta_lmax
armijo_c = args_required.armijo_c
measure_kappa = args_required.measure_kappa
kappa_measure_every = args_required.kappa_measure_every
deterministic_sls_seed = args_required.deterministic_sls_seed

print_every_test = 5
print_every_train = 5



filename = "results_"+str(seed)+"_"+algorithm+"_"+dataset+"_"+model+"_"+str(num_clients)+"_"+str(num_participating_clients)+"_"+str(num_rounds)+"_"+str(alpha)
filename_txt = filename + ".txt"


if(dataset=='CIFAR100'):
  n_c = 100
elif (dataset == 'EMNIST' or dataset == 'femnist'):
  n_c = 62
else: n_c = 10








np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True



dataset_train,   dataset_test_global = get_dataset(dataset, num_clients, n_c, alpha, True)

kappa_reference_sets = None
kappa_csv_path = None
if measure_kappa:
  kappa_reference_sets = build_kappa_reference_sets(dataset_train, seed)
  eta_token = format(eta_lmax, ".12g").replace(".", "p").replace("-", "m")
  c_token = format(armijo_c, ".12g").replace(".", "p").replace("-", "m")
  os.makedirs("results", exist_ok=True)
  kappa_csv_path = os.path.join(
      "results", f"kappa_measurements_seed{seed}_eta{eta_token}_c{c_token}.csv"
  )
  kappa_metadata_path = os.path.splitext(kappa_csv_path)[0] + ".json"
  with open(kappa_csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "round", "client_id", "local_step", "eta_returned",
        "loss_prev_batch", "loss_curr_batch", "f_ref_prev",
        "f_ref_curr", "grad_sq_norm", "line_search_failed", "seed"
    ])
  with open(kappa_metadata_path, "w") as metadata_file:
    json.dump({
        "seed": seed, "algorithm": algorithm, "reset_option": reset_option,
        "eta_lmax": eta_lmax, "armijo_c": armijo_c,
        "kappa_measure_every": kappa_measure_every,
        "deterministic_sls_seed": deterministic_sls_seed
    }, metadata_file, indent=2)



dict_results = {} ###dictionary to store results for all algorithms


###Default training parameters for all algorithms

args={
"bs":50,   ###batch size
"cp":20,   ### number of local steps
"device":'cuda:0',
"rounds":num_rounds, 
"num_clients": num_clients,
"num_participating_clients":num_participating_clients
}


net_glob_org = get_model(model,n_c).to(args['device'])






algs = [algorithm]


decay=  0.998
max_norm = 10
use_gradient_clipping = True
weight_decay = 0.0001
feddyn_alpha = 0.01  # FedDyn regularization coefficient

if(dataset == 'Shakespeare'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedprox = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedavgm = 1
 
  

  epsilon_fedexp = 0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001
if(dataset == 'femnist'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedprox = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedavgm = 1
 
  

  epsilon_fedexp = 0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001
if(dataset=='MNIST'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedprox = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedavgm = 1
 
  

  epsilon_fedexp = 0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001
  
if(dataset=='CIFAR10'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedprox = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedavgm = 1
 
  

  epsilon_fedexp = 0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001
  
elif(dataset=='CINIC10'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedprox = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedavgm = 1
  
 


  epsilon_fedexp =  0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001

elif(dataset=='CIFAR100'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedprox = 1
  eta_g_fedavgm = 1
  

  epsilon_fedexp = 0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001


elif(dataset=='shakespeare'):
  eta_l_fedavg = 0.1
  eta_l_fedexp = 0.1
  eta_l_scaffold = 0.1
  eta_l_scaffold_exp = 0.1
  eta_l_fedadagrad = 0.1
  eta_l_fedprox = 0.1
  eta_l_fedprox_exp = 0.1
  eta_l_fedadam = 0.1
  eta_l_fedavgm = 0.316
  eta_l_fedavgm_exp = 0.316

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedadagrad = 0.316
  eta_g_fedadam = 0.316
  eta_g_fedprox = 1
  eta_g_fedavgm = 1
 

  epsilon_fedexp = 0.1
  epsilon_scaffold_exp = 0.1
  epsilon_fedprox_exp = 0.1
  epsilon_fedavgm_exp = 0.1


epsilon_fedadagrad = 0.01
epsilon_fedadam = 0.01
  
if(dataset=='EMNIST'):
    epsilon_fedadagrad = 0.0316
    epsilon_fedadam = 0.0316
    


mu_fedprox = 0

if(dataset=='CIFAR10'):
  mu_fedprox = 0.1
  
if(dataset=='CINIC10'):
  mu_fedprox = 1
  
if(dataset=='EMNIST'):
  mu_fedprox = 0.001
 
if (dataset=='CIFAR100'):
  mu_fedprox = 0.001

 





eta_l_algs = {'fedavgm(exp)': eta_l_fedavgm_exp, 'fedavgm': eta_l_fedavgm,'fedadam':eta_l_fedadam, 'fedprox':eta_l_fedprox, 'fedprox(exp)': eta_l_fedexp, 'fedavg': eta_l_fedavg, 'fedadagrad': eta_l_fedadagrad, 'fedexp': eta_l_fedexp, 'scaffold': eta_l_scaffold, 'scaffold(exp)': eta_l_scaffold_exp , 'fedexpsls':eta_l_fedexp,'fedsls':eta_l_fedavg, 'feddyn': eta_l_fedavg}

eta_g_algs = {'fedavgm(exp)': 'adaptive', 'fedavgm': eta_g_fedavgm,'fedadam':eta_g_fedadam, 'fedprox':eta_g_fedprox, 'fedprox(exp)': 'adaptive', 'fedavg':eta_g_fedavg, 'fedadagrad': eta_g_fedadagrad, 'fedexp': 'adaptive', 'scaffold': eta_g_scaffold, 'scaffold(exp)': 'adaptive','fedexpsls': 'adaptive', 'fedsls':eta_g_fedavg, 'feddyn': 1}

epsilon_algs = {'fedavgm(exp)': epsilon_fedavgm_exp, 'fedavgm': 0,'fedadam': 0, 'fedprox':0, 'fedprox(exp)':0, 'fedavg': 0, 'fedadagrad':0, 'fedexp':epsilon_fedexp, 'scaffold': 0, 'scaffold(exp)': epsilon_scaffold_exp,'fedexpsls':epsilon_fedexp, 'fedsls': 0, 'feddyn': 0}

mu_algs = {'fedavgm(exp)': 0, 'fedavgm': 0, 'fedadam':0, 'fedprox': mu_fedprox, 'fedprox(exp)': mu_fedprox, 'fedavg':0, 'fedadagrad':0, 'fedexp':0, 'scaffold':0, 'scaffold(exp)':0,'fedexpsls':0,'fedsls':0, 'feddyn': 0}



n = len(dataset_train)
print ("No. of clients", n)

p = np.zeros((n))

for i in range(n):
  p[i] = len(dataset_train[i])
             
p = p/np.sum(p)


start_time = time.time()

for alg in algs:

      # ---- Wallclock timers (per algorithm run) ---- added 18 feb
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    alg_run_start = time.time()
    print(f"[WALLCLOCK] alg={alg} run_start_iso={datetime.now().isoformat()} run_start_ts={alg_run_start}", flush=True)

    prev_round_end = alg_run_start #added 18 feb

    dict_results[alg] = {}
    
    filename_model_alg = alg + "_" + filename+".pt"

    d = parameters_to_vector(net_glob_org.parameters()).numel()


    net_glob = copy.deepcopy(net_glob_org)


    net_glob.train()


    w_glob = net_glob.state_dict()

    loss_t = []

    train_loss_algo_tmp = []
    train_acc_algo_tmp = []
    test_loss_algo_tmp = []
    test_acc_algo_tmp = []
    eta_g_tmp = []

    
    grad_mom = torch.zeros(d).to(args['device'])
    if(alg=='scaffold' or alg=='scaffold(exp)'):
        mem_mat = torch.zeros((n, d), device='cpu')  ### needed for scaffold
    elif alg == 'feddyn':
        mem_mat = torch.zeros((n, d), device='cpu')
    else:
        mem_mat = None

    feddyn_h = torch.zeros(d, device=args['device']) if alg == 'feddyn' else None
    
    w_vec_estimate = torch.zeros(d).to(args['device'])
    
    delta = torch.zeros(d).to(args['device'])

    grad_norm_avg_running = 0

    
    
    local_lr = eta_l_algs[alg]
    global_lr = eta_g_algs[alg]
    epsilon = epsilon_algs[alg]
    mu = mu_algs[alg]

    
    for t in range(0,args['rounds']):
        measure_this_round = measure_kappa and (t % kappa_measure_every == 0)
        round_kappa_rows = [] if measure_this_round else None
        

        print ("Algo ", alg, " Round No. " , t)

                # ---- Wallclock: round start ---- #added 18 feb
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        round_start = time.time() #added 18 feb


        # Keep SCAFFOLD local LR constant, matching the constant LR used by FedAvg.
        # All non-SCAFFOLD algorithms retain their existing decay behavior.
        if alg not in ('scaffold', 'scaffold(exp)'):
          local_lr = decay * local_lr
        epsilon = decay*decay*epsilon

        args_hyperparameters = {'mu': mu, 'eta_l':local_lr, 'decay': decay, 'weight_decay': weight_decay, 'eta_g': global_lr, 'use_gradient_clipping': use_gradient_clipping, 'max_norm': max_norm, 'epsilon': epsilon, 'feddyn_alpha': feddyn_alpha, 'reset_option': reset_option, 'eta_lmax': eta_lmax, 'armijo_c': armijo_c, 'use_augmentation':True}
        
        
        if(dataset=='CIFAR10' or dataset=='CIFAR100' or dataset=='CINIC10'):
          args_hyperparameters['use_augmentation'] = True
        else:
          args_hyperparameters['use_augmentation'] = False
      
        S = args['num_participating_clients']

        ind = np.random.choice(n,S,replace=False)



        grad_avg = torch.zeros(d).to(args['device'])

        c = torch.zeros((d,)).to(args['device'])
       

        w_init = parameters_to_vector(net_glob.parameters()).to(args['device'])

        grad_norm_sum = 0
        
        p_sum = 0
        feddyn_delta_sum = torch.zeros(d, device=args['device']) if alg == 'feddyn' else None

        # FedSLS line-search statistics for this communication round
        round_local_steps = 0
        round_search_forwards = 0
        round_total_forwards = 0
        round_total_backwards = 0
        round_failed_searches = 0
        client_final_step_sizes = []

        
        if(alg=='scaffold' or alg=='scaffold(exp)'):
            c_cpu = torch.zeros((d,), device='cpu')
            for i in range(n):
                c_cpu = c_cpu + p[i]*mem_mat[i]
            c = c_cpu.to(args['device'])

            
        
        #end_time = None #commented 18 feb
                # ---- Wallclock: client/grad phase ----
        grad_phase_start = time.time() #added 18 feb

        for i in ind:

            sls_seed_context = None
            if deterministic_sls_seed:
                sls_seed_context = {
                    "round": t, "client_id": int(i), "seed": seed
                }

            kappa_context = None
            if measure_this_round:
                kappa_context = {
                    "round": t,
                    "client_id": int(i),
                    "seed": seed,
                    "reference_dataset": kappa_reference_sets[int(i)]
                }

            if measure_this_round:
                result = get_grad_kappa(
                    copy.deepcopy(net_glob),
                    args,
                    args_hyperparameters,
                    dataset_train[i],
                    alg,
                    i,
                    kappa_context,
                    deterministic_seed=deterministic_sls_seed
                )
            elif deterministic_sls_seed:
                result = get_grad_deterministic_sls(
                    copy.deepcopy(net_glob),
                    args,
                    args_hyperparameters,
                    dataset_train[i],
                    alg,
                    sls_seed_context
                )
            else:
                result = get_grad(
                    copy.deepcopy(net_glob),
                    args,
                    args_hyperparameters,
                    dataset_train[i],
                    alg,
                    i,
                    c,
                    mem_mat
                )

            if alg in ('fedsls', 'fedexpsls'):
                if measure_this_round:
                    grad, search_stats, client_kappa_rows = result
                    round_kappa_rows.extend(client_kappa_rows)
                else:
                    grad, search_stats = result

                round_local_steps += search_stats["local_steps"]
                round_search_forwards += search_stats["line_search_forwards"]
                round_total_forwards += search_stats["total_forwards"]
                round_total_backwards += search_stats["total_backwards"]
                round_failed_searches += search_stats["failed_searches"]

                client_final_step_sizes.append(
                    float(search_stats["final_step_size"])
                )
            else:
                grad = result

            grad_norm_sum += p[i] * torch.linalg.norm(grad)**2
            grad_avg = grad_avg + p[i] * grad
            p_sum += p[i]
            if alg == 'feddyn':
                feddyn_delta_sum += grad

        if measure_this_round:
            with open(kappa_csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(round_kappa_rows)

        if alg in ('fedsls', 'fedexpsls'):
            avg_trials_per_step = (
                round_search_forwards / round_local_steps
                if round_local_steps > 0
                else 0.0
            )

            avg_final_step_size = (
                sum(client_final_step_sizes) / len(client_final_step_sizes)
                if client_final_step_sizes
                else 0.0
            )

            max_final_step_size = (
                max(client_final_step_sizes)
                if client_final_step_sizes
                else 0.0
            )

            min_final_step_size = (
                min(client_final_step_sizes)
                if client_final_step_sizes
                else 0.0
            )

            print(
                f"[SLS] round={t} "
                f"local_steps={round_local_steps} "
                f"search_forwards={round_search_forwards} "
                f"total_forwards={round_total_forwards} "
                f"total_backwards={round_total_backwards} "
                f"failed_searches={round_failed_searches} "
                f"avg_trials_per_step={avg_trials_per_step:.4f} "
                f"avg_final_step_size={avg_final_step_size:.8g} "
                f"min_final_step_size={min_final_step_size:.8g} "
                f"max_final_step_size={max_final_step_size:.8g}",
                flush=True
            )
        # added 18 feb
        if torch.cuda.is_available(): 
          torch.cuda.synchronize()
        grad_phase_end = time.time()
        grad_phase_sec = grad_phase_end - grad_phase_start



        

        with torch.no_grad():



            grad_avg = grad_avg/p_sum
            
            grad_norm_avg = grad_norm_sum/p_sum

            if alg == 'feddyn':
              feddyn_avg_delta = feddyn_delta_sum / S
              feddyn_h = feddyn_h - feddyn_alpha * feddyn_avg_delta
              grad_avg = feddyn_avg_delta - feddyn_h / feddyn_alpha

            eta_g = args_hyperparameters['eta_g']

            grad_norm_avg_running = grad_norm_avg +0.9*0.5*grad_norm_avg_running

            
            
            if(alg=='fedavgm' or alg=='fedavgm(ep)'):

              grad_avg = grad_avg + 0.9*grad_mom
                
              grad_mom = grad_avg


            if(alg=='fedadagrad'):
              
              delta = delta + grad_avg**2
                
              grad_avg = grad_avg/(torch.sqrt(delta+epsilon_fedadagrad))

            
            if(alg=='fedadam'):

              grad_avg = 0.1*grad_avg + 0.9*grad_mom
              grad_mom = grad_avg

              delta = 0.01*grad_avg**2 + 0.99*delta

              grad_avg_normalized = grad_avg/(0.1)
              delta_normalized = delta/(0.01)

              grad_avg = (grad_avg_normalized/torch.sqrt(delta_normalized + epsilon_fedadam))
            
            
            

            grad_avg_norm = torch.linalg.norm(grad_avg)**2

            if(eta_g == 'adaptive'):

              if(alg!='fedavgm(exp)'):
                eta_g = (0.5*grad_norm_avg/(grad_avg_norm + S*epsilon)).cpu()
              else:
                eta_g = (0.5*grad_norm_avg_running/(grad_avg_norm + S*epsilon)).cpu()


              if(alg!='fedavgm(exp)'):
                eta_g = max(1,eta_g)

            eta_g_tmp.append(eta_g)

            w_vec_prev = w_vec_estimate
            
            w_vec_estimate =  parameters_to_vector(net_glob.parameters()) + eta_g*grad_avg

            if(t>0):
              w_vec_avg = (w_vec_estimate+w_vec_prev)/2
            else:
              w_vec_avg = w_vec_estimate


            vector_to_parameters(w_vec_estimate,net_glob.parameters())
        
        
        eval_phase_start = time.time() #added 18 feb

        net_eval = copy.deepcopy(net_glob)

        if(alg=='fedexp' or alg=='scaffold(exp)' or alg=='fedprox(exp)' or alg=='fedavgm(exp)' or alg == 'fedexpsls'):
          vector_to_parameters(w_vec_avg, net_eval.parameters())

        sum_loss_train = 0
        sum_acc_train = 0
        for i in range(n):
                test_acc_i, test_loss_i = test_img(net_eval, dataset_train[i],args)

                sum_loss_train += test_loss_i
                sum_acc_train += test_acc_i

        sum_loss_train = sum_loss_train/n
        sum_acc_train = sum_acc_train/n
        print ("Training Loss ", sum_loss_train, "Training Accuracy ", sum_acc_train)
        
        sum_loss_test = 0
        sum_acc_test = 0

          
       
        test_acc_i, test_loss_i = test_img(net_eval, dataset_test_global,args)
        sum_loss_test = test_loss_i
        sum_acc_test = test_acc_i


        print ("Test Loss", sum_loss_test, "Test Accuracy ", sum_acc_test)
        
        #added 18 feb
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_phase_end = time.time()
        eval_phase_sec = eval_phase_end - eval_phase_start

                # ---- Wallclock: round end ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        round_end = time.time()

        round_time_sec = round_end - round_start
        elapsed_total_sec = round_end - alg_run_start
        since_prev_round_sec = round_end - prev_round_end
        prev_round_end = round_end

        print(
            f"[WALLCLOCK] alg={alg} round={t} "
            f"elapsed_total_sec={elapsed_total_sec:.4f} "
            f"round_time_sec={round_time_sec:.4f} "
            f"grad_phase_sec={grad_phase_sec:.4f} "
            f"eval_phase_sec={eval_phase_sec:.4f} "
            f"train_loss={sum_loss_train:.6f} train_acc={sum_acc_train:.6f} "
            f"test_loss={sum_loss_test:.6f} test_acc={sum_acc_test:.6f}",
            flush=True
        )
