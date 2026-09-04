
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
import os
import csv
from datetime import datetime #added 18 feb


def capture_rng_state():
  state = {
      'python': random.getstate(),
      'numpy': np.random.get_state(),
      'torch_cpu': torch.get_rng_state(),
  }
  if torch.cuda.is_available():
    state['torch_cuda'] = torch.cuda.get_rng_state_all()
  return state


def restore_rng_state(state):
  random.setstate(state['python'])
  np.random.set_state(state['numpy'])
  torch.set_rng_state(state['torch_cpu'].cpu())
  if torch.cuda.is_available() and 'torch_cuda' in state:
    torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def save_checkpoint_atomic(path, checkpoint):
  path = os.path.abspath(path)
  parent = os.path.dirname(path)
  os.makedirs(parent, exist_ok=True)
  temporary_path = path + '.tmp'
  torch.save(checkpoint, temporary_path)
  os.replace(temporary_path, path)
  print(
      f"[CHECKPOINT] saved completed_rounds={checkpoint['completed_rounds']} "
      f"path={path}",
      flush=True
  )

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num_clients', type=int, required=True)
parser.add_argument('--num_participating_clients', type=int, required=True)
parser.add_argument('--num_rounds', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--fedexprox-alpha', type=float, default=1.0)
parser.add_argument('--eta-g', type=float, default=None)
parser.add_argument('--fedadam-eta-l', type=float, default=None)
parser.add_argument(
    '--fedadam-constant-client-lr',
    action='store_true'
)
parser.add_argument(
    '--fedadam-tau',
    type=float,
    default=None
)
parser.add_argument(
    '--fedadam-tau-outside-sqrt',
    action='store_true'
)
parser.add_argument('--sls-alpha', type=float, default=None,
                    help='Rule 3 gradient sufficient-decrease coefficient (default: 0.1)')
parser.add_argument('--eta-cap', type=float, default=None,
                    help='Optional Rule 3 ceiling applied to the initial backtracking trial')
parser.add_argument('--sls-reset-option', type=int, choices=(0, 1, 2), default=1,
                    help=('Scaffold-SLS reset: 0=previous accepted step, '
                          '1=FedSLS gradual increase, 2=initial step'))
parser.add_argument('--beta-slack', type=float, default=None,
                    help='Rule 3 drift-slack coefficient')
parser.add_argument('--checkpoint-path', type=str, default=None,
                    help='Save a resumable checkpoint at the requested round')
parser.add_argument('--resume-from', type=str, default=None,
                    help='Resume from a checkpoint; --num_rounds is the total target')
parser.add_argument('--save-checkpoint-at', type=int, default=None,
                    help='Completed round count at which to save (defaults to --num_rounds)')
parser.add_argument('--sls-diagnostics-path', type=str, default=None,
                    help=('Write per-accepted-step Scaffold-SLS diagnostics to CSV. '
                          'Only valid for Scaffold-SLS algorithms.'))

args_required = parser.parse_args()

if args_required.fedadam_tau is not None:
    if args_required.fedadam_tau <= 0:
        parser.error("--fedadam-tau must be greater than zero")

if args_required.fedadam_eta_l is not None:
    if args_required.fedadam_eta_l <= 0:
        parser.error("--fedadam-eta-l must be greater than zero")

fedadam_server_algorithms = {
    "fedadam", "fedadamsls", "fedadamexpsls", "fedadamexpsls-regularized", "fedadamexpsls-scaled-regularized", "fedadamexpsls-capped15-regularized"
}

if (
    args_required.fedadam_tau is not None
    or args_required.fedadam_tau_outside_sqrt
) and args_required.algorithm not in fedadam_server_algorithms:
    parser.error(
        "FedAdam server options require --algorithm fedadam, "
        "fedadamsls, fedadamexpsls"
    )

if (
    args_required.fedadam_eta_l is not None
    or args_required.fedadam_constant_client_lr
) and args_required.algorithm != "fedadam":
    parser.error(
        "Fixed client-LR FedAdam options require --algorithm fedadam; "
        "FedAdamSLS variants select client step sizes by line search"
    )

if args_required.algorithm == 'fedexprox' and args_required.fedexprox_alpha <= 0:
  parser.error("--fedexprox-alpha must be greater than zero")

if args_required.save_checkpoint_at is not None:
  if args_required.checkpoint_path is None:
    parser.error("--save-checkpoint-at requires --checkpoint-path")
  if not 1 <= args_required.save_checkpoint_at <= args_required.num_rounds:
    parser.error("--save-checkpoint-at must lie in [1, --num_rounds]")

scaffold_sls_algorithms = {
    'scaffoldsls-new', 'scaffoldsls-grad', 'scaffoldsls-noh', 'scaffoldsls-rule3', 'scaffoldsls-rule3-eta2',
    'scaffoldsls-surrogate'
}
if (args_required.sls_diagnostics_path is not None
        and args_required.algorithm not in scaffold_sls_algorithms):
  parser.error("--sls-diagnostics-path is only supported for Scaffold-SLS algorithms")

if args_required.algorithm in ('scaffoldsls-rule3', 'scaffoldsls-rule3-eta2'):
  if args_required.beta_slack is None or args_required.beta_slack < 0:
    parser.error("scaffoldsls-rule3 requires --beta-slack >= 0")
  if args_required.sls_alpha is not None and args_required.sls_alpha < 0:
    parser.error("scaffoldsls-rule3 requires --sls-alpha >= 0")
  if args_required.eta_cap is not None and args_required.eta_cap <= 0:
    parser.error("scaffoldsls-rule3 requires --eta-cap > 0")
else:
  if args_required.sls_alpha is not None:
    parser.error("--sls-alpha is only valid with scaffoldsls-rule3")
  if args_required.eta_cap is not None:
    parser.error("--eta-cap is only valid with scaffoldsls-rule3")
  if args_required.beta_slack is not None:
    parser.error("--beta-slack is only valid with scaffoldsls-rule3")



seed = args_required.seed
dataset = args_required.dataset
algorithm = args_required.algorithm
model = args_required.model
num_clients = args_required.num_clients
num_participating_clients = args_required.num_participating_clients
num_rounds = args_required.num_rounds
alpha = args_required.alpha
fedexprox_alpha = args_required.fedexprox_alpha

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



dict_results = {} ###dictionary to store results for all algorithms


###Default training parameters for all algorithms

args={
"bs":50,   ###batch size
"cp":20,   ### number of local steps
"device":'cuda:0',
"rounds":num_rounds, 
"num_clients": num_clients,
"num_participating_clients":num_participating_clients,
"sls_alpha": (args_required.sls_alpha if args_required.sls_alpha is not None else 0.1),
"beta_slack": args_required.beta_slack,
"eta_cap": args_required.eta_cap,
"sls_reset_option": args_required.sls_reset_option,
}

sls_diagnostics_fields = [
    'round', 'client_id', 'local_step', 'rule', 'eta', 'eta_before_cap', 'eta_after_cap', 'cap_active', 'drift',
    'grad_norm', 'direction_norm', 'f_before', 'f_after', 'h_before', 'h_after', 'eta_sum',
    'surrogate_change', 'armijo_rhs', 'armijo_margin', 'accepted_update_norm',
    'client_update_norm', 'normalized_client_update_norm', 'control_update_norm',
    'search_failed', 'backtrack_trials', 'beta_slack', 'sls_alpha'
]
sls_round_fields = [
    'round', 'rule', 'beta_slack', 'sls_alpha', 'eta_cap', 'global_train_loss', 'global_train_accuracy',
    'global_test_loss', 'global_test_accuracy', 'aggregate_client_update_norm',
    'applied_server_update_norm'
]
sls_diagnostics_path = None
sls_round_diagnostics_path = None
if args_required.sls_diagnostics_path is not None:
  sls_diagnostics_path = os.path.abspath(args_required.sls_diagnostics_path)
  diagnostics_root, diagnostics_ext = os.path.splitext(sls_diagnostics_path)
  sls_round_diagnostics_path = diagnostics_root + '.rounds' + (diagnostics_ext or '.csv')
  diagnostics_parent = os.path.dirname(sls_diagnostics_path)
  os.makedirs(diagnostics_parent, exist_ok=True)
  append_diagnostics = args_required.resume_from is not None
  file_has_content = (append_diagnostics and os.path.exists(sls_diagnostics_path)
                      and os.path.getsize(sls_diagnostics_path) > 0)
  if not append_diagnostics:
    with open(sls_diagnostics_path, 'w', newline='') as diagnostics_file:
      csv.DictWriter(diagnostics_file, fieldnames=sls_diagnostics_fields).writeheader()
    with open(sls_round_diagnostics_path, 'w', newline='') as round_file:
      csv.DictWriter(round_file, fieldnames=sls_round_fields).writeheader()
  elif not file_has_content:
    with open(sls_diagnostics_path, 'a', newline='') as diagnostics_file:
      csv.DictWriter(diagnostics_file, fieldnames=sls_diagnostics_fields).writeheader()
  if append_diagnostics and not (os.path.exists(sls_round_diagnostics_path)
          and os.path.getsize(sls_round_diagnostics_path) > 0):
    with open(sls_round_diagnostics_path, 'a', newline='') as round_file:
      csv.DictWriter(round_file, fieldnames=sls_round_fields).writeheader()
  print(
      f"[SLS_DIAGNOSTICS] steps={sls_diagnostics_path} "
      f"rounds={sls_round_diagnostics_path}",
      flush=True
  )


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
if args_required.fedadam_tau is not None:
    epsilon_fedadam = args_required.fedadam_tau
  
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

mu_fedexprox = mu_fedprox
if dataset in ('EMNIST', 'femnist', 'MNIST', 'shakespeare'):
  mu_fedexprox = 0.001

 





eta_l_algs = {'fedavgm(exp)': eta_l_fedavgm_exp, 'fedavgm': eta_l_fedavgm,'fedadam':eta_l_fedadam, 'fedprox':eta_l_fedprox, 'fedprox(exp)': eta_l_fedexp, 'fedexprox': eta_l_fedprox, 'fedavg': eta_l_fedavg, 'fedadagrad': eta_l_fedadagrad, 'fedexp': eta_l_fedexp, 'scaffold': eta_l_scaffold, 'scaffold(exp)': eta_l_scaffold_exp , 'fedexpsls':eta_l_fedexp,'fedsls':eta_l_fedavg, 'feddyn': eta_l_fedavg}

eta_g_algs = {'fedavgm(exp)': 'adaptive', 'fedavgm': eta_g_fedavgm,'fedadam':eta_g_fedadam, 'fedprox':eta_g_fedprox, 'fedprox(exp)': 'adaptive', 'fedexprox': fedexprox_alpha, 'fedavg':eta_g_fedavg, 'fedadagrad': eta_g_fedadagrad, 'fedexp': 'adaptive', 'scaffold': eta_g_scaffold, 'scaffold(exp)': 'adaptive','fedexpsls': 'adaptive', 'fedsls':eta_g_fedavg, 'feddyn': 1}

epsilon_algs = {'fedavgm(exp)': epsilon_fedavgm_exp, 'fedavgm': 0,'fedadam': 0, 'fedprox':0, 'fedprox(exp)':0, 'fedexprox': 0, 'fedavg': 0, 'fedadagrad':0, 'fedexp':epsilon_fedexp, 'scaffold': 0, 'scaffold(exp)': epsilon_scaffold_exp,'fedexpsls':epsilon_fedexp, 'fedsls': 0, 'feddyn': 0}

mu_algs = {'fedavgm(exp)': 0, 'fedavgm': 0, 'fedadam':0, 'fedprox': mu_fedprox, 'fedprox(exp)': mu_fedprox, 'fedexprox': mu_fedexprox, 'fedavg':0, 'fedadagrad':0, 'fedexp':0, 'scaffold':0, 'scaffold(exp)':0,'fedexpsls':0,'fedsls':0, 'feddyn': 0}

eta_l_algs['fedadamsls'] = eta_l_fedadam
eta_l_algs['fedadamexpsls'] = eta_l_fedadam
eta_l_algs['fedadamexpsls-regularized'] = eta_l_fedadam
eta_l_algs['fedadamexpsls-scaled-regularized'] = eta_l_fedadam
eta_l_algs['fedadamexpsls-capped15-regularized'] = eta_l_fedadam
eta_l_algs['fedsls-regularized'] = eta_l_fedavg
eta_l_algs['fedexpsls-regularized'] = eta_l_fedexp
eta_l_algs['scaffoldsls-new'] = eta_l_scaffold
eta_l_algs['scaffoldsls-grad'] = eta_l_scaffold
eta_l_algs['scaffoldsls-noh'] = eta_l_scaffold
eta_l_algs['scaffoldsls-rule3'] = eta_l_scaffold
eta_l_algs['scaffoldsls-rule3-eta2'] = eta_l_scaffold
eta_l_algs['scaffoldsls-surrogate'] = eta_l_scaffold

eta_g_algs['fedadamsls'] = eta_g_fedadam
eta_g_algs['fedadamexpsls'] = 'adaptive'
eta_g_algs['fedadamexpsls-regularized'] = 'adaptive'
eta_g_algs['fedadamexpsls-scaled-regularized'] = 'adaptive'
eta_g_algs['fedadamexpsls-capped15-regularized'] = 'adaptive'
eta_g_algs['fedsls-regularized'] = eta_g_fedavg
eta_g_algs['fedexpsls-regularized'] = 'adaptive'
eta_g_algs['scaffoldsls-new'] = 1
eta_g_algs['scaffoldsls-grad'] = 1
eta_g_algs['scaffoldsls-noh'] = 1
eta_g_algs['scaffoldsls-rule3'] = 1
eta_g_algs['scaffoldsls-rule3-eta2'] = 1
eta_g_algs['scaffoldsls-surrogate'] = 1

epsilon_algs['fedadamsls'] = 0
epsilon_algs['fedadamexpsls'] = epsilon_fedexp
epsilon_algs['fedadamexpsls-regularized'] = epsilon_fedexp
epsilon_algs['fedadamexpsls-scaled-regularized'] = epsilon_fedexp
epsilon_algs['fedadamexpsls-capped15-regularized'] = epsilon_fedexp
epsilon_algs['fedsls-regularized'] = 0
epsilon_algs['fedexpsls-regularized'] = epsilon_fedexp
epsilon_algs['scaffoldsls-new'] = 0
epsilon_algs['scaffoldsls-grad'] = 0
epsilon_algs['scaffoldsls-noh'] = 0
epsilon_algs['scaffoldsls-rule3'] = 0
epsilon_algs['scaffoldsls-rule3-eta2'] = 0
epsilon_algs['scaffoldsls-surrogate'] = 0

mu_algs['fedadamsls'] = 0
mu_algs['fedadamexpsls'] = 0
mu_algs['fedadamexpsls-regularized'] = 0
mu_algs['fedadamexpsls-scaled-regularized'] = 0
mu_algs['fedadamexpsls-capped15-regularized'] = 0
mu_algs['fedsls-regularized'] = 0
mu_algs['fedexpsls-regularized'] = 0
mu_algs['scaffoldsls-new'] = 0
mu_algs['scaffoldsls-grad'] = 0
mu_algs['scaffoldsls-noh'] = 0
mu_algs['scaffoldsls-rule3'] = 0
mu_algs['scaffoldsls-rule3-eta2'] = 0
mu_algs['scaffoldsls-surrogate'] = 0

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
    if alg in ('scaffold', 'scaffold(exp)', 'scaffoldsls-new', 'scaffoldsls-grad', 'scaffoldsls-noh', 'scaffoldsls-rule3', 'scaffoldsls-rule3-eta2', 'scaffoldsls-surrogate'):
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
    if args_required.fedadam_eta_l is not None:
      local_lr = args_required.fedadam_eta_l
    global_lr = eta_g_algs[alg]
    if args_required.eta_g is not None and alg not in ("fedadamexpsls", "fedadamexpsls-regularized", "fedadamexpsls-scaled-regularized", "fedadamexpsls-capped15-regularized"):
      global_lr = args_required.eta_g
    epsilon = epsilon_algs[alg]
    mu = mu_algs[alg]

    start_round = 0
    if args_required.resume_from is not None:
      checkpoint = torch.load(
          args_required.resume_from,
          map_location=args['device'],
          weights_only=False
      )

      expected_metadata = {
          'algorithm': alg,
          'dataset': dataset,
          'model': model,
          'seed': seed,
          'actual_num_clients': n,
          'num_participating_clients': args['num_participating_clients'],
          'alpha': alpha,
          'eta_g': global_lr,
          'batch_size': args['bs'],
          'local_steps': args['cp'],
          'beta_slack': args_required.beta_slack,
                'sls_alpha': args['sls_alpha'],
                'eta_cap': args_required.eta_cap,
      }
      saved_metadata = checkpoint.get('metadata', {})
      mismatches = {
          key: (saved_metadata.get(key), value)
          for key, value in expected_metadata.items()
          if saved_metadata.get(key) != value
      }
      if mismatches:
        raise ValueError(f"Checkpoint configuration mismatch: {mismatches}")

      start_round = int(checkpoint['completed_rounds'])
      if start_round >= args['rounds']:
        raise ValueError(
            f"Checkpoint already completed {start_round} rounds, but the "
            f"requested total is {args['rounds']}"
        )

      net_glob.load_state_dict(checkpoint['server_model'])
      if mem_mat is not None:
        saved_mem_mat = checkpoint.get('client_state')
        if saved_mem_mat is None or tuple(saved_mem_mat.shape) != tuple(mem_mat.shape):
          raise ValueError(
              "Checkpoint client control/state matrix is missing or has the wrong shape"
          )
        mem_mat.copy_(saved_mem_mat.cpu())

      optimizer_state = checkpoint.get('server_state', {})
      grad_mom.copy_(optimizer_state.get('grad_mom', grad_mom).to(args['device']))
      delta.copy_(optimizer_state.get('delta', delta).to(args['device']))
      w_vec_estimate.copy_(
          optimizer_state.get('w_vec_estimate', w_vec_estimate).to(args['device'])
      )
      grad_norm_avg_running = optimizer_state.get(
          'grad_norm_avg_running', grad_norm_avg_running
      )
      if feddyn_h is not None:
        saved_feddyn_h = optimizer_state.get('feddyn_h')
        if saved_feddyn_h is None:
          raise ValueError("FedDyn checkpoint is missing feddyn_h")
        feddyn_h.copy_(saved_feddyn_h.to(args['device']))
      local_lr = checkpoint.get('local_lr', local_lr)
      epsilon = checkpoint.get('epsilon', epsilon)
      restore_rng_state(checkpoint['rng_state'])
      print(
          f"[CHECKPOINT] resumed completed_rounds={start_round} "
          f"path={os.path.abspath(args_required.resume_from)}",
          flush=True
      )

    
    for t in range(start_round,args['rounds']):
        

        print ("Algo ", alg, " Round No. " , t)

                # ---- Wallclock: round start ---- #added 18 feb
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        round_start = time.time() #added 18 feb


        # Keep SCAFFOLD local LR constant, matching the constant LR used by FedAvg.
        # All non-SCAFFOLD algorithms retain their existing decay behavior.
        if (alg not in ('scaffold', 'scaffold(exp)', 'scaffoldsls-new', 'scaffoldsls-grad', 'scaffoldsls-noh', 'scaffoldsls-rule3', 'scaffoldsls-rule3-eta2', 'scaffoldsls-surrogate')
                and not (alg == 'fedadam' and args_required.fedadam_constant_client_lr)):
          local_lr = decay * local_lr
        epsilon = decay*decay*epsilon

        args_hyperparameters = {'mu': mu, 'eta_l':local_lr, 'decay': decay, 'weight_decay': weight_decay, 'eta_g': global_lr, 'use_gradient_clipping': use_gradient_clipping, 'max_norm': max_norm, 'epsilon': epsilon, 'feddyn_alpha': feddyn_alpha, 'use_augmentation':True, 'sls_reset_option': args['sls_reset_option'], 'fedadam_eta_l_override': args_required.fedadam_eta_l is not None}
        
        
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

        
        if alg in ('scaffold', 'scaffold(exp)', 'scaffoldsls-new', 'scaffoldsls-grad', 'scaffoldsls-noh', 'scaffoldsls-rule3', 'scaffoldsls-rule3-eta2', 'scaffoldsls-surrogate'):
            c_cpu = torch.zeros((d,), device='cpu')
            for i in range(n):
                c_cpu = c_cpu + p[i]*mem_mat[i]
            c = c_cpu.to(args['device'])

            
        
        #end_time = None #commented 18 feb
                # ---- Wallclock: client/grad phase ----
        grad_phase_start = time.time() #added 18 feb

        for i in ind:

            result = get_grad(
                copy.deepcopy(net_glob),
                args,
                args_hyperparameters,
                dataset_train[i],
                alg,
                i,
                c,
                mem_mat,
                round_idx=t,
                collect_sls_diagnostics=(sls_diagnostics_path is not None)
            )

            if alg in ('fedsls', 'fedexpsls', 'fedsls-regularized', 'fedexpsls-regularized', 'fedadamsls', 'fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized', 'scaffoldsls-new', 'scaffoldsls-grad', 'scaffoldsls-noh', 'scaffoldsls-rule3', 'scaffoldsls-rule3-eta2', 'scaffoldsls-surrogate'):
                grad, search_stats = result

                round_local_steps += search_stats["local_steps"]
                round_search_forwards += search_stats["line_search_forwards"]
                round_total_forwards += search_stats["total_forwards"]
                round_total_backwards += search_stats["total_backwards"]
                round_failed_searches += search_stats["failed_searches"]

                client_final_step_sizes.append(
                    float(search_stats["final_step_size"])
                )
                if search_stats.get("diagnostics"):
                    with open(sls_diagnostics_path, 'a', newline='') as diagnostics_file:
                        writer = csv.DictWriter(
                            diagnostics_file, fieldnames=sls_diagnostics_fields
                        )
                        writer.writerows(search_stats["diagnostics"])
            else:
                grad = result

            grad_norm_sum += p[i] * torch.linalg.norm(grad)**2
            grad_avg = grad_avg + p[i] * grad
            p_sum += p[i]
            if alg == 'feddyn':
                feddyn_delta_sum += grad

        if alg in ('fedsls', 'fedexpsls', 'fedsls-regularized', 'fedexpsls-regularized', 'fedadamsls', 'fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized', 'scaffoldsls-new', 'scaffoldsls-grad', 'scaffoldsls-noh', 'scaffoldsls-rule3', 'scaffoldsls-rule3-eta2', 'scaffoldsls-surrogate'):
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
            aggregate_client_update_norm = float(
                torch.linalg.vector_norm(grad_avg).item())

            if alg == 'feddyn':
              feddyn_avg_delta = feddyn_delta_sum / S
              feddyn_h = feddyn_h - feddyn_alpha * feddyn_avg_delta
              grad_avg = feddyn_avg_delta - feddyn_h / feddyn_alpha

            eta_g = args_hyperparameters['eta_g']

            grad_norm_avg_running = grad_norm_avg +0.9*0.5*grad_norm_avg_running

            # Compute the FedExp multiplier from the raw aggregated client update,
            # before FedAdam changes the direction's scale.
            fedadam_exp_raw_norm = None
            if alg in ('fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized'):
                fedadam_exp_raw_norm = torch.linalg.norm(grad_avg)**2
            
            if(alg=='fedavgm' or alg=='fedavgm(ep)'):

              grad_avg = grad_avg + 0.9*grad_mom
                
              grad_mom = grad_avg


            if(alg=='fedadagrad'):
              
              delta = delta + grad_avg**2
                
              grad_avg = grad_avg/(torch.sqrt(delta+epsilon_fedadagrad))

            
            if alg in ('fedadam', 'fedadamsls', 'fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized'):

              # Canonical FedAdam moments use the raw aggregated client update.
              raw_update = grad_avg

              grad_mom = (
                  0.9 * grad_mom
                  + 0.1 * raw_update
              )

              delta = (
                  0.99 * delta
                  + 0.01 * raw_update**2
              )

              # FedAdamSLS variants use the canonical FedAdam denominator.
              # Ordinary FedAdam retains its historical default unless the
              # opt-in compatibility flag is supplied.
              use_canonical_fedadam_denominator = (
                  alg in ("fedadamsls", "fedadamexpsls", "fedadamexpsls-regularized", "fedadamexpsls-scaled-regularized", "fedadamexpsls-capped15-regularized")
                  or args_required.fedadam_tau_outside_sqrt
              )

              if use_canonical_fedadam_denominator:
                  grad_avg = grad_mom / (
                      torch.sqrt(delta) + epsilon_fedadam
                  )
              else:
                  grad_avg = grad_mom / torch.sqrt(
                      delta + epsilon_fedadam
                  )
            

            grad_avg_norm = torch.linalg.norm(grad_avg)**2

            if alg == 'fedexprox':
              print(
                  f"[FedExProx] round={t} "
                  f"alpha={float(eta_g):.8g} mu={mu:.8g} "
                  f"gamma={1.0 / mu:.8g}",
                  flush=True
              )

            if eta_g == 'adaptive':

              if alg in ('fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized'):
                  fedadam_exp_multiplier = (
                      0.5 * grad_norm_avg
                      / (fedadam_exp_raw_norm + S * epsilon)
                  ).cpu()

                  if alg == 'fedadamexpsls-capped15-regularized':
                      fedadam_exp_multiplier = min(
                          1.5,
                          max(1, fedadam_exp_multiplier)
                      )
                  else:
                      fedadam_exp_multiplier = max(
                          1,
                          fedadam_exp_multiplier
                      )

                  if alg == 'fedadamexpsls-scaled-regularized':
                      base_eta_g = (
                          args_required.eta_g
                          if args_required.eta_g is not None
                          else eta_g_fedadam
                      )
                      eta_g = base_eta_g * fedadam_exp_multiplier
                      eta_g_mode = "base_times_multiplier"
                  else:
                      eta_g = fedadam_exp_multiplier
                      eta_g_mode = (
                          "bounded_1_to_1p5"
                          if alg == 'fedadamexpsls-capped15-regularized'
                          else "multiplier_only"
                      )

                  print(
                      f"[FedAdamExpSLS] round={t} "
                      f"mode={eta_g_mode} "
                      f"multiplier={float(fedadam_exp_multiplier):.8g} "
                      f"eta_g={float(eta_g):.8g}",
                      flush=True
                  )

              elif alg != 'fedavgm(exp)':
                  eta_g = (
                      0.5 * grad_norm_avg
                      / (grad_avg_norm + S * epsilon)
                  ).cpu()

              else:
                  eta_g = (
                      0.5 * grad_norm_avg_running
                      / (grad_avg_norm + S * epsilon)
                  ).cpu()

              if alg not in ('fedavgm(exp)', 'fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized'):
                  eta_g = max(1, eta_g)

            eta_g_tmp.append(eta_g)

            applied_server_update_norm = float(
                torch.linalg.vector_norm(eta_g * grad_avg).item())

            w_vec_prev = w_vec_estimate
            
            w_vec_estimate =  parameters_to_vector(net_glob.parameters()) + eta_g*grad_avg

            if(t>0):
              w_vec_avg = (w_vec_estimate+w_vec_prev)/2
            else:
              w_vec_avg = w_vec_estimate


            vector_to_parameters(w_vec_estimate,net_glob.parameters())
        
        
        eval_phase_start = time.time() #added 18 feb

        net_eval = copy.deepcopy(net_glob)

        if(alg=='fedexp' or alg=='scaffold(exp)' or alg=='fedprox(exp)' or alg=='fedavgm(exp)' or alg in ('fedexpsls', 'fedexpsls-regularized') or alg in ('fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized')):
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

        if sls_round_diagnostics_path is not None:
          rule_labels = {
              'scaffoldsls-new': 'original',
              'scaffoldsls-grad': 'grad_control',
              'scaffoldsls-noh': 'no_control_reward',
              'scaffoldsls-rule3': 'drift_slack',
              'scaffoldsls-rule3-eta2': 'quadratic_drift_slack',
              'scaffoldsls-surrogate': 'surrogate_armijo',
          }
          with open(sls_round_diagnostics_path, 'a', newline='') as round_file:
            writer = csv.DictWriter(round_file, fieldnames=sls_round_fields)
            writer.writerow({
                'round': t,
                'rule': rule_labels[alg],
                'beta_slack': args_required.beta_slack,
                'sls_alpha': args['sls_alpha'],
                'eta_cap': args_required.eta_cap,
                'global_train_loss': float(sum_loss_train),
                'global_train_accuracy': float(sum_acc_train),
                'global_test_loss': float(sum_loss_test),
                'global_test_accuracy': float(sum_acc_test),
                'aggregate_client_update_norm': aggregate_client_update_norm,
                'applied_server_update_norm': applied_server_update_norm,
            })

        completed_rounds = t + 1
        checkpoint_round = (
            args_required.save_checkpoint_at
            if args_required.save_checkpoint_at is not None
            else args['rounds']
        )
        if (args_required.checkpoint_path is not None
                and completed_rounds == checkpoint_round):
          checkpoint = {
              'checkpoint_version': 1,
              'completed_rounds': completed_rounds,
              'metadata': {
                  'algorithm': alg,
                  'dataset': dataset,
                  'model': model,
                  'seed': seed,
                  'requested_num_clients': num_clients,
                  'actual_num_clients': n,
                  'num_participating_clients': args['num_participating_clients'],
                  'alpha': alpha,
                  'eta_g': global_lr,
                  'batch_size': args['bs'],
                  'local_steps': args['cp'],
          'beta_slack': args_required.beta_slack,
                'sls_alpha': args['sls_alpha'],
                'eta_cap': args_required.eta_cap,
              },
              'server_model': net_glob.state_dict(),
              'client_state': mem_mat,
              'server_state': {
                  'grad_mom': grad_mom,
                  'delta': delta,
                  'w_vec_estimate': w_vec_estimate,
                  'grad_norm_avg_running': grad_norm_avg_running,
                  'feddyn_h': feddyn_h,
              },
              'local_lr': local_lr,
              'epsilon': epsilon,
              'rng_state': capture_rng_state(),
          }
          save_checkpoint_atomic(args_required.checkpoint_path, checkpoint)
