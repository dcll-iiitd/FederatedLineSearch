from utils_libs_new import *
from SLS.sls import Sls
def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # y_true= target.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        # correct += y_pred.eq(target.view_as(y_pred)).sum().item()


    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.item(), test_loss

class LocalUpdate_Radam(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        # optimizer = torch.optim.RAdam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-01, weight_decay=0, decoupled_weight_decay=False, foreach=None, maximize=False, capturable=False, differentiable=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum = 0.9, weight_decay=0)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
class LocalUpdate_Adagrad(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        # optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-1, foreach=None,maximize=False, differentiable=False, fused=None)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
class LocalUpdate_Adadelta(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        # optimizer = torch.optim.Adadelta(net.parameters(), lr=0.01, rho=0.9, eps=1e-01, weight_decay=0, foreach=None,capturable=False, maximize=False, differentiable=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
      

class LocalUpdate_Amsgrad(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-1, amsgrad=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
      
class LocalUpdate_Adamax(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum = 0.9, weight_decay=self.weight_decay)
       
        optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-02, weight_decay=0, foreach=None,  maximize=False, differentiable=False, capturable=False)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return

      
class LocalUpdate_RmsProp(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-1, weight_decay=0, momentum=0.9, centered=False, capturable=False, foreach=None, maximize=False, differentiable=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum = 0.9, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
      


      
class LocalUpdate_Adam(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9, weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-1, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9, weight_decay=self.weight_decay)


        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
class LocalUpdate_Nadam(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        # optimizer = torch.optim.NAdam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-01, weight_decay=0, momentum_decay=0.004, decoupled_weight_decay=False, foreach=None, maximize=False, capturable=False, differentiable=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum = 0.9, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
class LocalUpdate(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              output = net(images)
              # labels = torch.tensor(labels, dtype=torch.long)
              # log_probs = output[-1]
              
              
              loss = self.loss_func(output, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
class LocalUpdate_Sls(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=self.weight_decay)
        optimizer = Sls(net.parameters())

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              output = net(images)
              # labels = torch.tensor(labels, dtype=torch.long)
              # log_probs = output[-1]
              
              
              loss = self.loss_func(output, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return
class LocalUpdate_scaffold(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        

        
    def train_and_sketch(self, net, idx, mem_mat, c):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = 0, weight_decay = self.weight_decay)


        prev_net = copy.deepcopy(net)
        
        eta = self.lr

        batch_loss = []
        step_count = 0

        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.use_data_augmentation == True):
                  images = self.transform_train(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)


                state_params_diff = c-mem_mat[idx]
                local_par_list = None
                for param in net.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                
                loss_algo = torch.sum(local_par_list * state_params_diff)
                loss = loss + loss_algo



                loss.backward()

                if(self.use_gradient_clipping ==True):
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)

                optimizer.step()
                batch_loss.append(loss.item())
                step_count=step_count+1

            
            
                if(step_count >= self.args['cp']):
                    break

            if(step_count >= self.args['cp']):
              break

        with torch.no_grad():


                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                
                mem_mat[idx] = (mem_mat[idx]-c) - params_delta_vec/(step_count*eta)


                model_to_return = params_delta_vec
            
        return model_to_return

class LocalUpdate_fedprox(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.MSELoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.mu = args_hyperparameters['mu']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = 0, weight_decay = self.weight_decay)


        prev_net = copy.deepcopy(net)
        prev_net_vec = parameters_to_vector(prev_net.parameters())
        
        eta = self.lr
        mu = self.mu

        batch_loss = []
        step_count = 0

        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.use_data_augmentation == True):
                  images = self.transform_train(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                local_par_list = parameters_to_vector(net.parameters())
                loss_algo = torch.linalg.norm(local_par_list-prev_net_vec)**2
                loss = loss + mu*0.5*loss_algo



                loss.backward()

                if(self.use_gradient_clipping==True):
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)

                optimizer.step()
                batch_loss.append(loss.item())
                step_count=step_count+1
                
               
            
            
                if(step_count >= self.args['cp']):
                    break
                    
                    
            if(step_count >= self.args['cp']):
                break

        with torch.no_grad():


                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                
                model_to_return = params_delta_vec
            
        return model_to_return



def get_grad(net_glob, args, args_hyperparameters,  dataset, alg, idx,  c):
    if(alg == 'fedexpsls' or alg == 'fedsls'):
        local = LocalUpdate_Sls(args, args_hyperparameters, dataset=dataset)

        grad = local.train_and_sketch(copy.deepcopy(net_glob))

        return grad
    if(alg == 'fedexp' or alg =='fedavg' or alg=='fedavgm' or alg=='fedavgm(exp)'):

        local = LocalUpdate(args, args_hyperparameters, dataset=dataset)

        grad = local.train_and_sketch(copy.deepcopy(net_glob))

        return grad

    elif(alg=='scaffold' or alg=='scaffold(exp)'):

         local = LocalUpdate_scaffold(args, args_hyperparameters, dataset=dataset)

        #  grad = local.train_and_sketch(copy.deepcopy(net_glob),idx,mem_mat,c)

         return grad
    
    elif(alg=='fedprox' or alg=='fedprox(exp)'):            
        

         local = LocalUpdate_fedprox(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
    elif(alg == 'fedadam'):
        
        local = LocalUpdate_Adam(args, args_hyperparameters, dataset=dataset)

        grad = local.train_and_sketch(copy.deepcopy(net_glob))

        return grad
    elif(alg == 'fednadam'):
        local = LocalUpdate_Nadam(args, args_hyperparameters, dataset=dataset)

        grad = local.train_and_sketch(copy.deepcopy(net_glob))

        return grad
    elif(alg == 'fedradam'):
         local = LocalUpdate_Radam(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
    elif(alg == 'fedrmsprop'):
         local = LocalUpdate_RmsProp(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
    elif(alg=='fedadagrad'):
         local = LocalUpdate_Adagrad(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
    elif(alg == 'fedadadelta'):
         local = LocalUpdate_Adadelta(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
    elif(alg == 'fedamsgrad'):
         local = LocalUpdate_Amsgrad(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
    elif(alg == 'fedadamax'):
         local = LocalUpdate_Adamax(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
       
  
    
      
    
        
      
        
    


    
