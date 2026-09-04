from utils_libs import *
from SLS.sls import Sls
from scaffold_sls_new import LocalUpdateScaffoldSlsNew
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
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-1, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)



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
        self.optimizer_lr = (
            self.lr
            if args_hyperparameters.get('fedadam_eta_l_override', False)
            else 0.01
        )
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])

    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.optimizer_lr,
            weight_decay=self.weight_decay
        )

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


class LocalUpdate_FedDyn(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(dataset, batch_size=self.args["bs"], shuffle=True)
        self.lr = args_hyperparameters["eta_l"]
        self.alpha = args_hyperparameters["feddyn_alpha"]
        self.use_data_augmentation = args_hyperparameters["use_augmentation"]
        self.use_gradient_clipping = args_hyperparameters["use_gradient_clipping"]
        self.max_norm = args_hyperparameters["max_norm"]
        self.weight_decay = args_hyperparameters["weight_decay"]
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

    def train_and_sketch(self, net, idx, client_gradients):
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.lr, momentum=0, weight_decay=self.weight_decay
        )

        global_vec = parameters_to_vector(net.parameters()).detach().clone()
        client_gradient = client_gradients[idx].to(self.args["device"])
        step_count = 0

        while step_count < self.args["cp"]:
            for images, labels in self.ldr_train:
                images = images.to(self.args["device"])
                labels = labels.to(self.args["device"])

                if self.use_data_augmentation:
                    images = self.transform_train(images)

                optimizer.zero_grad()
                output = net(images)
                model_vec = parameters_to_vector(net.parameters())
                empirical_loss = self.loss_func(output, labels)
                linear_term = torch.dot(client_gradient, model_vec)
                proximal_term = 0.5 * self.alpha * torch.sum((model_vec - global_vec) ** 2)
                loss = empirical_loss - linear_term + proximal_term
                loss.backward()

                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_norm)

                optimizer.step()
                step_count += 1
                if step_count >= self.args["cp"]:
                    break

        with torch.no_grad():
            local_vec = parameters_to_vector(net.parameters())
            params_delta = local_vec - global_vec
            new_client_gradient = client_gradient - self.alpha * params_delta
            client_gradients[idx].copy_(new_client_gradient.cpu())

        return params_delta


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
        self.apply_sls_regularization = args_hyperparameters.get(
            'apply_sls_regularization', False
        )
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])

    def train_and_sketch(self, net):
        net.train()

        # Historical FedSLS was effectively unclipped because its closure cleared
        # the gradients that had been clipped before optimizer.step().
        optimizer = Sls(
            net.parameters(),
            max_grad_norm=(
                self.max_norm
                if self.apply_sls_regularization and self.use_gradient_clipping
                else None
            ),
            weight_decay=(
                self.weight_decay
                if self.apply_sls_regularization
                else 0.0
            )
        )
        prev_net = copy.deepcopy(net)

        step_count = 0
        total_line_search_forwards = 0
        total_forward_evaluations = 0
        failed_searches = 0

        while step_count < self.args['cp']:
            for images, labels in self.ldr_train:
                images = images.to(self.args['device'])
                labels = labels.to(self.args['device'])

                if self.use_data_augmentation:
                    images = self.transform_train(images)

                def closure():
                    optimizer.zero_grad()
                    output = net(images)
                    return self.loss_func(output, labels)

                loss, line_search_forwards, search_failed = optimizer.step(closure)

                total_line_search_forwards += line_search_forwards
                total_forward_evaluations += 1 + line_search_forwards
                failed_searches += int(search_failed)

                step_count += 1

                if step_count >= self.args['cp']:
                    break

        with torch.no_grad():
            vec_curr = parameters_to_vector(net.parameters())
            vec_prev = parameters_to_vector(prev_net.parameters())
            model_to_return = vec_curr - vec_prev

        search_stats = {
            "local_steps": step_count,
            "line_search_forwards": total_line_search_forwards,
            "total_forwards": total_forward_evaluations,
            "total_backwards": step_count,
            "failed_searches": failed_searches,
            "final_step_size": float(optimizer.state["step_size"]),
        }

        return model_to_return, search_stats


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
        client_control = mem_mat[idx].to(self.args["device"])

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


                state_params_diff = c - client_control
                local_par_list = parameters_to_vector(net.parameters())

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

                new_client_control = (client_control - c) - params_delta_vec / (step_count * eta)
                mem_mat[idx].copy_(new_client_control.detach().cpu())


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



def get_grad(net_glob, args, args_hyperparameters, dataset, alg, idx, c,
             mem_mat=None, round_idx=None, collect_sls_diagnostics=False):
    if alg == 'feddyn':
        local = LocalUpdate_FedDyn(args, args_hyperparameters, dataset=dataset)
        return local.train_and_sketch(copy.deepcopy(net_glob), idx, mem_mat)

    if alg in ('fedexpsls', 'fedsls', 'fedexpsls-regularized', 'fedsls-regularized', 'fedadamsls', 'fedadamexpsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized'):
        sls_hyperparameters = dict(args_hyperparameters)
        sls_hyperparameters['apply_sls_regularization'] = (
            alg in ('fedsls-regularized', 'fedexpsls-regularized', 'fedadamsls', 'fedadamexpsls-regularized', 'fedadamexpsls-scaled-regularized', 'fedadamexpsls-capped15-regularized')
        )
        local = LocalUpdate_Sls(args, sls_hyperparameters, dataset=dataset)

        grad,search_stats = local.train_and_sketch(copy.deepcopy(net_glob))

        return grad,search_stats
    if alg in ('scaffoldsls-new', 'scaffoldsls-grad', 'scaffoldsls-noh', 'scaffoldsls-rule3', 'scaffoldsls-rule3-eta2', 'scaffoldsls-surrogate'):
        acceptance_rules = {
            'scaffoldsls-new': 'original',
            'scaffoldsls-grad': 'grad_control',
            'scaffoldsls-noh': 'no_control_reward',
            'scaffoldsls-rule3': 'drift_slack',
            'scaffoldsls-rule3-eta2': 'quadratic_drift_slack',
            'scaffoldsls-surrogate': 'surrogate_armijo',
        }
        local = LocalUpdateScaffoldSlsNew(
            args, args_hyperparameters, dataset=dataset,
            acceptance_rule=acceptance_rules[alg],
            beta_slack=args.get('beta_slack'),
            alpha=args.get('sls_alpha', 0.1),
            eta_cap=args.get('eta_cap')
        )
        return local.train_and_sketch(
            copy.deepcopy(net_glob), idx, mem_mat, c,
            round_idx=round_idx,
            collect_diagnostics=collect_sls_diagnostics
        )
    if(alg == 'fedexp' or alg =='fedavg' or alg=='fedavgm' or alg=='fedavgm(exp)' or alg=='fedadam'):

        local = LocalUpdate(args, args_hyperparameters, dataset=dataset)

        grad = local.train_and_sketch(copy.deepcopy(net_glob))

        return grad

    elif alg == 'scaffold' or alg == 'scaffold(exp)':
        local = LocalUpdate_scaffold(args, args_hyperparameters, dataset=dataset)
        grad = local.train_and_sketch(copy.deepcopy(net_glob), idx, mem_mat, c)
        return grad

    elif alg == 'fedexprox':
         from fedexprox import LocalUpdateFedExProx
         local = LocalUpdateFedExProx(args, args_hyperparameters, dataset)
         return local.train_and_sketch(copy.deepcopy(net_glob))

    elif(alg=='fedprox' or alg=='fedprox(exp)'):


         local = LocalUpdate_fedprox(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad









