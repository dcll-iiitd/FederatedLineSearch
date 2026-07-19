from utils_libs import *


def stops_step_size(mean_proximal_objective, mean_delta, mu, epsilon=1e-10):
    """Return the multi-client SToPS extrapolation factor."""
    if mu <= 0:
        raise ValueError("FedExProx requires mu > 0 (equivalently gamma > 0)")
    denominator = mu * torch.linalg.norm(mean_delta) ** 2 + epsilon
    return 2.0 * mean_proximal_objective / denominator


class LocalUpdateFedExProx(object):
    """Approximate a client proximal point and return its proximal objective."""

    def __init__(self, args, hyperparameters, dataset):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(dataset, batch_size=args['bs'], shuffle=True)
        self.objective_loader = DataLoader(dataset, batch_size=args['bs'], shuffle=False)
        self.lr = hyperparameters['eta_l']
        self.mu = hyperparameters['mu']
        self.weight_decay = hyperparameters['weight_decay']
        self.use_augmentation = hyperparameters['use_augmentation']
        self.use_clipping = hyperparameters['use_gradient_clipping']
        self.max_norm = hyperparameters['max_norm']
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def train_and_sketch(self, net):
        if self.mu <= 0:
            raise ValueError("FedExProx requires mu > 0 (equivalently gamma > 0)")

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.lr, momentum=0,
            weight_decay=self.weight_decay
        )
        initial_parameters = parameters_to_vector(net.parameters()).detach().clone()
        local_step = 0

        while local_step < self.args['cp']:
            for images, labels in self.train_loader:
                images = images.to(self.args['device'])
                labels = labels.to(self.args['device'])
                if self.use_augmentation:
                    images = self.transform_train(images)

                optimizer.zero_grad()
                output = net(images)
                parameters = parameters_to_vector(net.parameters())
                loss = self.loss_func(output, labels)
                loss += 0.5 * self.mu * torch.linalg.norm(
                    parameters - initial_parameters
                ) ** 2
                loss.backward()
                if self.use_clipping:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_norm)
                optimizer.step()

                local_step += 1
                if local_step >= self.args['cp']:
                    break

        with torch.no_grad():
            client_delta = parameters_to_vector(net.parameters()) - initial_parameters
            net.eval()
            loss_sum = 0.0
            sample_count = 0
            for images, labels in self.objective_loader:
                images = images.to(self.args['device'])
                labels = labels.to(self.args['device'])
                output = net(images)
                loss_sum += F.cross_entropy(
                    output, labels, reduction='sum'
                ).item()
                sample_count += labels.numel()

            if sample_count == 0:
                raise ValueError("FedExProx received an empty client dataset")
            client_loss = loss_sum / sample_count
            proximal_objective = client_loss + 0.5 * self.mu * (
                torch.linalg.norm(client_delta).item() ** 2
            )

        return client_delta, proximal_objective
