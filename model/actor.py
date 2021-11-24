import torch

class Actor(torch.nn.Module):

    def initialize(self):

        torch.nn.init.kaiming_uniform_(self.lin1.weight)
        torch.nn.init.kaiming_uniform_(self.lin2.weight)
        torch.nn.init.kaiming_uniform_(self.lin3.weight)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)

    def __init__(self, statesize):


        super().__init__()

        self.lin1 = torch.nn.Linear(statesize, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128,2)

        self.initialize()

    def forward(self, x):
        y = torch.relu(self.lin1(x))
        y = torch.relu(self.lin2(y))
        y = self.lin3(y)

        mu = y[0]
        sigma = y[1]

        sigma = torch.log(torch.exp(sigma) + 1) + 1e-3

        dist = torch.distributions.Normal(mu, sigma)

        return dist


class Actor_v2(Actor):

    def forward(self, x):
        y = torch.relu(self.lin1(x))
        y = torch.relu(self.lin2(y))
        y = self.lin3(y)

        mu = y[:,0]

        sigma = y[:,1]

        sigma = torch.log(torch.exp(sigma) + 1) + 1e-3

        dist = torch.distributions.Normal(mu, sigma)

        return dist 

class Actor_ddpg(torch.nn.Module):

    def initialize(self):

        torch.nn.init.kaiming_uniform_(self.lin1.weight)
        torch.nn.init.kaiming_uniform_(self.lin2.weight)
        torch.nn.init.kaiming_uniform_(self.lin3.weight)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)

    def __init__(self, statesize):


        super().__init__()

        self.lin1 = torch.nn.Linear(statesize, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.lin3 = torch.nn.Linear(128,1)

        self.initialize()

    def forward(self, x):
        y = torch.relu(self.bn1(self.lin1(x)))
        y = torch.relu(self.bn2(self.lin2(y)))
        y = self.lin3(y)

        # normalize output to range [-2,2]

        y = 2* torch.tanh(y)

        return y



class Actor_ddpg_v2(torch.nn.Module):

    def initialize(self):

        torch.nn.init.kaiming_uniform_(self.lin1.weight)
        torch.nn.init.kaiming_uniform_(self.lin2.weight)
        torch.nn.init.kaiming_uniform_(self.lin3.weight)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)

    def __init__(self, statesize):


        super().__init__()

        self.lin1 = torch.nn.Linear(statesize, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128,1)

        self.initialize()

    def forward(self, x):
        y = torch.relu(self.lin1(x))
        y = torch.relu(self.lin2(y))
        y = self.lin3(y)

        # normalize output to range [-2,2]

        y = 2* torch.tanh(y)

        return y