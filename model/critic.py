import torch 



class Critic(torch.nn.Module):


    def initialize(self):

        torch.nn.init.kaiming_normal_(self.lin1.weight)
        torch.nn.init.kaiming_normal_(self.lin2.weight)
        torch.nn.init.kaiming_normal_(self.lin3.weight)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)

    def __init__(self, state_size):
        super().__init__()

        self.lin1 = torch.nn.Linear(state_size, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128,1)


        self.initialize()

    def forward(self,x):

        y = torch.relu(self.lin1(x))
        y = torch.relu(self.lin2(y))
        y = torch.tanh(self.lin3(y))

        return y


class Critic_ddpg(torch.nn.Module):


    def initialize(self):

        torch.nn.init.kaiming_normal_(self.lin1.weight)
        torch.nn.init.kaiming_normal_(self.lin2.weight)
        torch.nn.init.kaiming_normal_(self.lin3.weight)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)

    def __init__(self, state_size, action_size):
        super().__init__()

        self.lin1 = torch.nn.Linear(state_size + action_size, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.lin3 = torch.nn.Linear(128,1)


        self.initialize()

    def forward(self,state, action):

        # concat state and action
        x = torch.cat([state, action], dim=-1)

        y = torch.relu(self.bn1(self.lin1(x)))
        y = torch.relu(self.bn2(self.lin2(y)))
        y = self.lin3(y)

        return y

 
class Critic_ddpg_v2(torch.nn.Module):


    def initialize(self):

        torch.nn.init.kaiming_normal_(self.lin1.weight)
        torch.nn.init.kaiming_normal_(self.lin2.weight)
        torch.nn.init.kaiming_normal_(self.lin3.weight)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)

    def __init__(self, state_size, action_size):
        super().__init__()

        self.lin1 = torch.nn.Linear(state_size + action_size, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128,1)


        self.initialize()

    def forward(self,state, action):

        # concat state and action
        x = torch.cat([state, action], dim=-1)

        y = torch.relu(self.lin1(x))
        y = torch.relu(self.lin2(y))
        y = self.lin3(y)

        return y

    