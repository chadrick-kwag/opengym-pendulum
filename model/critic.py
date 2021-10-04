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

    