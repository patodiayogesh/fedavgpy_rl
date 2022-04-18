from torch import nn

class DQN(nn.Module):

    def __init__(self, input, output):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input, 512),
            nn.ReLU(),
            nn.Linear(512, output),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layers(x)
        return y