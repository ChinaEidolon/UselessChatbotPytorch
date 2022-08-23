import torch
import torch.nn as nn

class NeuralNet(nn.Module): # feed forward neural net, basically the standard neural net layer. This is just a standard NN lol.
    
    def __init__(self,input_size,hidden_size, num_classes):
        super(NeuralNet, self).__init__() # this just delegates the call to the superclass lmao...to nn.Module. Needed to initialize nn.Module properly.
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x): # and this is making the actual info
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(x)
        out = self.relu(out)
        out = self.l3(out)

        # no activation and no softmax just yet
        return out
