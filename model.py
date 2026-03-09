import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from spikingjelly.activation_based import neuron, functional
from tqdm import tqdm
from utils import inject_label_conv
class LeakyLayerSJ(nn.Linear):

    def __init__(self, in_features, out_features,epochs, activation="lif", bias=False):
        super().__init__(in_features, out_features, bias=bias)

        self.activation = neuron.LIFNode(
                tau=2.0,          
                detach_reset=True
            )
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 5.0
        self.num_epochs = epochs

    def forward(self, x):

        functional.reset_net(self.activation)
        out_accumulator = []
        for t in range(x.shape[0]):
             x_t=x[t]
             x_t = x_t.flatten() 
             x_direction = x_t / (torch.norm(x_t, p=2, dim=-1, keepdim=True) + 1e-4)
             weighted_input = torch.mm(x_direction.unsqueeze(0), self.weight.T).squeeze(0)
             spk = self.activation(weighted_input)
             mem = self.activation.v 
             out_accumulator.append(mem)

        output = torch.stack(out_accumulator, dim=0)

        return output

    def train(self, x_pos, x_neg):
        tot_loss = []
        for _ in tqdm(range(self.num_epochs), desc="Training LeakyLayerSJ"):

            # Goodness = mean squared activations
            out_pos = self.forward(x_pos)
            out_neg = self.forward(x_neg) 
            g_pos = out_pos.pow(2).mean(dim=(0, 1))
            g_neg = out_neg.pow(2).mean(dim=(0, 1))

            # FF loss
            loss = F.softplus(
                torch.cat([
                    (-g_pos + self.threshold).unsqueeze(0),
                    (g_neg - self.threshold).unsqueeze(0)
                ])
            ).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            tot_loss.append(loss.item())

       
        output= self.forward(x_pos).detach(), self.forward(x_neg).detach()
        return output, tot_loss

class Net(nn.Module):

    def __init__(self, dims, epochs, activation="lif"):
        super().__init__()

        self.layers = nn.ModuleList([
            LeakyLayerSJ(dims[d], dims[d+1], epochs, activation)
            for d in range(len(dims) - 1)
        ])

    def train_net(self, x_pos, x_neg):
       
        h_pos, h_neg = x_pos, x_neg
        layer_losses = []

        for i, layer in enumerate(self.layers):
            print(f'Training layer {i}...')
            outputs, loss = layer.train(h_pos, h_neg)
            h_pos, h_neg = outputs        # [T, out_features] detached
            layer_losses.append(loss)

        return layer_losses

    def forward_once(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def predict(self, x, num_classes=11):

        goodness_per_label = []

        for label in range(num_classes):

            # inject candidate label into sequence
            h = inject_label_conv(x, label, num_classes)  # [T, C, H, W]

            goodness = []
            for layer in self.layers:
                h = layer(h)                        
                g = h.pow(2).sum(0).mean()          # scalar
                goodness.append(g)

            total_goodness = sum(goodness)           # scalar
            goodness_per_label.append(total_goodness)

        goodness_tensor = torch.stack(goodness_per_label)  # [11]
        return goodness_tensor.argmax().item()             # predicted label