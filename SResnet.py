import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer
from tqdm import tqdm
import os
# ==============================
# Basic Spiking Residual Block
# ==============================
class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = layer.Conv2d(in_planes, planes, kernel_size=3,
                                  stride=stride, padding=1, bias=False , step_mode='m')
        self.bn1 = layer.BatchNorm2d(planes,step_mode='m')
        self.lif1 = neuron.LIFNode(step_mode='m')

        self.conv2 = layer.Conv2d(planes, planes, kernel_size=3,
                                  stride=1, padding=1, bias=False, step_mode='m')
        self.bn2 = layer.BatchNorm2d(planes,step_mode='m')
        self.lif2 = neuron.LIFNode(step_mode='m')

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                layer.Conv2d(in_planes, planes, kernel_size=1,
                             stride=stride, bias=False , step_mode='m'),
                layer.BatchNorm2d(planes,step_mode='m'),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.lif2(out)

        return out


# ==============================
# Spiking ResNet-18
# ==============================
class SpikingResNet18(nn.Module):

    def __init__(self, num_classes=11):
        super().__init__()

        self.in_planes = 64

        self.conv1 = layer.Conv2d(2, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False , step_mode='m')
        self.bn1 = layer.BatchNorm2d(64,step_mode='m')
        self.lif1 = neuron.LIFNode(step_mode='m')

        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1,step_mode='m')

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1),step_mode='m')
        self.fc = layer.Linear(512, num_classes,step_mode='m')

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(SpikingBasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes

        for _ in range(1, blocks):
            layers.append(SpikingBasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, 2, H, W]
        x = x.transpose(0, 1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 2)
        out = self.fc(out)

        # Average over time dimension
        out = out.mean(0)

        return out

def train_resnet(
    model,
    train_loader,
    optimizer,
    criterion,
    num_epochs=20,
    save_every=5,
    save_dir="checkpoints",
    start_epoch=0
):

    os.makedirs(save_dir, exist_ok=True)
    model.train()

    for epoch in range(start_epoch, num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        running_loss = 0.0

        for event, label in progress_bar:

            optimizer.zero_grad()

            output = model(event)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            functional.reset_net(model)

            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:

            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

            print(f"Checkpoint saved at {checkpoint_path}")

    print("\nTraining Complete ✅")