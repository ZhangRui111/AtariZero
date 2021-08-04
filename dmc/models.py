import numpy as np
import torch
import torch.nn as nn


class AtariRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1),
        )

    def forward(self, s, a, return_value=False, flags=None):
        s = self.features(s)
        s = self.avgpool(s)
        x = torch.cat((s, a), dim=1)
        val = self.classifier(x)
        if return_value:
            return dict(values=val)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(val.shape[0], (1,))[0]
            else:
                # -------------------- variable action space --------------------
                # To deal with variable action space, the input to the model is
                # a collection of (state, legal action encoding), thus, the size of
                # dim-0 equals to the size of legal actions.
                # ---------------------------------------------------------------
                action = torch.argmax(val, dim=0)[0]
            return dict(action=action)


class Model:
    def __init__(self, device=0):
        self.model = AtariRGB().to(torch.device('cuda:' + str(device)))

    def forward(self, x, training=False, flags=None):
        return self.model.forward(x, training, flags)

    def share_memory(self):
        # You can use the share_memory() function on an nn.Module so that
        # the same parameters can be accessed from multiple processes
        # (using the multiprocessing module).
        self.model.share_memory()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def get_model(self):
        return self.model
