import torch
import torch.nn as nn
from torchvision import models

class CustomResNet(models.ResNet):
    def __init__(self, block, layers, num_classes, weights_name):
        super().__init__(block=block, layers=layers) # block type and layer structure from resnet50

        self.n_classification_heads = len(num_classes) # stile 10, genere 27 -> num_classes = [10,27]
        
        self.fc_custom = nn.ModuleList()
        for i in range(len(num_classes)):
            self.fc_custom.append(nn.Linear(in_features=self.fc.in_features, out_features=num_classes[i]))
            nn.init.kaiming_normal_(self.fc_custom[i].weight, mode='fan_out', nonlinearity='leaky_relu')

        self.load_state_dict(weights_name, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        out = []
        for class_layer in self.fc_custom:
            out.append(class_layer(x))

        return out # n_tag
