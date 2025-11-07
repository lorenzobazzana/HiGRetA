import torch
import torch.nn as nn
from torchvision import models

class CustomVGG(models.vgg.VGG):
    def __init__(self, features, num_classes, weights_name):
        super().__init__(features=features)
        self.n_classification_heads = len(num_classes)

        self.classifier_custom = nn.ModuleList()
        for i in range(len(num_classes)):
            seq = nn.Sequential(
                nn.Linear(in_features=self.classifier[0].in_features, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=4096, out_features=num_classes[i], bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
            )
            self.classifier_custom.append(seq)
            for m in self.classifier_custom[i]:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    nn.init.constant_(m.bias, 0)

            self.load_state_dict(weights_name, strict=False)
            

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        out = []
        for class_layer in self.classifier_custom:
            out.append(class_layer(x))

        return out