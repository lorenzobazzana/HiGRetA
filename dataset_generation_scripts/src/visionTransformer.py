import torch
import torch.nn as nn
from torchvision import models

class CustomViT(models.VisionTransformer):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, num_classes, weights_name):
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim)
        self.n_classification_heads = len(num_classes)

        self.drop = nn.Dropout(p=0.5)
        self.fc_custom = nn.ModuleList()
        for i in range(len(num_classes)):
            seq = nn.Sequential(nn.Linear(in_features=self.heads[0].in_features, out_features=256),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=256, out_features=64),
                                nn.Dropout(0.5),
                                nn.Linear(in_features=64, out_features=num_classes[i]))
            self.fc_custom.append(seq)
            #for layer in self.fc_custom[i]:
            #    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')

        self.load_state_dict(weights_name, strict=False)

    def forward(self, x, image_out=False):
        # From pytorch's VisionTransformer source code
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1) # nn.Parameter(torch.zeros(1, 1, hidden_dim))
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.drop(x)
        out = [] # idea: usare un class token per ogni classe
        for classifier in self.fc_custom:
            out.append(classifier(x))

        if image_out:
            return out, x
        else:
            return out