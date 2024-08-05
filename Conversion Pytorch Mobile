import torch
import torch.nn as nn
from torchvision import models

class MultiTaskEfficientNet(nn.Module):
    def __init__(self):
        super(MultiTaskEfficientNet, self).__init__()
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features

        # Remove the final fully connected layer
        self.model.classifier = nn.Identity()

        self.pylone_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.BatchNorm1d(2)
        )
        self.antenne_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.BatchNorm1d(2)
        )
        self.fh_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.BatchNorm1d(2)
        )

    def forward(self, x):
        features = self.model(x)
        pylone_out = self.pylone_fc(features)
        antenne_out = self.antenne_fc(features)
        fh_out = self.fh_fc(features)
        return pylone_out, antenne_out, fh_out


model = MultiTaskEfficientNet()
model.load_state_dict(torch.load('model_trained.pth'))
model.eval()


scripted_model = torch.jit.script(model)


scripted_model._save_for_lite_interpreter("model_scripted.ptl")

print("Le modèle a été converti et sauvegardé sous 'model_scripted.ptl'")
