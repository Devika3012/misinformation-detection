import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel


class MultimodalModel(nn.Module):

    def __init__(self):
        super().__init__()

        # TEXT MODEL (BERT)
        self.text_model = BertModel.from_pretrained("bert-base-uncased")

        # IMAGE MODEL (ResNet18)
        self.image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # remove classification layer
        self.image_model.fc = nn.Identity()

        # FINAL CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, input_ids, attention_mask, images):

        # TEXT FEATURES
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = text_outputs.last_hidden_state[:, 0, :]

        # IMAGE FEATURES
        image_features = self.image_model(images)

        # CONCAT TEXT + IMAGE
        combined = torch.cat((text_features, image_features), dim=1)

        outputs = self.classifier(combined)

        return outputs