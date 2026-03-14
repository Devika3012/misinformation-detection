import pandas as pd
import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
from PIL import Image
from model import MultimodalModel
import os


print("Loading dataset...")

data = pd.read_csv("dataset/data.csv")

# LABEL MAPPING
label_map = {

    # TRUE
    "True": 0,
    "Mostly True": 0,
    "Correct Attribution": 0,

    # MIXED
    "Mixture": 1,
    "Unproven": 1,
    "Legend": 1,
    "Miscaptioned": 1,
    "Outdated": 1,

    # FALSE
    "False": 2,
    "Mostly False": 2,
    "Misattributed": 2,
    "Labeled Satire": 2,
    "Originated as Satire": 2
}

data["label"] = data["Rating"].map(label_map)

# REMOVE INVALID LABELS
data = data.dropna(subset=["label"])

data["label"] = data["label"].astype(int)

# COMBINE TEXT
data["text"] = data["Title"].astype(str) + " " + data["Body"].astype(str)

print("Dataset loaded")

# TOKENIZER
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# MODEL
model = MultimodalModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss_fn = torch.nn.CrossEntropyLoss()

# IMAGE TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Starting training...")

for index, row in data.iterrows():

    text = row["text"]

    img_path = "dataset/images/" + str(row["ID"]) + ".jpg"

    if not os.path.exists(img_path):
        continue

    label = torch.tensor([row["label"]], dtype=torch.long)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    image = Image.open(img_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    outputs = model(
        tokens["input_ids"],
        tokens["attention_mask"],
        image_tensor
    )

    loss = loss_fn(outputs, label)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if index % 10 == 0:
        print("Processed", index)

# SAVE MODEL
torch.save(model.state_dict(), "models/model.pt")

print("Training Finished")