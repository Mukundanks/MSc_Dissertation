# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg

import torch
from PIL import Image, ImageFilter
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms

import matplotlib.pyplot as plt

# %matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy
from keras.applications.resnet50 import preprocess_input
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

WORKERS = 2
CHANNEL = 3

import warnings

warnings.filterwarnings("ignore")
IMG_SIZE = 512
NUM_CLASSES = 5
SEED = 77
TRAIN_NUM = (
    1000  # use 1000 when you just want to explore new idea, use -1 for full train
)

dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor


train_dir = "/users/sgmsures/project/dataset/processed/output/train/"
val_dir = "/users/sgmsures/project/dataset/processed/output/val/"
test_dir = "/users/sgmsures/project/dataset/processed/output/test/"

labels = pd.read_csv("/users/sgmsures/project/dataset/train.csv")
print(len(labels))
labels.head()

# Example of images
plt.figure(figsize=[15, 15])
i = 1
for img_name in labels["id_code"][:10]:
    img = mpimg.imread(train_dir + img_name + ".png")
    plt.subplot(6, 5, i)
    plt.imshow(img)
    i += 1
plt.show()

"""
def load_ben_color1(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


def circle_crop1(img, sigmaX=10):

    # Create circular crop around image centre

    image = load_ben_color1(img, sigmaX=50)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


class ImageData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.id_code[index] + ".png"
        label = self.df.diagnosis[index]
        img_path = os.path.join(self.data_dir, img_name)

        image = mpimg.imread(img_path)

        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        image_circle = circle_crop1(image)
        image_circle = self.transform(image_circle)

        return image_circle, label
"""

bs = 5

data_transf = transforms.Compose(
    [
        transforms.ToPILImage(mode="RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-120, 120)),
        transforms.Resize(265),
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=data_transf)
valid_data = torchvision.datasets.ImageFolder(root=val_dir, transform=data_transf)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=data_transf)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=bs)
val_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=bs)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=bs)


"""
train_data = ImageData(df=labels, data_dir=train_dir, transform=data_transf)
train_loader = DataLoader(
    dataset=train_data, batch_size=32, drop_last=True, pin_memory=True, num_workers=10
)

val_data = ImageData(df=labels, data_dir=val_dir, transform=data_transf)
val_loader = DataLoader(
    dataset=val_data, batch_size=32, drop_last=True, pin_memory=True, num_workers=10
)
"""
X, y = next(iter(train_loader))


model1 = torchvision.models.resnet50()

# model1

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Changing number of model's output classes to 5
model1.fc = nn.Linear(2048, 5)
# model1

# Transfer execution to GPU
model1 = model1.to("cuda")

optimizer = optim.Adam(model1.parameters())
loss_func = nn.CrossEntropyLoss()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Train model


for epoch in range(1, n_epochs + 1):
    print("Epoch:", epoch)
    for i, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Training"
    ):
        total_samples = len(train_loader.dataset)

        # device
        model = model.to(device)
        x = data.to(device)
        y = target.to(device)
        optimizer.zero_grad()

        logits, attn_weights = model(x)
        proba = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(proba, y, reduction="sum")
        loss.backward()

        yhat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(yhat == y).item() / len(train_data)
        optimizer.step()

    model1.eval()
    total_samples = len(val_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for i, (data, target) in tqdm(
            enumerate(val_loader), total=len(val_loader), desc="Evaluation"
        ):

            model = model1.to(device)
            data = data.to(device)
            target = target.to(device)

            logits, attn_weights = model(x)
            proba = F.log_softmax(logits, dim=1)
            val_loss = F.nll_loss(proba, y, reduction="sum")
            _, pred = torch.max(logits, dim=1)


torch.save(model1, "/users/sgmsures/project/dataset/modelresnet.pth")

"""
# submit = pd.read_csv("/users/sgmsures/project/dataset/test.csv")
# submit["diagnosis"] = 0
test_data = ImageData(data_dir=test_dir, transform=data_transf)
test_loader = DataLoader(dataset=test_data, shuffle=False)
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
#


# testing
## loading the model
model = torch.load("/users/sgmsures/project/dataset/modelresnet.pth")
mat = torch.zeros(len(classes), len(classes))
acc_v = []
loss_v = []
pred = []
lb = []
with torch.no_grad():
    for epoch in range(1):
        with tqdm(test_loader, unit="iteration") as train_epoch:
            train_epoch.set_description(f"Epoch {epoch}")
            for step, data in enumerate(train_epoch):

                # #init
                inputs, labels = data
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                net = model.to("cuda")

                # #training
                out = net(inputs)
                loss = loss_fn(out, labels)

                # stats
                out = torch.argmax(out, 1)
                accuracy = (out == labels).float().mean()
                train_epoch.set_postfix(
                    loss=loss.item(), accuracy=100 * accuracy.item()
                )

                acc_v.append(accuracy.item())
                loss_v.append(loss.item())

                for t, p in zip(labels.view(-1), out.view(-1)):
                    mat[t.long(), p.long()] += 1
