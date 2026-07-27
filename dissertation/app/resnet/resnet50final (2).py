# !git clone https://github.com/Mukundanks/Comp-702.git

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

import torch
import torchvision
from tqdm import tqdm

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


import warnings

warnings.filterwarnings("ignore")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle

from sklearn.metrics import f1_score, fbeta_score

from sklearn.model_selection import train_test_split
import seaborn as sns


from torchsampler import ImbalancedDatasetSampler

WORKERS = 2
CHANNEL = 3

import warnings

# import pretrainedmodels

warnings.filterwarnings("ignore")
IMG_SIZE = 512
NUM_CLASSES = 5
SEED = 77
TRAIN_NUM = (
    1000  # use 1000 when you just want to explore new idea, use -1 for full train
)

dtype = torch.FloatTensor


# img_dir = "../input/aptos2019-blindness-detection/train_images/"

# data = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
# data.head()

# # Distribution of Training Data
# display(data["diagnosis"].value_counts())
# sns.countplot(data["diagnosis"], color="blue")


img_size = 300


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def preprocess_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (img_size, img_size))
    return img


def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance

    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

            img = np.stack([img1, img2, img3], axis=-1)

        return img


def load_ben_color1(image, sigmaX=50):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


def circle_crop1(img, sigmaX=10):
    """
    Create circular crop around image centre
    """

    image = load_ben_color1(img, sigmaX=50)

    return img


import os
from os import listdir
import shutil


crop_size = 255
transformations = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(crop_size),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# batch_size = 4

# train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transformations)
# train_data = torch.utils.data.DataLoader(
#     train_data,
#     sampler=ImbalancedDatasetSampler(train_data),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=2,
# )

# test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transformations)
# test_data = torch.utils.data.DataLoader(
#     test_data, batch_size=batch_size, shuffle=True, num_workers=2
# )

# classes = ("0", "1", "2", "3", "4")

# test_data.dataset.classes


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # get some random training images
# dataiter = iter(train_data)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

# device = ["cuda", "cpu"]

"""pre-trained models"""


# print(pretrainedmodels.model_names)

"""available parameters in a model

"""


# model = torchvision.models.resnet50(pretrained=True)

# model


# model.fc = nn.Linear(2048, 5)

# device = "cuda"

# model = model.to(device)

# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# loss_l = []
# acc_l = []
# accuracy = 0
# best_accuracy = 0
# running_loss = 0.0
# # print(len(test_data))
# for epoch in range(35):
#     with tqdm(train_data, unit="iteration") as train_epoch:
#         train_epoch.set_description(f"Epoch {epoch}")
#         for step, data in enumerate(train_epoch):

#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = loss_fn(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             #
#             running_loss = 0.0
#             outputs = torch.argmax(outputs, 1)
#             accuracy = (outputs == labels).float().mean()
#             train_epoch.set_postfix(loss=loss.item(), accuracy=100 * accuracy.item())

#         running_vloss = 0.0


# print("Finished Training")

# torch.save(model, f"model_resnet.pth")


# dataiter = iter(test_data)
# images, labels = next(dataiter)
# # images, labels = images.to(device), labels.to(device)

# print images
# imshow(torchvision.utils.make_grid(images))
# print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

# len(images)

# model

device = "cpu"
modelp = torch.load("./model_resnet.pth")
modelp = modelp.to(device)

images, labels = images.to(device), labels.to(device)

outputs = modelp(images)

_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_data:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = modelp(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10 test images: {100 * correct // total} %")

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_data:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = modelp(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


# ##confusion matrix

# from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd

# y_pred = []
# y_true = []
# modelp = modelp.to("cpu")

# # iterate over test data
# for inputs, labels in test_data:
#     inputs = inputs
#     labels = labels
#     output = modelp(inputs)  # Feed Network

#     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#     y_pred.extend(output)  # Save Prediction

#     labels = labels.data.cpu().numpy()
#     y_true.extend(labels)  # Save Truth

# # constant for classes
# classes = ("0", "1", "2", "3", "4")
# # Build confusion matrix
# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(
#     cf_matrix / np.sum(cf_matrix) * 10,
#     index=[i for i in classes],
#     columns=[i for i in classes],
# )
# plt.figure(figsize=(12, 7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig("output.png")
