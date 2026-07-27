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

# import torch.optim as optim

# !pip install torchsampler

# !pip install pretrainedmodels

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
import streamlit as st


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

from tqdm import tqdm
import PIL
from PIL import Image
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle

from sklearn.metrics import f1_score, fbeta_score
# from keras.utils import Sequence
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns


# from torchsampler import ImbalancedDatasetSampler

WORKERS = 2
CHANNEL = 3

import warnings
# import pretrainedmodels


device = 'cpu'
# modelp = torch.load(r"C:\Users\mkdnk\Desktop\dissertation\model_densenet.pth", map_location=torch.device('cpu'))
# modelp = modelp.to(device)


st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("DR Classifier")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type="png")


def predict(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    image= image
    model = torch.load(r"C:\Users\mkdnk\Desktop\dissertation\model_densenet.pth", map_location=torch.device("cpu"))

    # transform the input image through resizing, normalization
    transform = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )



img = Image.open(file_up)
x = transform(img)
x = torch.unsqueeze(x, 0)
model.eval()





correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    # for data in test_data:
    images, labels = data
    # images, labels = images.to(device), labels.to(device)
    # calculate outputs by running images through the network
    outputs = modelp(images)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()



# again no gradients needed
with torch.no_grad():
    for data in test_data:
        images, labels = data
        # images, labels = images.to(device), labels.to(device)
        outputs = modelp(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1






    # load the image, pre-process it, and make predictions
    
    images, labels = model(x)
    outputs = modelp(images)
    _, predicted = torch.max(outputs, 1)

    classes='dr', 'mild', 'moderate', 'no', 'severe'

    # # return the top 5 predictions ranked by highest probabilities
    # prob = torch.nn.functional.softmax(logits, dim=1)[0] * 100
    # _, indices = torch.sort(logits, descending=True)
    # return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Processing...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write(f"Prediction:= {i[0]} score {i[1]:.2f}")