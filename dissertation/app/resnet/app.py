# import warnings


# warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch
import streamlit as st
import cv2
from resnet50final import *

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

    model = torch.load(
        r"C:\Users\mkdnk\Desktop\app\resnet\model_resnet.pth",
        map_location=torch.device("cpu"),
    )

    # transform the input image through resizing, normalization
    transform = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    # print(img)
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    model.eval()
    # logits, attn_w = model(x)
    outputs = modelp(images)
    _, predictions = torch.max(outputs, 1)
    # collect the correct predictions for each class
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    with open(r"C:\Users\mkdnk\Desktop\app\resnet\classes.txt.txt", "r") as f:
        classes = f.read().split("\n")

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    _, predictions = torch.max(outputs, 1)
    return [(classes[idx], prob[idx].item()) for idx in predictions[0][:5]]
    # return logits


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
