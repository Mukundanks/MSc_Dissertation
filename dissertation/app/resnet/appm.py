import io
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import os


# loading the pre-trained model
def load_model():
    model = torch.load(
        r"C:\Users\mkdnk\Desktop\app\resnet\model_resnet1.pth",
        map_location=torch.device("cpu"),
    )
    model.eval()
    return model


# loading the labels
def load_labels():
    labels_path = r"C:\Users\mkdnk\Desktop\app\resnet\classes.txt.txt"
    labels_file = os.path.basename(labels_path)

    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


# loading the image
def load_image():
    uploaded_file = st.file_uploader(label="Pick an image", type="png")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


# running predictions


def predict(model, categories, image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, len(categories))
    for i in range(top5_prob.size(0)):
        st.write(categories[top5_catid[i]], top5_prob[i].item())


def main():
    st.title("Diabetic Retinopathy Classifier")
    model = load_model()
    categories = load_labels()
    image = load_image()
    result = st.button("Run on image")
    if result:
        st.write("Calculating results...")
        predict(model, categories, image)


if __name__ == "__main__":
    main()
