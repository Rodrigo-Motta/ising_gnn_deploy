import streamlit as st
import torch
import torch.nn.functional as func
from torch_geometric.nn import ChebConv, global_mean_pool, aggr
import numpy as np
import pandas as pd
from model import GCN
import utils as ut

# Load the model and set it to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(333, 3).to(device)
model.load_state_dict(torch.load('/Users/rodrigo/PycharmProjects/ising_gnn_deploy/model_params_333_TRUE.pth', map_location=device))
model.eval()

# Define prediction function
def predict(model, input):
    train_input, val_input = ut.create_graph(pd.DataFrame(input).T, pd.DataFrame(input).T,
                                             pd.DataFrame([1]), pd.DataFrame([1]), size=190, method={'knn': 15})
    train_loader, val_loader = ut.create_batch(train_input, val_input, batch_size=1)
    for y_i in val_loader:
        y_pred_aux = model(y_i)[1].detach().numpy().ravel()
    return y_pred_aux

# Streamlit app
st.title("Ising Model Temperature Prediction")

uploaded_file = st.file_uploader("Choose a connectivity matrix file (txt format)...", type=["txt"])

if uploaded_file is not None:
    # Read the contents of the file
    contents = uploaded_file.readlines()
    # Convert the contents of the file to an array
    data_array = np.array([float(x) for x in contents])
    # Predict the temperature
    temperature = round(predict(model, data_array)[0], 2)
    # Display the prediction
    st.write(f'Connectome temperature is {temperature}')


# import streamlit as st
# import io
# from PIL import Image
# from urllib.request import urlopen
# import torch
# from torchvision import models, transforms
#
# # Load a pre-trained DenseNet model from torchvision.models
# model = models.densenet121(pretrained=True)
# # Switch the model to evaluation mode
# model.eval()
#
# # Load the class labels from a file
# class_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# class_labels = urlopen(class_labels_url).read().decode("utf-8").split("\n")
#
# # Define the transformation of the input image
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
#
# def predict(model, transform, image, class_labels):
#
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#     # Transform the image and convert it to a tensor
#     image_tensor = transform(image).unsqueeze(0)
#
#     # Pass the image through the model
#     with torch.no_grad():
#         output = model(image_tensor)
#
#     # Select the class with the highest probability and look up the name
#     m = torch.nn.Softmax(dim=1)
#     class_prob = round(float(m(output).max()), 3) * 100
#     class_id = torch.argmax(output).item()
#     class_name = class_labels[class_id]
#
#     # Return the class name
#     return f"{class_name} - Confidence: {class_prob}%"
#
#
# # Streamlit app
# st.title("Image Classification with DenseNet")
#
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#
# if uploaded_file is not None:
#     # Read the image file
#     image_bytes = uploaded_file.read()
#     image = Image.open(io.BytesIO(image_bytes))
#
#     # Display the image
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")
#
#     # Predict the class
#     class_name = predict(model, transform, image, class_labels)
#
#     # Display the prediction
#     st.write(class_name)