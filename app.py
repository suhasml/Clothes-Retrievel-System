import streamlit as st
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os
import pickle

# Load the precomputed image features and filenames
features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

st.title('Clothes Retrieval System')

def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized

def recommend(features, features_list, img_files_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Extract features from the uploaded image
    input_features = extract_img_features(uploaded_file, model)
    
    # Get recommendations based on the input features
    recommended_indices = recommend(input_features, features_list, img_files_list)
    
    # Display the most similar images as recommendations
    st.header("Here are the Similar Images:")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for idx, col in enumerate([col1, col2, col3, col4, col5]):
        with col:
            st.image(img_files_list[recommended_indices[0][idx]])
