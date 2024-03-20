import math
import os
import shutil

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model

from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
# Ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

# Ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = img.resize((224,224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Xu ly : ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Trich dac trung
    vector = model.predict(img_tensor)[0]
    # Chuan hoa vector = chia chia L2 norm (tu google search)
    vector = vector / np.linalg.norm(vector)
    return vector

# Dinh nghia anh can tim kiem
search_image = "testimage/147.jpeg"

# Khoi tao model
model = get_extract_model()

# Trich dac trung anh search
search_vector = extract_vector(model, search_image)

# Load 4700 vector tu vectors.pkl ra bien
vectors = pickle.load(open("vectors.pkl","rb"))
paths = pickle.load(open("paths.pkl","rb"))

# Tinh khoang cach tu search_vector den tat ca cac vector
distance = np.linalg.norm(vectors - search_vector, axis=1)

# Sap xep va lay ra K vector co khoang cach ngan nhat
K = 9
ids = np.argsort(distance)[:K]

# Tao oputput
# nearest_image = [(paths[id], distance[id]) for id in ids]
nearest_image = []
path1 = []
dist1 = []
# dist2 = dist1.replace("\\","/")
for id in ids:
    path = paths[id]
    # newpath = path.replace("\\","/")
    new_path = os.path.abspath(path)
    path1.append(new_path)
    dist = distance[id]

    dist1.append(dist)
    nearest_image.append((path,dist))
# print(path1)

# for i in path1:
#     t= 0
#     fig, ax = plt.subplots()
#     img = Image.open(i)
#     ax.imshow(img)
#     plt.show()
    
    # image = Image.open(i)
    # image.show()
    # plt.imshow(Image.open(i))

# os.makedirs("similar_images_folder", exist_ok= True)
image_list = path1
similarity_scores = dist1
thisfolder = "similar_images_folder"
# for i,img in enumerate(path1):
#     image = Image.fromarray(img)
#     image.save(os.path.join(thisfolder, f"image_{i}.png"))
    
for img in path1:
    filename = os.path.basename(img)
    target_path = os.path.join(thisfolder, filename)
    shutil.copy(img, target_path)
    # image.save(os.path.join(thisfolder, img))

# for i, image_name in enumerate(image_list):
#     similarity_score = similarity_scores[i]
#     if similarity_score >= 0.1: 
#         os.rename(image_name, os.path.join("similar_images_folder",image_name))

# Ve len man hinh cac anh gan nhat do

import matplotlib.pyplot as plt

axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize=(10,5))
# images = []
for id in range(K):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id+1))
    axes[-1].set_title(draw_image[1])
    # plt.imshow(Image.open(draw_image[0]))
    # t = plt.imshow(Image.open()
    # os.makedirs(os.path.join('clusters', str(draw_image[0])))
    # image.save(os.path.join('clusters', str(draw_image[0]), image.filename))
    

# images_folder = os.path.join('.', 'images')

# for filename in os.listdir('new_images'):
#     src_path = os.path.join('new_images', filename)
#     dst_path = os.path.join(images_folder, filename)

#     os.copy(src_path, dst_path)

# fig.tight_layout()
# plt.show()

# Example similarity threshold (adjust as needed)
similarity_threshold = 0.95

# Create a list to store unique representative arrays
unique_arrays = []

# Iterate through the arrays
for array in vector_groups:
    is_unique = True

    for unique_array in unique_arrays:
        # Calculate the similarity between the current array and a unique array
        similarity = sum(a == b for a, b in zip(array, unique_array)) / len(array)

        if similarity >= similarity_threshold:
            is_unique = False
            break

    if is_unique:
        unique_arrays.append(array)

# unique_arrays now contains only unique representative arrays
# print(unique_arrays)
# Calculate the number of arrays in unique_arrays
num_arrays = len(unique_arrays)

# Print the result
print("Number of unique arrays:", num_arrays)