import math
import os
import shutil
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
# Load 4700 vector tu vectors.pkl ra bien
vectors = pickle.load(open("vectors.pkl","rb"))
paths = pickle.load(open("paths.pkl","rb"))

# vectors.info()
# Calculate pairwise distances between vectors (you can choose a different distance metric)
# distances = ssd.pdist(vectors)

# # Perform K-Means clustering with 6 clusters (you can adjust the number of clusters)
kmeans = KMeans(n_clusters=7)
cluster_labels = kmeans.fit_predict(vectors)

# Create a dictionary to store lists of image paths for each cluster
cluster_images = {}

# Iterate through vectors and collect image paths for each cluster
for i, cluster_id in enumerate(cluster_labels):
    image_path = paths[i]  # Get the image path for the current vector
    if cluster_id not in cluster_images:
        cluster_images[cluster_id] = []
    cluster_images[cluster_id].append(image_path)

# Create folders for each cluster and copy images to their respective folders
output_folder = "kmeans_clusters"  # Change the output folder name as needed
os.makedirs(output_folder, exist_ok=True)

for cluster_id, image_paths in cluster_images.items():
    cluster_folder = os.path.join(output_folder, str(cluster_id))
    os.makedirs(cluster_folder, exist_ok=True)
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        target_path = os.path.join(cluster_folder, image_name)
        shutil.copy(image_path, target_path)



# Assuming X is your numpy array of vectors
# X = vectors  # Your data

# # Create an empty list to store the SSE (Sum of Squared Errors) for different values of K
# sse = []

# # Loop through a range of K values (e.g., from 1 to 10) and fit K-means for each K
# for k in range(1, 25):
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
#     sse.append(kmeans.inertia_)

# # Plot the Elbow graph
# plt.figure()
# plt.plot(range(1, 25), sse, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters (K)')
# plt.ylabel('SSE')
# plt.show()


# import seaborn as sns

# clusters = []
# for i in range(1, 25):
#   km = KMeans(n_clusters=i).fit(X)
#   clusters.append(km.inertia_)

# fig, ax = plt.subplots(figsize=(12, 25))
# sns.lineplot(x=list(range(1, 25)), y=clusters, ax=ax)
# ax.set_title('Đồ thị Elbow')
# ax.set_xlabel('Số lượng nhóm')
# ax.set_ylabel('Giá trị Inertia')
# plt.show()
# plt.cla()
# Qua đồ thị trên, chúng ta thấy số lượng cluster thích hợp là từ 3 đến 5


# print(vectors[68])