import os
import pickle
import shutil
import matplotlib.pyplot as plt
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
import os
from sklearn.cluster import KMeans
# from mpl_toolkits.mplot3d import Axes3Ds
# Load your data
# vectors = pickle.load(open("vectors.pkl", "rb"))
# paths = pickle.load(open("paths.pkl", "rb"))

# Calculate pairwise distances between vectors (you can choose a different distance metric)
# distances = ssd.pdist(vectors)

# # Perform K-Means clustering with 8 clusters (you can adjust the number of clusters)
# kmeans = KMeans(n_clusters=8)
# cluster_labels = kmeans.fit_predict(vectors)

# # Create a dictionary to store lists of image paths for each cluster
# cluster_images = {i: [] for i in range(8)}  # Initialize for the number of clusters

# # Iterate through vectors and collect image paths for each cluster
# for i, cluster_id in enumerate(cluster_labels):
#     image_path = paths[i]  # Get the image path for the current vector
#     cluster_images[cluster_id].append((vectors[i], image_path))

# # Create scatter plots for each cluster
# # Create a 3D scatter plot for each cluster
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# for cluster_id, data in cluster_images.items():
#     cluster_vectors, image_paths = zip(*data)
#     cluster_vectors = list(cluster_vectors)  # Convert to a list of vectors

#     # Extract X, Y, and Z coordinates from the vectors for the scatter plot
#     x = [vector[0] for vector in cluster_vectors]
#     y = [vector[1] for vector in cluster_vectors]
#     z = [vector[2] for vector in cluster_vectors]

#     # Create a 3D scatter plot
#     ax.scatter(x, y, z, label=f'Cluster {cluster_id}')

# # Add labels and legend
# ax.set_xlabel('X-Axis Label')
# ax.set_ylabel('Y-Axis Label')
# ax.set_zlabel('Z-Axis Label')
# ax.legend()

# # Show the 3D scatter plot or save it to a file
# plt.show()
# To save the scatter plot to a file, use plt.savefig("scatter_plot.png")

# Create folders for each cluster and copy images to their respective folders
# output_folder = "kmeans_clusters"  # Change the output folder name as needed
# os.makedirs(output_folder, exist_ok=True)

# for cluster_id, image_paths in cluster_images.items():
#     cluster_folder = os.path.join(output_folder, str(cluster_id))
#     os.makedirs(cluster_folder, exist_ok=True)
#     for image_path in image_paths:
#         image_name = os.path.basename(image_path)
#         target_path = os.path.join(cluster_folder, image_name)
#         shutil.copy(image_path, target_path)

# Load your data
# vectors = pickle.load(open("vectors.pkl", "rb"))
# paths = pickle.load(open("paths.pkl", "rb"))

# Calculate pairwise distances between vectors (you can choose a different distance metric)
import os
import pickle
import shutil
import matplotlib.pyplot as plt
from PIL import Image  # Pillow library for image handling

# Load your data
vectors = pickle.load(open("vectors.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))

# Calculate pairwise distances between vectors (you can choose a different distance metric)
distances = ssd.pdist(vectors)

# Perform K-Means clustering with 8 clusters (you can adjust the number of clusters)
kmeans = KMeans(n_clusters=7)
cluster_labels = kmeans.fit_predict(vectors)

# Create a dictionary to store lists of image paths for each cluster
cluster_images = {i: [] for i in range(7)}  # Initialize for the number of clusters

# Iterate through vectors and collect image paths for each cluster
for i, cluster_id in enumerate(cluster_labels):
    image_path = paths[i]  # Get the image path for the current vector
    cluster_images[cluster_id].append(image_path)

# Create scatter plots for each cluster
for cluster_id, image_paths in cluster_images.items():
    cluster_vectors = np.array(vectors)[cluster_labels == cluster_id]  # Get cluster vectors
    num_points = len(cluster_vectors)

    # Extract X and Y coordinates from the vectors for the scatter plot
    x = cluster_vectors[:, 0]
    y = cluster_vectors[:, 1]

    # Create a scatter plot with transparency to reduce overlap
    plt.scatter(x, y, label=f'Cluster {cluster_id}', alpha=0.5)

    # Show the associated images larger alongside the scatter plot
    fig, ax = plt.subplots(num_points, 1, figsize=(4, 4 * num_points))
    fig.suptitle(f'Cluster {cluster_id} Images')
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        ax[i].imshow(img)
        ax[i].axis('off')

# Add labels and legend to the scatter plot
plt.xlabel('X-Axis Label')
plt.ylabel('Y-Axis Label')
plt.legend()

# Show the scatter plot
plt.show()