import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.features import shapes
import shapely.geometry as geom


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Example usage
edges = preprocess_image('path_to_chandrayaan_image.jpg')
plt.imshow(edges, cmap='gray')
plt.title('Preprocessed Image')
plt.show()


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # Assuming 3 terrain types
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and summarize the model
model = create_model()
model.summary()

def load_data():
    # Load and preprocess your training data here
    # For simplicity, using dummy data
    X_train = np.random.rand(1000, 128, 128, 1)  # 1000 grayscale images of size 128x128
    y_train = np.random.randint(3, size=(1000,))  # Random labels for 3 terrain types
    return X_train, y_train

X_train, y_train = load_data()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


def detect_hazards(elevation_data, threshold):
    hazard_map = elevation_data > threshold  # Example thresholding for steep slopes
    return hazard_map

# Example usage with dummy elevation data
elevation_data = np.random.rand(128, 128) * 100  # Dummy elevation data
hazard_map = detect_hazards(elevation_data, 50)
plt.imshow(hazard_map, cmap='gray')
plt.title('Hazard Map')
plt.show()


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(start, goal, hazard_map):
    open_list = []
    heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        _, current = heappop(open_list)

        if current == goal:
            break

        for next in neighbors(current, hazard_map):
            new_cost = cost_so_far[current] + cost(current, next, hazard_map)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heappush(open_list, (priority, next))
                came_from[next] = current

    return reconstruct_path(came_from, start, goal)

def neighbors(node, hazard_map):
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = []
    for d in dirs:
        neighbor = (node[0] + d[0], node[1] + d[1])
        if 0 <= neighbor[0] < hazard_map.shape[0] and 0 <= neighbor[1] < hazard_map.shape[1]:
            if not hazard_map[neighbor]:
                result.append(neighbor)
    return result

def cost(current, next, hazard_map):
    return 1  # Example: uniform cost

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# Example usage
start = (0, 0)
goal = (127, 127)
path = a_star_search(start, goal, hazard_map)
path_x, path_y = zip(*path)
plt.imshow(hazard_map, cmap='gray')
plt.plot(path_y, path_x, color='blue')
plt.title('Optimized Route')
plt.show()

# Integrate all parts to create a comprehensive workflow

def main(image_path, elevation_data_path, start, goal):
    # Step 1: Preprocess Image
    edges = preprocess_image(image_path)

    # Step 2: Classify Terrain
    # Note: Here we assume model is already trained and loaded
    terrain_class = model.predict(np.expand_dims(edges, axis=0))

    # Step 3: Load Elevation Data
    with rasterio.open(elevation_data_path) as src:
        elevation_data = src.read(1)

    # Step 4: Detect Hazards
    hazard_map = detect_hazards(elevation_data, threshold=50)

    # Step 5: Optimize Route
    path = a_star_search(start, goal, hazard_map)
    
    # Step 6: Visualize Results
    path_x, path_y = zip(*path)
    plt.imshow(hazard_map, cmap='gray')
    plt.plot(path_y, path_x, color='blue')
    plt.title('Optimized Route')
    plt.show()

# Example usage
main('path_to_chandrayaan_image.jpg', 'path_to_elevation_data.tif', (0, 0), (127, 127))
