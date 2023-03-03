import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u) + np.square(v))

def get_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.arctan2(v, u)


def mark_movement(before_image: np.ndarray, u: np.ndarray, v: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    magnitude = get_magnitude(u, v)

    marked_image = before_image.copy()
    marked_image[magnitude > threshold] = [0, 0, 255]
    marked_image[magnitude <= threshold] = [0, 0, 0]

    return marked_image

def segment_movement(before_image: np.ndarray, u: np.ndarray, v: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    movement_data = np.array([get_magnitude(u, v), get_direction(u, v)])
    movement_data = np.transpose(movement_data)

    segmented_image = np.array(np.zeros(before_image.shape[:2]), dtype=np.uint8)
    # image_regions = np.array([segmented_image])
    # print(np.append(image_regions, np.array([segmented_image]), axis=0).shape)
    # add the first pixel to the first region

    X = movement_data
    X = X.reshape(-1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10
    ret,label,center=cv2.kmeans(X,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((segmented_image.shape))

    plt.axis('off')
    plt.imshow(result_image)
    # kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    return segmented_image
