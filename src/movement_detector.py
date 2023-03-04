import numpy as np
import cv2

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

def segment_movement(before_image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    movement_data = np.array([get_magnitude(u, v), get_direction(u, v)])
    print(movement_data.shape)
    movement_data = np.reshape(movement_data, (640, 352, 2))

    twoDimage = movement_data.reshape(-1, 2).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10

    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((movement_data.shape))

    for i in range(result_image.shape[0]):
        for j in range(result_image.shape[1]):
            if result_image[i][j].all() == 0:
                before_image[i][j] = [0, 0, 0]
            if result_image[i][j].all() == 1:
                before_image[i][j] = [0, 0, 255]

    return before_image
