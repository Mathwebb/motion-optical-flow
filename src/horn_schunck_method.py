import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve as filter2
import os

def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg

def draw_optical_flow(img, u, v, step=16):
    u = u.astype(np.int16)
    v = v.astype(np.int16)
    h, w = img.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            if np.linalg.norm([u[y, x], v[y, x]]) > 5:
                cv2.arrowedLine(img, (x, y), (x + u[y, x], y + v[y, x]), (0, 0, 255))
    return img

def draw_quiver(u,v,beforeImg):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(beforeImg)

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.show()

def compute_derivatives(img1, img2, channelsToUse=[0, 1, 2]):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    if len(img1.shape) == 2:
        fx = filter2(img1,x_kernel) + filter2(img2,x_kernel)
        fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
        ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)
        return [fx,fy, ft]
    else:
        fx = np.zeros(img1.shape)
        fy = np.zeros(img1.shape)
        ft = np.zeros(img1.shape)
        for channel in channelsToUse:
            fx[:,:,channel] = filter2(img1[:,:,channel],x_kernel) + filter2(img2[:,:,channel],x_kernel)
            fy[:,:,channel] = filter2(img1[:,:,channel], y_kernel) + filter2(img2[:,:,channel], y_kernel)
            ft[:,:,channel] = filter2(img1[:,:,channel], -t_kernel) + filter2(img2[:,:,channel], t_kernel)
        return [fx,fy, ft]

def compute_horn_schunck(img1, img2, alpha, delta=10**-1, n_iter=300, channelsToUse=[0, 1, 2]):

    img1 = img1.astype(float)
    img2 = img2.astype(float)

    img1  = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    u = np.zeros((img1.shape[0], img1.shape[1]))
    v = np.zeros((img1.shape[0], img1.shape[1]))
    fx, fy, ft = compute_derivatives(img1, img2, channelsToUse)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    for iter_counter in range(n_iter):
        print("iteration number: ", iter_counter)
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)

        if len(img1.shape) == 2:
            p = fx * u_avg + fy * v_avg + ft
            d = 4 * alpha**2 + fx**2 + fy**2
            prev = u

            u = u_avg - fx * (p / d)
            v = v_avg - fy * (p / d)
        else:
            u_arr = [0, 0, 0]
            v_arr = [0, 0, 0]
            for i in channelsToUse:
                p = fx[:,:,i] * u_avg + fy[:,:,i] * v_avg + ft[:,:,i]
                d = 4 * alpha**2 + fx[:,:,i]**2 + fy[:,:,i]**2
                prev = u

                u_arr[i] = (u_avg - fx[:,:,i] * (p / d))
                v_arr[i] = (v_avg - fy[:,:,i] * (p / d))
            u_sum = np.zeros((img1.shape[0], img1.shape[1]))
            v_sum = np.zeros((img1.shape[0], img1.shape[1]))
            for i in channelsToUse:
                u_sum += u_arr[i]
                v_sum += v_arr[i]
            u = u_sum / len(channelsToUse)
            v = v_sum / len(channelsToUse)

        diff = np.linalg.norm(u - prev, 2)
        if  diff < delta or iter_counter > n_iter:
            break

    return u, v

def create_entry_frames(video_name):
    cap = cv2.VideoCapture(video_name)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    i = 0  
    for filename in os.listdir("entry_frames"):
        os.remove("entry_frames/" + filename)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite("entry_frames/frame%d.jpg" % i, frame)
            i += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_entry_frames("assets/videos/videox5.mp4")

    img2 = cv2.imread("entry_frames/frame184.jpg")
    img1 = cv2.imread("entry_frames/frame185.jpg")
    if img1 is None or img2 is None:
        raise Exception("Could not read the images.")
    
    img1 = cv2.resize(img1, (0,0), fx=1, fy=1)
    img2 = cv2.resize(img2, (0,0), fx=1, fy=1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)
    img1_copy = img1.copy()
    img2_copy = img2.copy()


    u,v = compute_horn_schunck(img1, img2, alpha = 15, delta = 10**-1, n_iter = 900, channelsToUse = [0, 1, 2])

    # img = cv2.cvtColor(img1_copy, cv2.COLOR_GRAY2BGR)
    img = img1_copy

    draw_quiver(u, v, img)

    cv2.waitKey(0)