import cv2 as cv
import numpy as np

def compute_image_gradients(img):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    grad_x = cv.filter2D(img, -1, x_kernel)
    grad_y = cv.filter2D(img, -1, y_kernel)
    return grad_x, grad_y

def compute_temporal_gradients(img1, img2):
    t_kernel = np.ones((2, 2)) * 0.25
    grad_t = cv.subtract(cv.filter2D(img2, -1, t_kernel), cv.filter2D(img1, -1, t_kernel))
    return cv.threshold(grad_t, 2, 150, cv.THRESH_BINARY)[1]

def check_values_convergence(u, v, u_prev, v_prev, threshold):
    return np.mean(np.abs(u - u_prev)) < threshold and np.mean(np.abs(v - v_prev)) < threshold

def compute_optical_flow(img1, img2, alpha, n_iter=50):
    grad_x, grad_y = compute_image_gradients(img1)
    grad_t = compute_temporal_gradients(img1, img2)
    cv.imshow("grad_x", grad_x)
    cv.imshow("grad_y", grad_y)
    cv.imshow("grad_t", grad_t)

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)

    for _ in range(n_iter):
        u_avg = cv.filter2D(u, -1, avg_kernel)
        v_avg = cv.filter2D(v, -1, avg_kernel)

        u = u_avg - grad_x * ((grad_x * u_avg + grad_y * v_avg + grad_t) / (4*alpha**2 + grad_x**2 + grad_y**2))
        v = v_avg - grad_y * ((grad_x * u_avg + grad_y * v_avg + grad_t) / (4*alpha**2 + grad_x**2 + grad_y**2))
        # if check_values_convergence(u, v, u_avg, v_avg, 50):
        #     break
    return u, v

def draw_optical_flow(img, u, v, step=16):
    u = u.astype(np.int16)
    v = v.astype(np.int16)
    h, w = img.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            cv.arrowedLine(img, (x, y), (x + u[y, x], y + v[y, x]), (0, 255, 0))
    return img

def video_optical_flow(video_path):
    cap = cv.VideoCapture(video_path)
    ret, frame1 = cap.read()
    frame1 = cv.GaussianBlur(frame1, (5, 5), 0)
    frame1 = cv.resize(frame1, (0,0), fx=0.1, fy=0.1)
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    while(1):
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame2 = cv.GaussianBlur(frame2, (5, 5), 0)
        frame2 = cv.resize(frame2, (0,0), fx=0.1, fy=0.1)
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        u, v = compute_optical_flow(prvs, next, 15, 100)
        img = draw_optical_flow(frame2, u, v, step=8)
        cv.imshow('frame2', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next
    cv.destroyAllWindows()

if __name__ == "__main__":
    video_optical_flow("video4.mp4")
    # Read images
    # img1 = cv.imread("./261.jpg", 0)
    # img2 = cv.imread("./262.jpg", 0)
    # if img1 is None or img2 is None:
    #     raise Exception("Could not read the images.")
    
    # img1 = cv.resize(img1, (0,0), fx=0.5, fy=0.5)
    # img2 = cv.resize(img2, (0,0), fx=0.5, fy=0.5)
    # img1_copy = img1.copy()
    # img2_copy = img2.copy()
    # img1 = cv.GaussianBlur(img1, (5, 5), 0)
    # img2 = cv.GaussianBlur(img2, (5, 5), 0)
    # cv.imshow("Original image", img1_copy)
    # # cv.imshow("Original image", img2_copy)
    
    # # Compute optical flow
    # u, v = compute_optical_flow(img1, img2, 15, 100)

    # # Draw optical flow
    # img1_copy = cv.cvtColor(img1_copy, cv.COLOR_GRAY2BGR)
    # img = draw_optical_flow(img1_copy, u, v, step=8)

    # # Show result
    # cv.imshow("Optical flow", img)

    # cv.waitKey(0)