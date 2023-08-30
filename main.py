# Interpolating frames for a video

# we have a video with 30 frames per second

# problem: it is not smooth enough for our eyes

# solution: we can interpolate frames and increase the frame rate to make video smoother

# interpolation is a process of generating new data points within the range of a discrete set of known data points

# extrapolation is a process of generating new data points outside the range of a discrete set of known data points

import cv2
import numpy as np
import os
import csv
from scipy import signal

# todo: normalise responses etc, goal is parity w/ opencv

def convolution2D(window, kernel):
    return np.sum(window*kernel)

def gradient(grey_frame, kernel):
    # kernel = [[-1, 0, 1], 
    #           [-2, 0, 2], 
    #           [-1, 0, 1]]
    # grey_frame = np.array([
    #               [1, 2, 3, 4, 5 ], 
    #               [6, 7, 8, 9, 10], 
    #               [11,12,13,14,15],
    #               [16,17,18,19,20],
    #               [21,22,23,24,25]
    #               ])
    
    offset = len(kernel) // 2
    height, width = grey_frame.shape
    # TODO: make sure this is correct (check the size of the gradient image)
    gradient_image = np.zeros((height, width))
    gradient_image[offset:height-offset, offset:width-offset] = \
    np.array([
        [convolution2D(grey_frame[y-offset:y+offset+1, x-offset:x+offset+1], kernel) 
        for x in range(offset, width-offset)] 
        for y in range(offset, height-offset)
    ])

    return gradient_image


def otsu_threshold(img, block_size, c):
    pass


def adaptive_threshold_mean(img, block_size, c):
    # Check that the block size is odd and nonnegative
    assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

    # Calculate the local threshold for each pixel
    height, width = img.shape
    binary = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # Calculate the local threshold using a square neighborhood centered at (i, j)
            x_min = max(0, i - block_size // 2)
            y_min = max(0, j - block_size // 2)
            x_max = min(height - 1, i + block_size // 2)
            y_max = min(width - 1, j + block_size // 2)
            block = img[x_min:x_max+1, y_min:y_max+1]
            thresh = np.mean(block) - c
            if img[i, j] >= thresh:
                binary[i, j] = 255

            binary[i,j] = 1 if abs(img[i,j]) > 0.0026 else 0 #tested between 0.00255 & ~259 but saw little change compared to ~26

    return binary

def get_corners_from_threshold(thresholded_img, offset, height, width):
    corners = []
    for x in range(offset, width - offset):
        for y in range(offset, height - offset):
            if thresholded_img[y, x] != 0:
                corners.append([(x, y)])
    return corners


def good_features_to_track(prev_grey_frame, threshold_func, method):

    if method == 'opencv':
        maxCorners = 5000
        qualityLevel=0.1
        minDistance=1
        corners = cv2.goodFeaturesToTrack(
            prev_grey_frame,
            maxCorners=maxCorners,
            qualityLevel=qualityLevel,
            minDistance=minDistance,
            blockSize=3,
        )
        
    elif method == 'custom':
        # Calculate flow derivatives for x and y using the gradient function and Sobel kernels
        sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], 
                            [0, 0, 0], 
                            [1, 2, 1]])

        offset = len(sobel_x) // 2
        height, width = prev_grey_frame.shape
        print("calculating gradients...")
        mode = 'same'
        Ix = signal.convolve2d(prev_grey_frame, sobel_x, boundary='symm', mode=mode)
        Iy = signal.convolve2d(prev_grey_frame, sobel_y, boundary='symm', mode=mode)

        # Ix = gradient(prev_grey_frame, sobel_x)
        # Iy = gradient(prev_grey_frame, sobel_y)
        print("gradients calculated")

        # Calculate Ix^2, Iy^2, IxIy, IxIt, IyIt
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        IxIy = Ix * Iy
        IxIt = Ix * (-prev_grey_frame)
        IyIt = Iy * (-prev_grey_frame)

        # Sum squared derivatives in a window
        corner_response = np.zeros(prev_grey_frame.shape)

        k = 0.04
    
        response_matrix = np.zeros((height, width))

        if os.path.isfile("response_matrix.csv"):
            with open("response_matrix.csv", "r") as f:
                reader = csv.reader(f)
                data = list(reader)
                response_matrix = np.array(data, dtype=float)

        else:
            for x in range(offset, width - offset):
                print(f"{x}/{width}")
                for y in range(offset, height - offset):
                    Ix2_sum, Iy2_sum, IxIy_sum = \
                        [np.sum(arr[y - offset: y + offset + 1, x - offset: x + offset + 1]) for arr in [Ix2, Iy2, IxIy]]

                    harris_matrix = np.array([[Ix2_sum, IxIy_sum], [IxIy_sum, Iy2_sum]])
                    trace = Ix2_sum + Iy2_sum
                    det = Ix2_sum * Iy2_sum - IxIy_sum**2
                    response = det - k * trace**2
                    response_matrix[y - offset, x - offset] = response
            np.savetxt("response_matrix.csv", response_matrix, delimiter=",")

        norm = np.linalg.norm(response_matrix, 1)

        response_matrix_normalised = response_matrix/norm

        c=4
        thresholded_img = threshold_func(response_matrix_normalised, len(sobel_x), c)
        corners = get_corners_from_threshold(thresholded_img, offset, height, width)

    
    corners = np.intp(corners)

    marked_frame = cv2.cvtColor(prev_grey_frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale frame to BGR, didn't work otherwise
    
    for corner in corners:
        x, y = corner[0]
        cv2.circle(marked_frame, (x, y), 1, (0, 0, 255), -1)


    if method=="opencv":
        cv2.imwrite(f"corner_img_{method}_{maxCorners}_{qualityLevel}_{minDistance}.jpg", marked_frame)
    
    else:
        cv2.imwrite(f"corner_img_{method}_{c}_{k}.jpg", marked_frame)

    breakpoint()
    return corners



            # https://towardsdatascience.com/hands-on-otsu-thresholding-algorithm-for-image-background-segmentation-using-python-9fa0575ac3d2
            # https://medium.com/geekculture/image-thresholding-from-scratch-a66ae0fb6f09






def get_optical_flow(cap):
    # get first frame
    successfully_read, frame = cap.read()
    prev_grey_frame = None

    # TODO: convert to grey manually as well
    if successfully_read:
        prev_grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas Kanade optical flow
    lucas_kanade_params = dict(
        winSize=(15, 15), 
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    optical_flow_frames = {
        "corner_points": [],
        "status": [],
        "err": []
    }

    frame_counter = 0 #TESTING

    while True:
        # corners we're tracking from
        prev_corner_points = None  # Initialize to None
        if frame_counter == 40:  # TESTING only run on the 40th frame because it's lit up
            prev_corner_points = good_features_to_track(prev_grey_frame, threshold_func=adaptive_threshold_mean, method='custom')

            breakpoint()
        # get next frame and convert to grey
        successfully_read, frame = cap.read()
        if not successfully_read:
            break
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_corner_points is not None:
            # TODO do this mathematically
            corner_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_grey_frame, grey_frame, prev_corner_points, None, **lucas_kanade_params
            )

            optical_flow_frames['corner_points'].append(corner_points)
            optical_flow_frames['status'].append(status)
            optical_flow_frames['err'].append(err)

        prev_grey_frame = grey_frame.copy()
        frame_counter += 1 

    return optical_flow_frames
        


def main():
    # TODO: read video
    file_name = "example.mp4"
    cap = cv2.VideoCapture(file_name)
    optical_flow_frames = get_optical_flow(cap)
    
    # TODO: mathematically derive optical flow
    # TODO: we need to use optical flow to interpolate frames
    
    


    # average frames
    # write video
    
    cap.release()




if __name__ == '__main__':
    main()
    # https://www.geekering.com/programming-languages/python/brunorsilva/harris-corner-detector-python/
    # play arounf with thresholding function
    # plot corners on image itself and compare
    # optimise code to make it faster
    # compare our optical flow method to the standard one.