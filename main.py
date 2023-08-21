# Interpolating frames for a video

# we have a video with 30 frames per second

# problem: it is not smooth enough for our eyes

# solution: we can interpolate frames and increase the frame rate to make video smoother

# interpolation is a process of generating new data points within the range of a discrete set of known data points

# extrapolation is a process of generating new data points outside the range of a discrete set of known data points

import cv2
import numpy as np

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
    print(len(grey_frame[0]))
    gradient_image[offset:height-offset, offset:width-offset] = \
    np.array([
        [convolution2D(grey_frame[y-offset:y+offset+1, x-offset:x+offset+1], kernel) 
        for x in range(offset, width-offset)] 
        for y in range(offset, height-offset)
    ])

    return gradient_image

def otsu_threshold(img, block_size, c):
    pass

def gaussian_threshold(img, block_size, c):
    # Check that the block size is odd and nonnegative
    assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

    # Calculate the local threshold for each pixel using Gaussian weighted mean
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
            
            # Calculate Gaussian weighted mean as the threshold for the current pixel
            weights = np.exp(-(np.square(np.arange(-block_size // 2, block_size // 2 + 1)) / (2 * c ** 2)))
            weights = weights / np.sum(weights)  # Normalize weights
            thresh = np.sum(block * weights)  # Apply weighted mean for the current pixel

            if img[i, j] >= thresh:
                binary[i, j] = 255

    return binary



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

    return binary


def good_features_to_track(prev_grey_frame, threshold_func, method):

    if method == 'opencv':
        corners = cv2.goodFeaturesToTrack(
            prev_grey_frame,
            maxCorners=10000,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )
        return corners
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
        Ix = gradient(prev_grey_frame, sobel_x)
        Iy = gradient(prev_grey_frame, sobel_y)
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

        thresholded_img = threshold_func(response_matrix, len(sobel_x), 4)

        corners = []
        for x in range(offset, width - offset):
            for y in range(offset, height - offset):
                if thresholded_img[y, x] != 0:
                    corners.append([(x, y)])
    
    corners = np.intp(corners)

    marked_frame = cv2.cvtColor(prev_grey_frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale frame to BGR, didn't work otherwise
    for corner in corners:
        x, y = corner[0]
        cv2.circle(marked_frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imwrite("corner_img.jpg", marked_frame)

    breakpoint()



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
            prev_corner_points = good_features_to_track(prev_grey_frame, threshold_func=gaussian_threshold, method='opencv')

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