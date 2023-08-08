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
    height = grey_frame.shape[0]
    width = grey_frame.shape[1]
    # TODO: make sure this is correct (check the size of the gradient image)
    gradient_image = np.zeros((height, width))
    print(len(grey_frame[0]))
    # https://medium.com/@thepyprogrammer/2d-image-convolution-with-numpy-with-a-handmade-sliding-window-view-946c4acb98b4
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            window = grey_frame[y-offset:y+offset+1, x-offset:x+offset+1]

            convolution = convolution2D(window, kernel)
            # TODO: add convolution to gradient image
            gradient_image[y, x] = convolution
    
    return gradient_image
            

# ... (other code remains unchanged)

def good_features_to_track(prev_grey_frame, max_corners=200, quality_level=0.01, min_distance=30, blockSize=3):
    # Calculate flow derivatives for x and y using the gradient function and Sobel kernels
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    offset = len(sobel_x) // 2
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

    k=0.04
    response_matrix = np.zeros((prev_grey_frame.shape[0], prev_grey_frame.shape[1]))
    
    for y in range(offset, prev_grey_frame.shape[1] - offset):
        print(f"{y}/{prev_grey_frame.shape[1]}")
        for x in range(offset, prev_grey_frame.shape[0] - offset):
            Ix2_sum = np.sum(Ix2[ y - offset: y + offset + 1, x - offset: x + offset + 1])
            Iy2_sum = np.sum(Iy2[y - offset: y + offset + 1, x - offset: x + offset + 1])
            IxIy_sum = np.sum(IxIy[y - offset: y + offset + 1, x - offset: x + offset + 1])
            IxIt_sum = np.sum(IxIt[y - offset: y + offset + 1, x - offset: x + offset + 1])
            IyIt_sum = np.sum(IyIt[y - offset: y + offset + 1, x - offset: x + offset + 1])

            harris_matrix = np.array([[Ix2_sum, IxIy_sum], [IxIy_sum, Iy2_sum]])
            trace = Ix2_sum + Iy2_sum
            det = Ix2_sum * Iy2_sum - IxIy_sum**2
            response = det - k*trace**2
            response_matrix[y - offset, x - offset] = response

    breakpoint()
    # thresholding



            





def get_optical_flow(cap):

    # get first frame
    successfully_read, frame = cap.read()
    prev_grey_frame = None

    # TODO: convert to grey manually as well
    if successfully_read:
        prev_grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas Kanade optical flow
    # TODO see other parameters lk can take
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

    while True:
        # corners we're tracking from
        prev_corner_points = good_features_to_track(prev_grey_frame)
        prev_corner_points = cv2.goodFeaturesToTrack(prev_grey_frame,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
        
        # get next frame and convert to grey
        successfully_read, frame = cap.read()
        if not successfully_read:
            break
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        # TODO do this mathematically
        corner_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_grey_frame, grey_frame, prev_corner_points, None, **lucas_kanade_params
        )

        optical_flow_frames['corner_points'].append(corner_points)
        optical_flow_frames['status'].append(status)
        optical_flow_frames['err'].append(err)
        
        prev_grey_frame = grey_frame.copy()
    
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
    arr1 = np.array([[1,2,3],[4,5,6], [7,8,9]])
    win1 = np.array([[1,1],[1,1]])
    breakpoint()
    main()
    # https://www.geekering.com/programming-languages/python/brunorsilva/harris-corner-detector-python/
    