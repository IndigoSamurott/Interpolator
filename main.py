# Interpolating frames for a video

# we have a video with 30 frames per second

# problem: it is not smooth enough for our eyes

# solution: we can interpolate frames and increase the frame rate to make video smoother

# interpolation is a process of generating new data points within the range of a discrete set of known data points

# extrapolation is a process of generating new data points outside the range of a discrete set of known data points

import cv2


def get_optical_flow(cap):
    # get frames
    while True:
        successfully_read, frame = cap.read()
        if not successfully_read:
            break
        
       
        # make frames gray
        # get Lucas Kanade feature paramters for frames
        # calculate optical flow


def main():
    # TODO: read video
    file_name = "example.mp4"
    cap = cv2.VideoCapture(file_name)
    optical_flow = get_optical_flow(cap)

    


    # average frames
    # write video
    pass




if __name__ == '__main__':
    main()
    