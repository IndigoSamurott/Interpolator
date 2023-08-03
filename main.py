# Interpolating frames for a video

# we have a video with 30 frames per second

# problem: it is not smooth enough for our eyes

# solution: we can interpolate frames and increase the frame rate to make video smoother

# interpolation is a process of generating new data points within the range of a discrete set of known data points

# extrapolation is a process of generating new data points outside the range of a discrete set of known data points

import cv2

def get_optical_flow(cap):

    # get first frame
    successfully_read, frame = cap.read()
    prev_grey_frame = None

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
    main()
    