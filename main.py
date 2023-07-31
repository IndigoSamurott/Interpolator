# Interpolating frames for a video

# we have a video with 30 frames per second

# problem: it is not smooth enough for our eyes

# solution: we can interpolate frames and increase the frame rate to make video smoother

# interpolation is a process of generating new data points within the range of a discrete set of known data points

# extrapolation is a process of generating new data points outside the range of a discrete set of known data points

import cv2


def get_optical_flow(cap):

    #get data for VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #create VideoWriter object
    out = cv2.VideoWriter('grey.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (fWidth, fHeight), isColor=False)


    # get frames
    while True:
        successfully_read, frame = cap.read()
        if not successfully_read:
            break

        #make grey
        greyVer = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #tests
        print(greyVer)
        print(greyVer[100][200])
        out.write(greyVer)
       
    out.release()
    cap.release()
        # graph the frames
        
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
    