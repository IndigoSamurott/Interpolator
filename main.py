import warnings;warnings.filterwarnings("ignore")

import cv2
import numpy as np
from tqdm import tqdm, trange
import time

# proper interpolation output
# priority 1: passing video segment times
# priority 2: upload interpolated vid to mediafire
# priority 3: at frames in regular odd interval, hash frame data, hash table, mediafire link


sobel = { 'x': np.array([[-1,  0,  1], 
                         [-2,  0,  2], 
                         [-1,  0,  1]
                       ]),
          'y': np.array([[-1, -2, -1], 
                         [ 0,  0,  0], 
                         [ 1,  2,  1]
                       ]) }


class data_type:
    def __init__(self):
        self.data = []

    def push(self, item):
        self.data.append(item)

    def pop(self):
        if self.is_empty():
            return None
        return self.data.pop()

    def peek(self):
        if self.is_empty():
            return None
        else:
          return self.data[-1]

    def size(self):
        return len(self.data)

    def is_empty(self):
        return self.size() == 0
        

class stack(data_type):
    def __init__(self, length):
        super().__init__()
        self.__length = length

    def push(self, item):
        if self.size() <= self.__length: #static size constraint
            super().push(item)
        else:
            raise Exception("Stack overflow")

    def pop(self):
        if self.is_empty():
            raise Exception("Stack underflow")
        else:
            return super().pop()


class heap(data_type):
    def heapify(self, currentindex):
        smallest = currentindex
        left_child = 2*currentindex + 1
        right_child = 2*currentindex + 2

        if left_child < self.size() and self.data[left_child] < self.data[smallest]:
            smallest = left_child

        if right_child < self.size() and self.data[right_child] < self.data[smallest]:
            smallest = right_child

        if smallest != currentindex:
            self.data[currentindex], self.data[smallest] = self.data[smallest], self.data[currentindex] #swap
            self.heapify(smallest)
        
    def pop(self):
        if self.size() == 0:
            return None

        root = self.data[0]
        last_node = super().pop()
        if self.size() > 0:
            self.data[0] = last_node
            self.heapify(0)

        return root

    def push(self, item):
        super().push(item)
        i = self.size() - 1
        while i > 0 and self.data[i] < self.data[(i - 1) // 2]:
            self.data[i], self.data[(i - 1) // 2] = self.data[(i - 1) // 2], self.data[i]
            i = (i - 1) // 2
        
    def peek(self):
        if self.is_empty():
            super().peek()
        else:
          return self.data[0]


def insertion_sort(inp): #O(n²), best for small lists because little overhead (no recursion)
    for i in range(1, len(inp)):
        temp = inp[i]
        j = i-1
        while j >= 0 and inp[j] > temp:
            inp[j + 1] = inp[j]
            j -= 1
        inp[j + 1] = temp
    return inp

def heapsort(inp): #O(nlogn) for unusual cases where quicksort goes wrong
    out = []
    theheap = heap()
    for i in inp:
        theheap.push(i)

    while not theheap.is_empty():
        out.append(theheap.pop())
    
    return out

def introsort(inp, maxdepth): #introspective
    if len(inp) < 16:
        insertion_sort(inp)
    elif maxdepth == 0:
        heapsort(inp)
    else:
        #quicksort logic adapted for depth constraint, avg O(nlogn) worst O(n2)
        pivot = inp[0]
        left = []
        equal = []
        right = []

        for i in range(len(inp)):
            if inp[i] == pivot:
                equal.append(inp[i])
            elif inp[i] < pivot:
                left.append(inp[i])
            elif inp[i] > pivot:
                right.append(inp[i])

        introsort(left, maxdepth - 1)
        introsort(right, maxdepth - 1)

        inp[:] = left + equal + right

def sort(inp):
    if isinstance(inp, dict):
        inp = list(inp.keys())
    maxdepth = 2 * np.log2(len(inp))
    introsort(inp, maxdepth)
    return inp


def makegrey(BGR): #cv2 decodes frames to BGR rather than RGB
    return np.array([[0.114*j[0] + 0.587*j[1] + 0.299*j[2] for j in i] for i in tqdm(BGR, desc='Converting to greyscale')]) #weighted colours

def dot(a, b):
    a = np.array(a)
    b = np.array(b)
    product = np.zeros((a.shape[0],b.shape[1]))

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                product[i][j] += a[i][k] * b[k][j]

    return product


def convolve(imgwin, kernel):
    kern_y, kern_x = kernel.shape
    filterpix = 0
    for x in range(kern_x):
        for y in range(kern_y):
            filterpix += kernel[y][x] * imgwin[y][x]
    return filterpix

def gradient(grey_frame, kernel):
    offset = len(kernel) // 2
    height, width = grey_frame.shape
    gradient_image = np.zeros_like(grey_frame)

    gradient_image[offset:height-offset, offset:width-offset] = \
    np.array([
        [convolve(grey_frame[y-offset:y+offset+1, x-offset:x+offset+1], kernel) 
        for x in range(offset, width-offset)] 
        for y in trange(offset, height-offset)
    ])
    return gradient_image


def adaptive_mean(pixel_responses, winsize): #thresholds corners in local neighbourhoods
    offset = winsize//2
    height, width = pixel_responses.shape
    corners = []

    for x in trange(width, desc='Applying thresholds'):
        for y in range(height):

            lower_x = max(0, x - offset) #setting edges of neighbourhood within image bounds
            upper_x = min(width - 1, x + offset)
            lower_y = max(0, y - offset)
            upper_y = min(height - 1, y + offset)

            nhood = pixel_responses[lower_y:upper_y+1, lower_x:upper_x+1]
            mean = np.sum(nhood)//winsize**2
            sumsq = sum(x**2 for y in nhood for x in y)
            standard_dev = (sumsq/winsize**2 - mean**2) ** 0.5
            thresh = mean + 1*standard_dev #set to like 2.25 for testing as well as 1 for tuneability
            if pixel_responses[y, x] > thresh:
                corners.append([(x, y)])

    return corners


def corner_det(grey_frame, threshold_func):
    offset = len(sobel['x']) // 2
    height, width = grey_frame.shape
    print("Computing x gradients:")
    dx = gradient(grey_frame, sobel['x'])
    print("Computing y gradients:")
    dy = gradient(grey_frame, sobel['y'])

    dx2 = dx ** 2 ; dy2 = dy ** 2 ; dxdy = dx * dy
    potential_matrix = np.zeros_like(grey_frame)
    for x in trange(offset, width - offset, desc='Rating corners'):
        for y in range(offset, height - offset):
            # sum squared derivatives in a window
            window_dx2 = dx2[y - offset: y + offset + 1, x - offset: x + offset + 1]
            window_dy2 = dy2[y - offset: y + offset + 1, x - offset: x + offset + 1]
            window_dxdy = dxdy[y - offset: y + offset + 1, x - offset: x + offset + 1]
            dx2_sum, dy2_sum, dxdy_sum = np.sum(window_dx2), np.sum(window_dy2), np.sum(window_dxdy)

            """ harris_matrix = np.array([[dx2_sum, dxdy_sum], 
                                          [dxdy_sum, dy2_sum]]) """
            
            harris_trace = dx2_sum + dy2_sum #sum of diagonal
            harris_determinant = dx2_sum * dy2_sum - dxdy_sum*dxdy_sum #diagonal product - diagonal product
            harris_response = harris_determinant - 0.04 * harris_trace**2 #the function that rates corners
            potential_matrix[y - offset, x - offset] = harris_response

    corners = threshold_func(potential_matrix, len(sobel['x']))

    # visualise corners for testing  
    marked_frame = cv2.cvtColor(grey_frame.astype(np.float32), cv2.COLOR_GRAY2BGR)  # convert grayscale frame to BGR, didn't work otherwise
    for corner in corners:
        x, y = corner[0]
        cv2.circle(marked_frame, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite("corner_img.jpg", marked_frame)
    return corners, dx, dy


def lk_nocorner(previmg, dx, dy, dt): #use for testing?
    height, width = previmg.shape
    offset = len(sobel['x'])//2
    u = v = np.zeros_like(previmg)

    for y in trange(offset, height - offset, desc='Finding flow'):
        for x in range(offset, width - offset):
            nhood_dx = dx[y - offset: y + offset + 1, x - offset: x + offset + 1]
            nhood_dy = dy[y - offset: y + offset + 1, x - offset: x + offset + 1]
            nhood_dt = dt[y - offset: y + offset + 1, x - offset: x + offset + 1]
            S = np.array([nhood_dx, nhood_dy]).reshape(-1, 2)
            S_T = [[row[i] for row in S] for i in range(len(S[0]))] #transposed; flipped along diagonal
            S_ST = dot(S_T, S)

            inv_SST = np.linalg.pinv(S_ST) #pseudo-inverse works on ill-conditioned matrices

            u[y, x], v[y, x] = dot(dot(inv_SST, S_T), nhood_dt.reshape(9,-1))

    
    flow = np.array([u, v])*10
    return flow

def lk(previmg, dx, dy, dt, coords):
    height, width = previmg.shape
    offset = len(sobel['x'])//2
    u = v = np.zeros((height, width))

    for i in tqdm(coords, desc='Finding flow'):
        for (y, x) in i:
            if offset <= y < height - offset and offset <= x < width - offset:
                nhood_dx = dx[y - offset: y + offset + 1, x - offset: x + offset + 1]
                nhood_dy = dy[y - offset: y + offset + 1, x - offset: x + offset + 1]
                nhood_dt = dt[y - offset: y + offset + 1, x - offset: x + offset + 1]
                S = np.array([nhood_dx, nhood_dy]).reshape(-1, 2)
                S_T = [[row[i] for row in S] for i in range(len(S[0]))] #transposed; flipped along diagonal
                S_ST = dot(S_T, S)

                inv_SST = np.linalg.pinv(S_ST) #pseudo-inverse works on ill-conditioned matrices

                u[y, x], v[y, x] = dot(dot(inv_SST, S_T), nhood_dt.reshape(9,-1))

                # populate the optical flow in a small area around the coordinate with the same flow
                for i in range(-offset, offset+1):
                    for j in range(-offset, offset+1):
                        if 0 <= y+i < height and 0 <= x+j < width:
                            u[y+i, x+j] = u[y, x]
                            v[y+i, x+j] = v[y, x]

    flow = np.array([u, v])*10
    return flow

def motionify(origin_img, flow): #nearest-neighbour pixel remap
    height, width = origin_img.shape[0], origin_img.shape[1]
    remapped = np.zeros_like(origin_img)

    for y in trange(height, desc='Interpolating'):
        for x in range(width):
            
            new_x, new_y = int(round(x + flow[y, x, 0])), int(round(y + flow[y, x, 1]))

            new_x = np.clip(new_x, 0, width - 1) #retain boundaries
            new_y = np.clip(new_y, 0, height - 1)

            remapped[y, x] = origin_img[new_y, new_x]

    return remapped


def interpolate(vid, segment):
    t = time.perf_counter()
    
    frame_counter = 1
    interpolated = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), (vid.get(cv2.CAP_PROP_FPS)*2), (1920,1080)) 

    while frame_counter <= segment[1]:

        successfully_read, frame = vid.read()
        if not successfully_read: #error check
            print('Failed whilst reading video.'); break
        
        if frame_counter == segment[0]:
            interpolated.write(frame)
            grey_frame = makegrey(frame)
            prev_corner_points, dx, dy = corner_det(grey_frame, threshold_func=adaptive_mean)
            prev_frame = frame.copy()

        elif segment[0] < frame_counter:
            prev_grey = grey_frame.copy()
            grey_frame = makegrey(frame)
            dt = grey_frame - prev_grey
            #optical_flow = lk_nocorner(prev_grey, dx, dy, dt)
            optical_flow = lk(prev_grey, dx, dy, dt, prev_corner_points)
            optical_flow = np.moveaxis(optical_flow, 0, -1)
            interpolated_frame = motionify(prev_frame, optical_flow)
            interpolated.write(interpolated_frame)
            interpolated.write(frame)
            prev_corner_points, dx, dy = corner_det(grey_frame, threshold_func=adaptive_mean)
            prev_frame = frame.copy()

        frame_counter += 1
    print(time.perf_counter()-t)
    interpolated.release()
    pass


#img1 = cv2.imread(r"C:\Users\deept_oeog1pt\Downloads\eval-color-allframes\eval-data\Army\frame11.png")
#img2 = cv2.imread(r"C:\Users\deept_oeog1pt\Downloads\eval-color-allframes\eval-data\Army\frame12.png")
img1 = cv2.imread('frame1.jpg')
img2 = cv2.imread('frame3.jpg')

""" def tempgen(frame_1, frame_2):
    grey1 = makegrey(frame_1)
    grey2 = makegrey(frame_2)
    corners, dx, dy = corner_det(grey1, adaptive_mean)
    dt = grey2-grey1
    optical_flow = lk(grey1, dx, dy, dt, corners)
    optical_flow = np.moveaxis(optical_flow, 0, -1)
    interpolated_frame = motionify(frame_1, optical_flow)
    cv2.imwrite('framexgen.jpg',interpolated_frame)
tempgen(img1, img2) """


def batch_select():
    global files; files = {}; fileframes = {}
    while True:
        path = input('\nEnter a file path: ').replace('"','') #windows applies quotes around path
        if not path:
            if len(files) == 0: continue #no initial input
            else: break

        validcheck = cv2.VideoCapture(path)
        if not validcheck.isOpened(): #error check
            print("There's no video in that location. Try again.")
            validcheck.release(); continue

        total_frames = int(validcheck.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = validcheck.get(cv2.CAP_PROP_FPS)

        successfully_read, frame1 = validcheck.read()
        if not successfully_read: #error check
            print("Could not read that video.")
            validcheck.release(); continue
        
        frame1 = cv2.imencode('.jpg', frame1)
        if not frame1[0]: #error check
            print('Errored whilst parsing that video.')
            validcheck.release(); continue
        validcheck.release()
        
        frame_size = len(frame1[1].tobytes())

        segment_choice = input('Would you like to interpolate that between specific timestamps? [Y/N]: ')
        if segment_choice.lower() != 'y':
            frame1 = 1
            frame2 = total_frames
        else:
            try:
                timestamp1 = input('Enter first timestamp  (HH:MM:SS[:ms]): ')
                parts = timestamp1.split(':')
                if len(parts) == 3:
                    hh, mm, ss = map(int, parts)
                    ms = 0
                elif len(parts) == 4:
                    hh, mm, ss, ms = map(int, parts)
                else:
                    print("Invalid timestamp format. Try again.")
                    continue
                frame1 = round((hh * 3600 + mm * 60 + ss) * fps  +  ms * (fps / 1000))
                if frame1 < 0 or frame1 > total_frames:
                    print("That's not within the video duration. Try again.")
                    continue

                timestamp2 = input('Enter second timestamp (HH:MM:SS[:ms]): ')
                parts = timestamp2.split(':')
                if len(parts) == 3:
                    hh, mm, ss = map(int, parts)
                    ms = 0
                elif len(parts) == 4:
                    hh, mm, ss, ms = map(int, parts)
                else:
                    print("Invalid timestamp format. Try again.")
                    continue
            except: #in case of non-int input
                print("Invalid timestamp format. Try again.")
                continue
        
            frame2 = round((hh * 3600 + mm * 60 + ss) * fps  +  ms * (fps / 1000))
            if frame2 < 0 or frame2 > total_frames:
                print("That's not within the video duration. Try again.")
                continue
            elif frame2 == frame1:
                print('The second timestamp must be further along. Try again.')
                continue
            elif frame2 < frame1:
                frame1, frame2 = frame2, frame1
        frame1 = frame1 if (frame1 > 0) else 1
        frame2 = frame2 if (frame2 > 1) else 2

        segment_size = (frame2 - frame1) * frame_size

        while True: #if not unique then decrease priority by one
            try:
                x = files[segment_size]
                segment_size +=1
            except:
                break

        files[segment_size] = path
        fileframes[segment_size] = [frame1, frame2]

    nextvids = stack(len(files))

    for i in (sort(files))[::-1]:
        nextvids.push(i)
    return nextvids, fileframes



def main():
    projects, fileframes = batch_select()
    while not projects.is_empty():
        print(f'\n(Video {(len(files) - projects.size() + 1)}/{len(files)})')
        print('>>>>>>>>>> ' + files[projects.peek()].split('\\')[-1].split('/')[-1] + ' <<<<<<<<<<\n')
        interpolate(cv2.VideoCapture( files[projects.peek()] ), fileframes[projects.pop()])

main()