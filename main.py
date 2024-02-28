import cv2
import numpy as np
from scipy import signal
import time
# np.set_printoptions(threshold = np.inf)

# priority 0: oop concern
# priority 1: working threshold
# priority 2: convolve dot product, convolve makegrey, signal convolve
# priority 3: working interpolation
# priority 4: passing video segment times
# priority 5: upload interpolated vid to mediafire
# priority 6: at frames in regular odd interval, hash frame data, hash table, mediafire link


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
            return None
        else:
          return self.data[0]

#############SORTING#############
def insertion_sort(inp):
    for i in range(1, len(inp)):
        temp = inp[i]
        j = i-1
        while j >= 0 and inp[j] > temp:
            inp[j + 1] = inp[j]
            j -= 1
        inp[j + 1] = temp
    return inp

def heapsort(inp):
    out = []
    theheap = heap()
    for i in inp:
        theheap.push(i)

    while not theheap.is_empty():
        out.append(theheap.pop())
    
    return out


def introsort(inp, maxdepth):
    if len(inp) < 16:
        insertion_sort(inp)
    elif maxdepth == 0:
        heapsort(inp)
    else:
        #quicksort logic adapted for depth constraint
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
    return np.array([[0.114*j[0] + 0.587*j[1] + 0.299*j[2] for j in i] for i in BGR]) #weighted colours
    # weights = np.array([0.114, 0.587, 0.299])
    # return np.dot(BGR, weights) SO CONVOLVE?


def convolve(imgwin, kernel):
    kern_y, kern_x = kernel.shape
    filterpix = 0
    factor = kern_y * kern_x
    for x in range(kern_x):
        for y in range(kern_y):
            filterpix += kernel[y][x] * imgwin[y][x]
    return int((filterpix / factor)) #normalise

""" def convolve(imgwin, kernel, x=0, y=0):
    kern_y, kern_x = kernel.shape
    filterpix = 0 #initialise
    if x < kern_x and y < kern_y:
        filterpix = kernel[y][x] * imgwin[y][x]
        return filterpix + convolve(imgwin, kernel, x + 1, y) if x+1 < kern_x else convolve(imgwin, kernel, 0, y + 1)
    else:
        return 0
 """

def gradient(grey_frame, kernel):
    offset = len(kernel) // 2
    height, width = grey_frame.shape
    gradient_image = np.zeros((height, width))
    gradient_image[offset:height-offset, offset:width-offset] = \
    np.array([
        [convolve(grey_frame[y-offset:y+offset+1, x-offset:x+offset+1], kernel) 
        for x in range(offset, width-offset)] 
        for y in range(offset, height-offset)
    ])
    return gradient_image

""" def gradient(grey_frame, kernel):
    offset = len(kernel) // 2
    height, width = grey_frame.shape
    gradient_image = np.zeros((height, width)) #initialise
    # slide convolution across image, normalising it in range 0-255
    gradient_image[offset:height-offset , offset:width-offset] = \
    np.array([
        [ int((convolve(grey_frame[y-offset:y+offset+1, x-offset:x+offset+1], kernel) / (kernel.shape[0] * kernel.shape[1])) * 255) 
        for x in range(offset, width-offset) ] 
        for y in range(offset, height-offset)
    ])
    return gradient_image """


def adaptive_mean(pixel_responses, winsize,c): #thresholds corners in local neighbourhoods
    offset = winsize//2
    height, width = pixel_responses.shape
    boolimg = np.zeros_like(pixel_responses)
    corners = []

    for x in range(width):
        for y in range(height):

            lower_x = max(0, x - offset) #setting edges of neighbourhood within image bounds
            upper_x = min(width - 1, x + offset)
            lower_y = max(0, y - offset)
            upper_y = min(height - 1, y + offset)

            nhood = pixel_responses[lower_y:upper_y+1, lower_x:upper_x+1]
            mean = sum(sum(nhood))/winsize**2
            sumsq = sum(x**2 for y in nhood for x in y)
            standard_dev = (sumsq/winsize**2 - mean**2) ** 0.5
            thresh = mean + 0.0026# change for sobels; this arbitrary threshold works by examining results; need to make std work
            if pixel_responses[y, x] >= thresh:
                boolimg[y, x] = 1
                corners.append([(x, y)])

    return boolimg, corners


def corner_det(prev_grey_frame, threshold_func):
    # calculate flow derivatives for x and y using the gradient function and Sobel kernels

    offset = len(sobel['x']) // 2
    height, width = prev_grey_frame.shape
    print("calculating gradients...")
    # dx = signal.convolve2d(prev_grey_frame, sobel['x'], 'symm', 'same')
    # dy = signal.convolve2d(prev_grey_frame, sobel['y'], 'symm', 'same')

    dx = gradient(prev_grey_frame, sobel['x'])
    dy = gradient(prev_grey_frame, sobel['y'])
    print("gradients calculated")

    dx2 = dx ** 2 ; dy2 = dy ** 2 ; dxdy = dx * dy
    
    potential_matrix = np.zeros_like(prev_grey_frame)

    for x in range(offset, width - offset):
        print(f"{x}/{width}")
        for y in range(offset, height - offset):
            # sum squared derivatives in a window
            window_dx2 = dx2[y - offset: y + offset + 1, x - offset: x + offset + 1]
            window_dy2 = dy2[y - offset: y + offset + 1, x - offset: x + offset + 1]
            window_dxdy = dxdy[y - offset: y + offset + 1, x - offset: x + offset + 1]
            dx2_sum, dy2_sum, dxdy_sum = np.sum(window_dx2), np.sum(window_dy2), np.sum(window_dxdy)

            harris_matrix = np.array([[dx2_sum, dxdy_sum], 
                                      [dxdy_sum, dy2_sum]])
            harris_trace = dx2_sum + dy2_sum #sum of diagonal
            harris_determinant = dx2_sum * dy2_sum - dxdy_sum*dxdy_sum #diagonal product - diagonal product
            harris_response = harris_determinant - 0.04 * harris_trace**2 #the function that rates corners
            potential_matrix[y - offset, x - offset] = harris_response

    #normalise the matrix of corner likelihoods
    norm = np.linalg.norm(potential_matrix, 1)
    potential_matrix = potential_matrix/norm

    print('\n,threshtest');breakpoint()
    thresholded_img, corners = threshold_func(potential_matrix, len(sobel['x']),'c')

    print(len(corners))

    # visualise corners for testing  
    marked_frame = cv2.cvtColor(prev_grey_frame, cv2.COLOR_GRAY2BGR)  # convert grayscale frame to BGR, didn't work otherwise
    for corner in corners:
        x, y = corner[0]
        cv2.circle(marked_frame, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite(f"corner_img.jpg", marked_frame)

    breakpoint()
    return corners, dx, dy


def lk(prev_corners, new_corners, dx, dy):
    #tau: threshold param to be tested
    tau = 1e-3
    height, width = prev_corners.shape
    offset = len(sobel['x'])//2

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_dx = dx[y - offset : y + offset + 1, x - offset : x + offset + 1]
            flatwindx = [i for j in window_dx for i in j]
            window_dy = dy[y - offset : y + offset + 1,x - offset : x + offset + 1]


            A = [[window_dx[i], window_dy[i]] for i in range(len(window_dx))]
            A_T = [[row[i] for row in A] for i in range(len(A[0]))] #transposed; flipped along diagonal
            A_AT = sum([A_T[i][0]*A[i] for i in range(len(A))]) #dot product of A & A_T


def interpolate(vid):
    frame_counter = 0 #TESTING

    while True:
        # corners tracked from
        prev_corner_points = None  # initialize to None
        if frame_counter == 40:  # TESTING only run on the 40th frame because it's lit up
            prev_corner_points, dx, dy = corner_det(prev_grey_frame, threshold_func=adaptive_mean)
            t=time.perf_counter()
            grey_frame = makegrey(frame)
            print('TIMED',time.perf_counter()-t)
            breakpoint()
        # get next frame and convert to grey
        successfully_read, frame = vid.read()
        if not successfully_read: # if frame read unsuccessful
            break

        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # grey_frame = makegrey(frame)
        if prev_corner_points is not None:
            lk()


        prev_grey_frame = grey_frame.copy()
        frame_counter += 1 
        pass
    pass
pass
""" def interpolate(vid):
    prev_corner_points = None
    while True:
        successfully_read, frame = vid.read()
        if not successfully_read:  # if frame read unsuccessful
            break

        grey_frame = makegrey(frame)

        new_corner_points = corner_det(grey_frame, threshold_func=adaptive_mean)
        

        if prev_corner_points is not None:
            lk(prev_corner_points, new_corner_points, dx, dy)

        prev_grey_frame = grey_frame.copy()
        prev_corner_points, dx, dy = corner_det(prev_grey_frame, threshold_func=adaptive_mean) """



def batch_select():
    global files; files = {}

    while True:
        path = input('Enter file path: ')
        if not path:
            break

        validcheck = cv2.VideoCapture(path)
        if not validcheck.isOpened():
            print('Not a video file. Try again.')
            validcheck.release()
            continue

        total_frames = int(validcheck.get(cv2.CAP_PROP_FRAME_COUNT)) #for timings
        fps = validcheck.get(cv2.CAP_PROP_FPS)
        validcheck.release()
        
        f = open(path, 'rb')
        f.seek(0,2) #put pointer at end of binary file to return pos as filesize 
        files[f.tell()] = path

    nextvids = stack(len(files))

    for i in (sort(files))[::-1]:
        nextvids.push(i)
    return nextvids


def main():
    projects = batch_select()

    while projects.is_empty() != True:
        interpolate(cv2.VideoCapture(files[projects.pop()]))


main()