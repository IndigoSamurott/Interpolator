import cv2
import numpy as np
from tqdm import tqdm, trange
np.seterr(all='ignore')


# PRIORITY 0: CORNERS (INCREASE THRESH?)
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
    boolimg = np.zeros_like(pixel_responses)
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
            thresh = mean + 2.85*standard_dev
            if pixel_responses[y, x] > thresh:
                boolimg[y, x] = 1
                corners.append([(x, y)])

    return boolimg, corners


def corner_det(prev_grey_frame, threshold_func):
    offset = len(sobel['x']) // 2
    height, width = prev_grey_frame.shape
    print("Computing x gradients:")
    dx = gradient(prev_grey_frame, sobel['x'])
    print("Computing y gradients:")
    dy = gradient(prev_grey_frame, sobel['y'])

    dx2 = dx ** 2 ; dy2 = dy ** 2 ; dxdy = dx * dy
    potential_matrix = np.zeros_like(prev_grey_frame)
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

    thresholded_img, corners = threshold_func(potential_matrix, len(sobel['x']))

    # visualise corners for testing  
    marked_frame = cv2.cvtColor(prev_grey_frame, cv2.COLOR_GRAY2BGR)  # convert grayscale frame to BGR, didn't work otherwise
    for corner in corners:
        x, y = corner[0]
        cv2.circle(marked_frame, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite(f"corner_img.jpg", marked_frame)

    breakpoint()
    return corners, dx, dy


def lk(previmg, newimg, dx, dy, dt):
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

    
    flow = np.array([u, v])*15
    return flow


def interpolate(vid):
    successfully_read, frame = vid.read()
    prev_grey_frame = None
    if successfully_read:
        prev_grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ##########################################

    frame_counter = 1
    prev_corner_points = None

    while True:
        if not successfully_read:
            break

        successfully_read, frame = vid.read()
        if not successfully_read:
            break

        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ###############################################
        dt = grey_frame - prev_grey_frame

        if frame_counter == 20: # TESTING only run on the 20th frame because it's lit up
            prev_corner_points, dx, dy = corner_det(prev_grey_frame, threshold_func=adaptive_mean)
        elif frame_counter == 21:
            new_corner_points, _, _ = corner_det(grey_frame, threshold_func=adaptive_mean)
            flow = lk(prev_grey_frame, grey_frame, dx, dy, dt)

        prev_grey_frame = grey_frame.copy()
        frame_counter += 1


def batch_select():
    global files; files = {}
    while True:
        path = input('\nEnter file path: ').replace('"','') #windows applies quotes around path
        if not path and len(files) != 0:
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








img1 = cv2.imread(r"C:\Users\deept_oeog1pt\Downloads\eval-color-allframes\eval-data\Army\frame11.png")
img2 = cv2.imread(r"C:\Users\deept_oeog1pt\Downloads\eval-color-allframes\eval-data\Army\frame12.png")



def motionify(origin_img, flow):
    height, width = origin_img.shape[0], origin_img.shape[1]
    remapped = np.zeros_like(origin_img)

    for y in range(height):
        for x in range(width):
            
            new_x, new_y = int(round(x + flow[y, x, 0])), int(round(y + flow[y, x, 1]))

            new_x = np.clip(new_x, 0, width - 1) #retain boundaries
            new_y = np.clip(new_y, 0, height - 1)

            remapped[y, x] = origin_img[new_y, new_x]

    return remapped


def tempgen(frame_1, frame_2):
    grey1 = makegrey(frame_1)
    grey2 = makegrey(frame_2)
    dx = gradient(grey1,sobel['x'])
    dy = gradient(grey1,sobel['y'])
    dt = grey2-grey1

    optical_flow = (lk(grey1, grey2, dx, dy, dt)).astype(np.float32)
    optical_flow = np.moveaxis(optical_flow, 0, -1)
    height, width = optical_flow.shape[0], optical_flow.shape[1]

    optical_flow *= -1
    interpolated_frame = motionify(frame_1, optical_flow)
    cv2.imwrite('frame12.jpg',interpolated_frame)

tempgen(img1, img2)





def main():
    projects = batch_select()
    while not projects.is_empty():
        interpolate(cv2.VideoCapture(files[projects.pop()]))

main()