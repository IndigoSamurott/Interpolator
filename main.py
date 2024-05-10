import warnings;warnings.filterwarnings("ignore")

import cv2
import numpy as np
from multiprocessing import Pool
import subprocess
import pickle, requests
from tqdm import tqdm, trange

from google_auth_oauthlib.flow import InstalledAppFlow
import googleapiclient.discovery


sobel = { 'x': np.array([[-1,  0,  1], 
                         [-2,  0,  2], 
                         [-1,  0,  1]
                       ]),
          'y': np.array([[-1,  2,  -1], 
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
        else:
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
        if self.size() < self.__length: #static size constraint
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

    def push(self, item):
        super().push(item)
        i = self.size() - 1
        while i > 0 and self.data[i] < self.data[(i - 1) // 2]: #if smaller than its parent
            self.data[i], self.data[(i - 1) // 2] = self.data[(i - 1) // 2], self.data[i] #swap
            i = (i - 1) // 2
        
    def pop(self):
        if self.size() == 0:
            return None

        root = self.data[0] #the data we want
        last_node = super().pop()
        if self.size() > 0:
            self.data[0] = last_node
            self.heapify(0) #rebuild structure

        return root
        
    def peek(self):
        if self.is_empty():
            super().peek()
        else:
          return self.data[0]


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
        theheap.push(i) #will autosort

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


def dot(a, b):
    a = np.array(a)
    b = np.array(b)
    product = np.zeros((a.shape[0], b.shape[1]))

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                product[i][j] += a[i][k] * b[k][j]

    return product


def makegrey(BGR): #cv2 decodes frames to BGR rather than RGB
    return np.array([[0.114*j[0] + 0.587*j[1] + 0.299*j[2] for j in i] for i in
                     tqdm(BGR, desc='Converting to greyscale', leave=False)]) #weighted colours


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
        for y in trange(offset, height-offset, desc='Computing gradients', leave=False)
    ])
    return gradient_image


def adaptive_mean(pixel_responses, winsize): #thresholds corners in local neighbourhoods
    offset = winsize//2
    height, width = pixel_responses.shape
    corners = []

    for x in trange(width, desc='Applying thresholds', leave=False):
        for y in range(height):

            lower_x = max(0, x - offset) #setting edges of neighbourhood within image bounds
            upper_x = min(width - 1, x + offset)
            lower_y = max(0, y - offset)
            upper_y = min(height - 1, y + offset)

            nhood = pixel_responses[lower_y:upper_y+1, lower_x:upper_x+1]
            mean = np.sum(nhood)//winsize**2
            sumsq = sum(x**2 for y in nhood for x in y)
            standard_dev = (sumsq/winsize**2 - mean**2) ** 0.5
            thresh = mean + standard_dev
            if pixel_responses[y, x] > thresh:
                corners.append([(x, y)])

    return corners


def corner_det(grey_frame, threshold_func):
    offset = len(sobel['x']) // 2
    height, width = grey_frame.shape
    dx = gradient(grey_frame, sobel['x'])
    dy = gradient(grey_frame, sobel['y'])
    dx2 = dx ** 2 ; dy2 = dy ** 2 ; dxdy = dx * dy
    potential_matrix = np.zeros_like(grey_frame)

    for x in trange(offset, width - offset, desc='Rating corners', leave=False):
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

    return corners, dx, dy


def lk(prev_img, dx, dy, dt, coords):
    height, width = prev_img.shape
    offset = len(sobel['x'])//2
    u = np.zeros_like(prev_img); v = np.zeros_like(prev_img)

    for i in tqdm(coords, desc='Finding flow', leave=False):
        for (x, y) in i:
            if offset <= y < (height - offset) and offset <= x < (width - offset):
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

def motionify(origin_img, flow): #pixel remap
    height, width = origin_img.shape[0], origin_img.shape[1]
    remapped = np.zeros_like(origin_img)

    for y in trange(height, desc='Interpolating', leave=False):
        for x in range(width):
            
            new_x, new_y = int(round(x + flow[y, x, 0])), int(round(y + flow[y, x, 1]))

            new_x = np.clip(new_x, 0, width - 1) #retain boundaries
            new_y = np.clip(new_y, 0, height - 1)

            remapped[y, x] = origin_img[new_y, new_x]

    return remapped


def interpolate(path, segment):
    vid = cv2.VideoCapture(path)
    vid.set(cv2.CAP_PROP_POS_FRAMES, segment[0]-1) #seek to the first timestamp

    path_parts = path.replace('/','\\').split('\\')
    directory = '' if len(path_parts) == 1 else '\\'.join(path_parts[:-1]) + '\\' #without filename
    filename = path_parts[-1]
    temp_path = f'{directory}temp_{filename}' #for video without audio
    audio_path = f'{directory}AUDIO_{filename}' #temp extracted audio
    output_path = f'{directory}INTERPOLATED_{filename}'

    start_time = segment[0] / vid.get(cv2.CAP_PROP_FPS) #in secs
    duration = (segment[1] - segment[0]) / vid.get(cv2.CAP_PROP_FPS) #in secs
    subprocess.run(f'ffmpeg -y -i "{path}" -ss {start_time} -t {duration} -vn -loglevel fatal "{audio_path}"')
    
    interpolated = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), (vid.get(cv2.CAP_PROP_FPS)*2),
                                (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for frame_counter in trange(segment[0], segment[1]+1, unit='frame'):
        successfully_read, frame = vid.read()
        if not successfully_read: #error check
            print('Failed whilst reading video.'); break
        
        if frame_counter == segment[0]:
            interpolated.write(frame)
            grey_frame = makegrey(frame)
            prev_corners, dx, dy = corner_det(grey_frame, threshold_func=adaptive_mean)
            prev_frame = frame.copy()

        elif segment[0] < frame_counter:
            prev_grey = grey_frame.copy()
            grey_frame = makegrey(frame)
            dt = grey_frame - prev_grey
            optical_flow = lk(prev_grey, dx, dy, dt, prev_corners)
            optical_flow = np.moveaxis(optical_flow, 0, -1)
            interpolated_frame = motionify(prev_frame, optical_flow)
            interpolated.write(interpolated_frame)
            interpolated.write(frame)
            prev_corners, dx, dy = corner_det(grey_frame, threshold_func=adaptive_mean)
            prev_frame = frame.copy()

    vid.release()
    interpolated.release()

    subprocess.run(f'ffmpeg -y -i "{temp_path}" -i "{audio_path}" -c copy "{output_path}" -loglevel fatal')
    subprocess.run(f'del "{temp_path}" "{audio_path}"', shell=True) #append audio to video then delete temp files
    return output_path


def check_history(id):
    try:
        with open('past_videos.pkl', 'rb') as f:
            history = pickle.load(f)
        if id in history:
            return True, history
        else:
            return False, history
    except FileNotFoundError:
        return False, {}
    
def save_result(vid_id, history, drive_link):
    history[vid_id] = drive_link
    with open('past_videos.pkl', 'wb') as f:
        pickle.dump(history, f)


def upload(file_path, drive_service, local_store):
    try:        
        folder_id = '1yHeaxE5etil3-XNS6JyXzH5GYzhDwPw4'

        file_metadata = {
            'name': file_path.replace('/','\\').split('\\')[-1], #filename from path
            'parents': [folder_id]
        }

        with open(file_path, 'rb') as f:
            media = googleapiclient.http.MediaIoBaseUpload(f, mimetype='video/mp4', resumable=True)
            file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        permission = {'type': 'anyone', 'role': 'reader'}
        drive_service.permissions().create(fileId=file['id'], body=permission).execute()
        link = f'https://drive.google.com/file/d/{file["id"]}/view?usp=sharing'


        headers = { #as per smolurl api docs
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {"url": link}

        response = requests.post("https://smolurl.com/api/links", headers=headers, json=data)

        if response.status_code == 201:
            save_result(local_store[0], local_store[1], response.json()['data']['short_url']) #store w/ smol
        else:
            save_result(local_store[0], local_store[1], link) #store w/ drive

        return
    
    except Exception as e: #for any network problem etc
        print(f'\nFailed to upload {file_path}:\n{e}')
        return


def batch_select():
    files = {}; fileframes = {}
    while True:
        path = input('\nEnter a file path: ').replace('"','') #windows applies quotes around path
        if not path:
            if len(files) == 0: continue #no initial input
            else: validcheck.release(); break

        
        validcheck = cv2.VideoCapture(path)
        if not validcheck.isOpened(): #error check
            print("There's no video in that location. Try again.")
            continue

        total_frames = int(validcheck.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = validcheck.get(cv2.CAP_PROP_FPS)

        successfully_read, frame1 = validcheck.read()
        if not successfully_read: #error check
            print("Could not read that video.")
            continue
        
        frame1 = cv2.imencode('.jpg', frame1)
        if not frame1[0]: #error check
            print('Errored whilst parsing that video.')
            continue
        
        frame_size = len(frame1[1].tobytes())

        segment_choice = input('Would you like to interpolate that between specific timestamps? [Y/N]: ')
        if segment_choice.lower() != 'y':
            start_frame = 1
            end_frame = total_frames
        else:
            try:
                timestamp1 = input('Enter first timestamp  (HH:MM:SS[:ms]): ')
                parts = timestamp1.split(':')
                if len(parts) == 3:
                    hh, mm, ss = map(int, parts)
                    ms = 0
                elif len(parts) == 4:
                    parts[-1] = parts[-1].ljust(3, '0') #pad ms with 0s
                    hh, mm, ss, ms = map(int, parts)
                else:
                    print("Invalid timestamp format. Try again.")
                    continue
                start_frame = round((hh * 3600 + mm * 60 + ss) * fps  +  ms * (fps / 1000))
                if start_frame < 0 or start_frame > total_frames:
                    print("That's not within the video duration. Try again.")
                    continue

                timestamp2 = input('Enter second timestamp (HH:MM:SS[:ms]): ')
                parts = timestamp2.split(':')
                if len(parts) == 3:
                    hh, mm, ss = map(int, parts)
                    ms = 0
                elif len(parts) == 4:
                    parts[-1] = parts[-1].ljust(3, '0')
                    hh, mm, ss, ms = map(int, parts)
                else:
                    print("Invalid timestamp format. Try again.")
                    continue
            except: #in case of non-int input
                print("Invalid timestamp format. Try again.")
                continue
            end_frame = round((hh * 3600 + mm * 60 + ss) * fps  +  ms * (fps / 1000))
            if end_frame < 0 or end_frame > total_frames:
                print("That's not within the video duration. Try again.")
                continue
            elif end_frame == start_frame:
                print('The second timestamp must be further along. Try again.')
                continue
            elif end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
        start_frame = start_frame if (start_frame > 0) else 1
        end_frame = end_frame if (end_frame > 1) else 2

        segment_size = (end_frame - start_frame) * frame_size

        validcheck.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        id_frame_1 = np.sum(validcheck.read()[1])
        id_frame_2 = np.sum(validcheck.read()[1]) #if interpolated result is interpolated again, this will differ
        id = (path.replace('/','\\').split('\\')[-1], segment_size, id_frame_1, id_frame_2) 

        done, history = check_history(id)
        if done:
            cont = input("\nYou seem to have requested this before! The previous result was saved at:"
                         f"\n{history[id]}"
                         "\nWould you like to continue anyways? [Y/N]: ")
            if cont.lower() != 'y':
                continue #back to top

        while segment_size in files:
            if fileframes[segment_size][2] == id:
                break #user has just entered same thing twice, proceed to overwrite
            segment_size +=  1 #else if not unique then decrease priority by one

        files[segment_size] = path
        fileframes[segment_size] = (start_frame, end_frame, id)
        print('File added.')

    return files, fileframes, history



def main():
    gflow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', 
                                                      ['https://www.googleapis.com/auth/drive']) #download from gcloud console
    creds = gflow.run_local_server(port=0) #OS autoselects port
    drive_service = googleapiclient.discovery.build('drive', 'v3', credentials=creds)

    files, fileframes, history = batch_select()

    projects = stack(len(files))
    for i in (sort(files))[::-1]:
        projects.push(i)

    pool = Pool()
    while not projects.is_empty():
        print(f'\n(Video {(len(files) - projects.size() + 1)}/{len(files)})') #improves UX
        print('>>>>>>>>>> ' + files[projects.peek()].replace('/','\\').split('\\')[-1] + ' <<<<<<<<<<\n')
        output = interpolate( files[projects.peek()] , fileframes[projects.peek()] )

        local_storage = (fileframes[projects.pop()][2], history)
        pool.apply_async(upload, (output, drive_service, local_storage))
                         #submit upload tasks to separate processes without blocking main

    pool.close()
    pool.join()

    print(f'\n\nAll done! Find your new file{'s' if len(files) > 1 else ''} in '
          f'{'their' if len(files) > 1 else 'its'} original director{'ies' if len(files) > 1 else 'y'},'
          f'or download {'them'  if len(files) > 1 else 'it'} at one of:\n'
            '- https://drive.google.com/drive/folders/1yHeaxE5etil3-XNS6JyXzH5GYzhDwPw4?usp=sharing\n'
            '- https://smolurl.com/N5rF9a\n')



if __name__ == "__main__": #ensuring main process
    main()