import cv2
import numpy as np
import time
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Manager

from utils.helper import getOutputsNames, draw_pred

# Inisialisasi model YOLOv3

# Inisialisasi video
 # Ganti dengan path video Anda

# Inisialisasi variabel untuk menyimpan status objek
objects = {}  # Dictionary untuk menyimpan informasi objek
abandoned_threshold = 100  # Jarak (dalam piksel) sebelum objek dianggap terbengkalai

def classify_frame(net, inputQueue, outputQueue):
    
    while True:
        
        # time.sleep(delay)
        if not inputQueue.empty():
        
            start_time_process = time.time()
            
            image = inputQueue.get()
            blob = cv2.dnn.blobFromImage(image, 0.00392, (412, 412), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            detections = net.forward(getOutputsNames(net))
        
            end_time_process = time.time()
            
            time_process = end_time_process - start_time_process
            var_manager['time_process'] = "{:.2f}".format(time_process)
            
            outputQueue.put(detections)


def detect():
    global detections
    cap = cv2.VideoCapture("video1.avi") 

    with open('coco.names', "r") as f:
        classes = f.read().strip().split('\n')\

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    while True:
        hasFrame, image = cap.read()
        
        if image is None:
            image = np.zeros((1020,480,3), dtype=np.uint8)
            continue
        
        Width = image.shape[1]
        Height = image.shape[0]

        if inputQueue.full():
            continue
        else:
            inputQueue.put(image)

        if not outputQueue.empty():
            detections = outputQueue.get()
        
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []

        if detections is not None:
            
            for out in detections:    
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.8:
                        
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x,y,w,h])
                        objects[class_id] = {"last_position": (x, y)}

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
               
            for i in range(len(boxes)):
                if i in indices:
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]   
                    draw_pred(classes, colors, image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

            if len(indices) > 0:
                for i in range(len(boxes)):
                    if i in indices:
                        class_id = class_ids[i]
                        box = boxes[i]
                        x, y, w, h = box
                        
                        # Mengukur jarak tempuh
                        last_x, last_y = objects[class_id]["last_position"]
                        distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)

                        if distance > abandoned_threshold:
                            cv2.putText(image, "Abandoned", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                        # Perbarui posisi terakhir
                        objects[class_id]["last_position"] = (x, y)              
            
            frameQueue.put(image)

def output():
    while True:
        # time.sleep(delay)
        if frameQueue.empty() !=True:
            image = frameQueue.get()
            
            # currTime = time.time()
            # prevTime = currTime
            # fps = 1 / (currTime - prevTime)
            fps = 0
            cv2.putText(image, str(round(fps,2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1) #FPS Value
                
            var_manager['image'] = image
            
            print('')
            print("PROCESS DETEKSI : ", var_manager['time_process'])
            print("QUEUE Input ", inputQueue.qsize())
            # print("QUEUE Frame ", frameQueue.qsize())
            # print("FPS:", fps)

            cv2.imshow("Frame", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":

    
    manager = Manager()
    
    # LOAD MODEL AND CONFIG
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

    # net = cv2.dnn.readNetFromDarknet(config['model'], config['config'])
    
    var_manager = manager.dict({'image' : None, 'time_process': 0,})
    
    inputQueue = Queue(maxsize=10)
    outputQueue = Queue(maxsize=10)
    frameQueue = Queue(maxsize=10)
    detections = None

    print("[INFO] starting process...")
    p_detect = Process(target=detect)
    p_classify_frame = Process(target=classify_frame, args=(net, inputQueue, outputQueue,))
    p_output = Process(target=output)
    
    p_classify_frame.start()
    p_detect.start()
    p_output.start()
    
    p_detect.join()
    p_classify_frame.join()
    p_output.join()
