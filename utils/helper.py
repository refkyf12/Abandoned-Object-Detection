import cv2, pytz, json, ftplib, time, os

from datetime import datetime

import paho.mqtt.client as mqtt

def date_time():
    nows = datetime.now()
    localtz = pytz.timezone('Asia/Jakarta')
    dates = localtz.localize(nows).strftime("%d%m%Y%H%M%S")
    return str(dates)

def getOutputsNames(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_detection(img, x, y, x_plus_w, y_plus_h, label, confidence, color):
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h + 15), color, 1)    
    draw_border(img, (x,y), (x_plus_w,y_plus_h + 15), color, 5, 0, 20)
    cv2.putText(img, str(round(confidence, 2)), (x+10,y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(img, str(label), (x+100,y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def draw_box(img, x, y, x_plus_w, y_plus_h, label, confidence, color):
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    
    conf = str(round(confidence, 2))
    cv2.putText(img, conf, (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

# Darw a rectangle surrounding the object and its class name
def draw_pred(classes, colors, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label  = str(classes[class_id])
    # color = colors[class_id]
    
    color_ = (0,255,0)
    
    # if (class_id == 1 or class_id == 2 or class_id == 4 or class_id == 5):
    #     color_ = (0,0,255)
    # else:
    #     color_ = (0,255,0)
    
    draw_detection(img, x, y, x_plus_w, y_plus_h, label, confidence, color_)
        
def date_time():
    nows = datetime.now()
    localtz = pytz.timezone('Asia/Jakarta')
    dates = localtz.localize(nows).strftime("%d-%m-%Y %H:%M:%S")
    return dates

def send_mqtt(broker, broker_port, topic_pub, username, password, msg):   

    client = mqtt.Client()   
    client.username_pw_set(username, password)
    code = client.connect(broker, broker_port, 30)
    
    if code == 0 :

        print ("MQTT Connected successfully ") 

        client.loop_start()

        while True:
            client.publish(topic_pub, json.dumps(msg), qos=0, retain=False)
            print("MQTT message sent")
            client.disconnect()
            break
    else:
        print('MQTT Bad connection. Code: ', code)
        client.disconnect()
        
    client.disconnect()
    print("MQTT Disconnected")

def send_ftp(host, port, username, password, folder, image_path, image_file):
    ftp = ftplib.FTP()
    
    try:
        ftp.connect(host, port)
        ftp.login(username, password)
        
        ftp.cwd(folder)

        with open(image_path + image_file, 'rb') as file:
            # Set the transfer mode to binary
            ftp.sendcmd('TYPE I')
            # Upload the file to the server
            ftp.storbinary(f'STOR {image_file}', file)
        
        ftp.quit()
        os.remove(image_path + image_file)
        return True  # Return success
    except ftplib.all_errors as e:
        print(f"FTP error occurred: {e}")
        return False  # Return failure
 
def retry_send_ftp(host, port, username, password, folder, image_path, image_file):

    max_attempts=3 
    delay=5

    attempts = 0
    success = False
    
    while attempts < max_attempts and not success:
        success = send_ftp(host, port, username, password, folder, image_path, image_file)
        
        if not success:
            attempts += 1
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    if success:
        print("FTP operation succeeded.")
    else:
        print("FTP operation failed after maximum attempts.")