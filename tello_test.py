from pkgutil import get_data
from re import X
from time import sleep
import tellopy
from tellopy._internal.utils import *
from tellopy._internal.protocol import *
import stream_tello
import threading
import pygame
import cv2
import numpy as np
import time
import av
import sys
import traceback

prev_flight_data = None
font = None
x = 5


def frameRescale(frame, scale=0.92):

    withh = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimesion = (withh,height)

    return cv2.resize(frame, dimesion, interpolation=cv2.INTER_AREA)


def detectFace(gray_image, face_cascade):
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def control_drone(drone, face_rects, frame_size):
    frame_width, frame_height = frame_size
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    threshold_x = int(frame_width * 0.1)

    if len(face_rects) > 0:
        x, y, w, h = face_rects[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        error_x = face_center_x - frame_center_x
        error_y = face_center_y - frame_center_y

        move_x = int(abs(error_x) / frame_width * 100)

        print("error_x:", error_x)  # Agregamos esto para depurar
        print("move_x:", move_x)  # Agregamos esto para depurar

        if abs(error_x) > threshold_x:
            if error_x > 0:
                print("Activando clockwise")  # Agregamos esto para depurar
                drone.clockwise(move_x)
            else:
                print("Activando counter_clockwise")  # Agregamos esto para depurar
                drone.counter_clockwise(move_x)
        else:
            drone.clockwise(0)
            drone.counter_clockwise(0)
    else:
        drone.clockwise(0)
        drone.counter_clockwise(0)





def Stream(drone):

    cont = 0
    path = '/home/p1t/tello-ai/tello-yollov7-ia/images/'
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    try:
        container = av.open(drone.get_video_stream())
        # skip first 300 frames
        frame_skip = 300
        while True:

            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue

                start_time = time.time()
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image_small = cv2.resize(gray_image, (0, 0), fx=0.5, fy=0.5)

                faces = detectFace(gray_image_small, face_cascade)


                # image_resize = frameRescale(image)
                #cv2.imshow('Original', image)

                for (x, y, w, h) in faces:
                    x, y, w, h = x * 2, y * 2, w * 2, h * 2
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


                #if cont%10 == 0:
                #    cv2.imwrite(path + 'IMG_%04d.jpg' % cont, image)            
                #cont += 1

                #control_drone(drone, faces, image.shape[:2])

                cv2.imshow('Original',image)
                #cv2.imshow('Canny', cv2.Canny(image_resize, 100, 200))
                cv2.waitKey(1)

                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
                    
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)

    cv2.destroyAllWindows()




def handler(event, sender, data, **args):
    global prev_flight_data
    global x
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)
        #print(data.battery_percentage)
        x = data.battery_percentage
        #print(x)
        return x
    elif event is drone.EVENT_LOG_DATA:
        log_data = data
        #print(log_data)



def test(drone):
    try:
        sleep(10)
        drone.takeoff()
        sleep(10)
        drone.land()
    except Exception as ex:
        print(ex)
        show_exception(ex)
    print('end.')

if __name__ == '__main__':
    drone = tellopy.Tello()
    drone.connect()
    drone.wait_for_connection(20.0)
    drone.start_video()
    drone.subscribe(drone.EVENT_LOG_DATA, handler)
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    video = threading.Thread(target=stream_tello.Stream, args=(drone,))
    commandos = threading.Thread(target=test, args=(drone,))

    print(x)
    video.start()
    commandos.start()
    sleep(2)
    print(x)
    # drone.quit()
    # print('END')