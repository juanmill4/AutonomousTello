from configparser import Interpolation
import sys
import traceback
import av
import cv2  # for avoidance of pylint error
import numpy
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)



def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def is_x_gesture(hand_landmarks):
    # Puedes modificar estos umbrales según sea necesario
    min_distance = 0.1
    max_distance = 0.25

    # Puntos clave de interés
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calcular distancias entre los puntos clave
    thumb_index_distance = distance(thumb_tip, index_finger_tip)
    index_middle_distance = distance(index_finger_tip, middle_finger_tip)
    middle_ring_distance = distance(middle_finger_tip, ring_finger_tip)
    ring_pinky_distance = distance(ring_finger_tip, pinky_tip)

    # Comprobar si se forma una "X" con los dedos índice y anular
    if min_distance < thumb_index_distance < max_distance and min_distance < ring_pinky_distance < max_distance:
        return True

    return False

def detect_x_gesture(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    x_gesture_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_x_gesture(hand_landmarks):
                x_gesture_detected = True
                break

    return x_gesture_detected


def frameRescale(frame, scale=0.92):

    withh = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimesion = (withh,height)

    return cv2.resize(frame, dimesion, interpolation=cv2.INTER_AREA)


def detectFace(gray_image, face_cascade):
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def control_drone(drone, face_rects, frame_size):

    if len(face_rects) > 0:
        x, y, w, h = face_rects[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        frame_width, frame_height = frame_size

        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        error_x = face_center_x - frame_center_x
        error_y = face_center_y - frame_center_y

        threshold_x = int(frame_width * 0.1)
        threshold_y = int(frame_height * 0.1)

        move_x = int(abs(error_x) / frame_width * 100)
        move_y = int(abs(error_y) / frame_height * 100)

        # Movimiento izquierda/derecha y rotación
        if abs(error_x) > threshold_x:
            if error_x > 0:
                #drone.right(move_x)
                drone.clockwise(move_x)
            else:
                #drone.left(move_x)
                drone.counter_clockwise(move_x)
        else:
            drone.right(0)
            drone.clockwise(0)

        # Movimiento arriba/abajo
        if abs(error_y) > threshold_y:
            if error_y > 0:
                drone.down(move_y)
            else:
                drone.up(move_y)
        else:
            drone.up(0)

        # Movimiento adelante/atrás
        if w < frame_width * 0.3:
            drone.forward(move_x)
        elif w > frame_width * 0.4:
            drone.backward(move_x)
        else:
            drone.forward(0)
            drone.backward(0)
    else:
        drone.right(0)
        drone.up(0)
        drone.clockwise(0)
        drone.forward(0)
        drone.down(0)



def Stream(drone):

    cont = 0
    path = '/home/p1t/tello-ai/tello-yollov7-ia/images/'
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    try:
        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')
        # skip first 300 frames
        frame_skip = 300
        while True:

            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue

                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
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
                faces_adjusted = [(x * 2, y * 2, w * 2, h * 2) for x, y, w, h in faces]
                control_drone(drone, faces_adjusted, image.shape[:2])

                cv2.imshow('Original',image)
                if detect_x_gesture(image):
                    # Haz que el dron aterrice si se detecta el gesto "X"
                    print("holaadsdasdasssssssssssssssss")
                    drone.land()
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






