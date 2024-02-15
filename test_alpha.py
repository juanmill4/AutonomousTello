import cv2
import numpy as np
import tellopy
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

# ...
# Dentro de tu bucle principal de procesamiento de imágenes
if detect_x_gesture(image):
    # Haz que el dron aterrice si se detecta el gesto "X"
    drone.land()


def handle_video_frame(event, sender, data, **args):
    frame = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow("Tello Video", frame)

    # Presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sender.quit()


def handle_video_data(event, sender, data, **args):
    print(f"Video data length: {len(data)}")


def main():
    drone = tellopy.Tello()

    # Suscribirse a los eventos de video
    drone.subscribe(drone.EVENT_VIDEO_FRAME, handle_video_frame)
    drone.subscribe(drone.EVENT_VIDEO_DATA, handle_video_data)

    drone.connect()
    drone.wait_for_connection(60.0)
    drone.start_video()



if __name__ == '__main__':
    main()
