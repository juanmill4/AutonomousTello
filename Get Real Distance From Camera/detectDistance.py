import cv2
import cv2.aruco as aruco
import numpy as np


SIZE = 94.5

# Cargar los datos de calibración de la cámara
with np.load('/home/p1t/tello-ai/calib_data/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

url = 'http://10.42.0.65:4747/video'
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    resized_frame = cv2.resize(frame, (1200, 700))

    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, mtx, dist)
        for i, id in enumerate(ids):
            # Dibujar cuadrado y ID
            aruco.drawDetectedMarkers(resized_frame, corners, ids)
            
            # Calcular y mostrar la distancia a la cámara
            tvec = tvecs[i][0]
            distancia_a_camara = np.linalg.norm(tvec)
            cv2.putText(resized_frame, f"ID: {id[0]}, Dist: {distancia_a_camara:.2f}", (int(corners[i][0][0][0]), int(corners[i][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        
        marker_positions = {id[0]: tvec[0] for id, tvec in zip(ids, tvecs)}
        # Dibujar líneas entre marcadores
        if len(corners) > 1:
            for i in range(len(corners) - 1):
                for j in range(i + 1, len(corners)):
                    id_i, id_j = ids[i][0], ids[j][0]
                    if id_i in marker_positions and id_j in marker_positions:
                        pos_i, pos_j = marker_positions[id_i], marker_positions[id_j]
                        distancia = np.linalg.norm(pos_i - pos_j)
                        midpoint = (corners[i][0][0] + corners[j][0][0]) // 2
                        cv2.line(resized_frame, tuple(corners[i][0][0]), tuple(corners[j][0][0]), (255, 0, 0), 2)
                        cv2.putText(resized_frame, f"{distancia:.2f}m", tuple(midpoint.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    # Mostrar imagen
    cv2.imshow('frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
