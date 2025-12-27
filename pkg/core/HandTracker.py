# pkg/core/handTraker.py
import numpy as np
import cv2, mediapipe as mp
from pkg.config import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque



class HandTracker:
    def __init__(self,
                 model_path="hand_landmarker.task",
                 num_hands= MAX_HANDS_DETECTIONS,
                 real_press_threshold=PRESS_TRESHOLD):
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        # crea le opzioni per il rilevatore di mani
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,  
            num_hands=num_hands ,
            min_hand_detection_confidence= MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            result_callback = self.callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.real_press_threshold = real_press_threshold
        self.landmarks = None
        self.last_hand_pos = None
        self.last_is_real_press = False

        # dati per la calibrazione del piano
        self.calibration_data = deque(maxlen=CALIBRATION_POINTS)    # punti di calibrazione
        self.touch_plane  = None       
        self.calibration_rmse = None # matrice di qualitá               

    def callback(self , result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.landmarks = result
       

    def get_hand(self,frame,timestamp_ms=0):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.detector.detect_async(mp_image, timestamp_ms)
        return self.landmarks
    

    def process(self, frame, timestamp_ms=0, draw=True):
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        result = self.get_hand(frame, timestamp_ms)
        hand_pos = None
        is_real_press = False

        if result != None and result.hand_landmarks:
            lms = result.hand_landmarks[0]

            #wrist = lms[0]
            #mcp_index = lms[5]
            #pip_index = lms[6]
            dip_index = lms[7]
            tip_index = lms[8]
            #mcp_middle = lms[9]
            #mcp_ring = lms[13]
            #mcp_pinky = lms[17]
            
            x_px = int(tip_index.x * w)
            y_px = int(tip_index.y * h)
            hand_pos = (x_px, y_px)

            # Calcola la posizione della mano rispetto al piano di tocco
            if self.touch_plane is not None:
                point = np.array([tip_index.x, tip_index.y, tip_index.z])
                
                # Distanza con segno (normalizzata)
                distance = np.dot(point, self.touch_plane[:3]) + self.touch_plane[3]
                distance_normalized = distance / np.linalg.norm(self.touch_plane[:3])
                
                # Soglia di pressione con isteresi
                if distance_normalized < -self.real_press_threshold:
                    is_real_press = True
                else:
                    is_real_press = False
            else:
                print("non calibrato")
        
            
            if draw:
                for lm in lms:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                cv2.circle(frame, hand_pos, 8, (255, 0, 0), -1)
                if self.touch_plane is not None:
                    text = "PRESS" if is_real_press else "HOVER"
                    text += f" | dist: {distance_normalized:.4f}"
                    if self.calibration_rmse > 0.02:
                        text = f"⚠️  AVVISO: RMSE alto ({self.calibration_rmse:.4f})"
                        color = (0, 0, 255)
                        cv2.putText(frame, text, (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else: 
                    text = "NON CALIBRATO "
               
                text += f" {self.calibration_data.__len__()}/{CALIBRATION_POINTS} "
                color = (0, 255, 0) if is_real_press else (0, 0, 255)
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                

        #self.last_hand_pos = hand_pos
        #self.last_is_real_press = is_real_press
        return frame, hand_pos, is_real_press
        
    def calibrate_touch_plane(self,frame, timestamp_ms):

        if self.calibration_data.__len__() < CALIBRATION_POINTS:
            result = self.get_hand(frame,timestamp_ms)
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                calibration_lms = [
                    #lms[0],     # wrist
                    lms[4],     # tip_thumb
                    lms[8],     # tip_index
                    lms[12],    # tip_middle
                    lms[16],    # tip_ring
                    lms[20]     # tip_pinky
                ]
                
                
                calibration_point = np.zeros(3, dtype=float)
                print(calibration_point)
                for lm in calibration_lms:
                    calibration_point += np.array([lm.x, lm.y, lm.z]) 
                calibration_point /= calibration_lms.__len__()
                self.calibration_data.append(calibration_point)
            else:
                print("Impossibile calibrare")
                return False
            if self.calibration_data.__len__() == CALIBRATION_POINTS:
                self.touch_plane_calculator()

                
        else:
            print("Calibrazione giá completata")
            return True 
            
    def touch_plane_calculator(self):

        points = np.array(  [
                            self.calibration_data[0],  # Angolo A
                            self.calibration_data[1],  # Angolo B
                            self.calibration_data[2],  # Angolo C
                            self.calibration_data[3],  # Angolo D
                            self.calibration_data[4]   # CENTRO 
                            ], dtype=float)
        # Calcola la media dei  punti
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        
        # Applica SVD (Singular Value Decomposition)
        # Usa la Matrice di Covarianza 
        # @ moltiplicazione tra matrici
        U, S, Vt = np.linalg.svd(points_centered.T @ points_centered)
        
        # La normale è il vettore singolare con autovalore minore
        # (ultima riga di U, non Vt)
        normal = U[:, -1]  # Autovettore corrispondente a σ_min
        
        # Calcola d del piano ax + by + cz + d = 0
        d = -np.dot(normal, centroid)
        
        # Memorizza il piano
        self.touch_plane = np.array([normal[0], normal[1], normal[2], d])

        # Calcola la qualità del fitting (RMSE) Radice dell'errore quadratico medio
        distances = np.abs(points @ self.touch_plane[:3] + self.touch_plane[3]) / \
                    np.linalg.norm(self.touch_plane[:3])
        self.calibration_rmse = np.sqrt((distances ** 2).mean())

        print(f"✓ Piano calibrato | RMSE: {self.calibration_rmse:.6f}")
        print(f"  Normale: ({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})")
        print(f"  d: {d:.6f}")
        
        if self.calibration_rmse > 0.02:
            print(f"⚠️  AVVISO: RMSE alto ({self.calibration_rmse:.4f})")
            print("   Ricalibra con punti più separati o mano più ferma")
            return False
        return True
    
    def reset(self):
        if self.calibration_data != None:
            self.calibration_data.clear()
        if self.touch_plane != None:
            self.touch_plane.clear()
