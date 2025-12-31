# pkg/core/handTraker.py
import numpy as np
import cv2, mediapipe as mp
from pkg.config import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import torch
import torch.nn.functional as F
np.set_printoptions(precision=4, suppress=True)


class HandTracker:
    def __init__(self,
                 model_path="hand_landmarker.task"):
        #creazione del tracker
        base_options = python.BaseOptions(model_asset_path=model_path)
        # configura le opzioni per il rilevatore di mani
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,  
            num_hands=MAX_HANDS_DETECTIONS ,
            min_hand_detection_confidence= MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            result_callback = self.callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.timestamp_ms_step = 33 # 1S/30FPS 1S = 1000ms


        # dati per la calibrazione del piano
        self.calibration_data = deque(maxlen=CALIBRATION_POINTS)    # punti di calibrazione
        self.touch_plane  = np.array([])       
        self.calibration_rmse = None # matrice di qualitá   
        self.depth_map = None 
        

        # Seleziona device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {self.device}")
        print("Caricando MiDaS (DPT) da Hugging Face...")

         # ✅ MiDaS SMALL (molto più leggero!)
        print("Caricando MiDaS SMALL (leggero)...")
        self.midas_model_type = "MiDaS_small"  # Modello leggero v2.1
        self.midas_model = torch.hub.load("isl-org/MiDaS", self.midas_model_type)
        self.midas_model = self.midas_model.to(self.device)
        self.midas_model.eval()
        
        midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
        self.midas_transform = midas_transforms.small_transform  # Trasform leggero
        
        print("✓ MiDaS SMALL caricato! (~60MB, molto veloce)")

        # Attributi Funzionali
        self.real_press_threshold = PRESS_TRESHOLD
        self.hands = None



    def callback(self , result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.hands = result

    def load_hands(self,frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.detector.detect_async(mp_image, self.timestamp_ms)
        self.timestamp_ms += self.timestamp_ms_step
        return self.hands

    def get_hands(self):
        return self.hands
    

    def process(self, frame, draw=True):
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
       
        hand_pos = None
        is_real_press = False
        if self.hands != None and self.hands.hand_landmarks:
            
            lms = self.hands.hand_landmarks[0]
            wrist = lms[0]
            #mcp_index = lms[5]
            #pip_index = lms[6]
            dip_index = lms[7]
            tip_index = lms[8]
            
            x_px = int(tip_index.x * w)
            y_px = int(tip_index.y * h)

            z_depth = 0
            if self.depth_map is not None and 0 <= y_px < h and 0 <= x_px < w:
                z_depth = self.depth_map[y_px, x_px]
            
            z_combined = MP_Z_WEIGTH * tip_index.z + MIDAS_Z_WEIGTH * z_depth
            
            hand_pos = (x_px, y_px)

            # Calcola la posizione della mano rispetto al piano di tocco
            if self.touch_plane.__len__() > 0:
                point = np.array([x_px, y_px, z_combined])
                
                # Distanza con segno (normalizzata)
                distance = np.dot(point, self.touch_plane[:3]) + self.touch_plane[3]
                distance_normalized = distance / np.linalg.norm(self.touch_plane[:3])
                
                # Soglia di pressione con isteresi
                if distance_normalized < 0 : #< -self.real_press_threshold:
                    is_real_press = True
                else:
                    is_real_press = False

        
            
            if draw:
                
                if self.touch_plane.__len__() > 0:
                    text = "PRESS" if is_real_press else "HOVER"
                    text += f" | alt: {z_combined}, dist: {distance:.4f} distN: {distance_normalized:.4f}"
                    color = (0, 255, 0) if is_real_press else (0, 0, 255)
                    cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if self.calibration_rmse > 0.02:
                        text = f" AVVISO: RMSE alto ({self.calibration_rmse:.4f})"
                        color = (0, 0, 255)
                        cv2.putText(frame, text, (40, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else: 
                    text = "NON CALIBRATO "
               
                    text += f" {self.calibration_data.__len__()}/{CALIBRATION_POINTS} "
                    color =  (0, 0, 255)
                    cv2.putText(frame, text, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
        return frame, hand_pos, is_real_press
        
    def calibrate_touch_plane(self,frame):

        if self.calibration_data.__len__() < CALIBRATION_POINTS:
            if self.hands != None:
                hand = self.hands[0]
                if hand.hand_landmarks:
                    lms = hand.hand_landmarks[0]
                    calibration_lms = [
                        lms[0],     # wrist
                        lms[5],     # tip_index
                        lms[9],    # tip_middle
                        lms[13],    # tip_ring
                        lms[17]     # tip_pinky
                    ]
                    
                    print("cd bef:",self.calibration_data)
                    calibration_point = np.zeros(3, dtype=float)
                    h, w = frame.shape[:2]
                    for lm in calibration_lms:
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        z_depth = 0
                        if self.depth_map is not None and 0 <= y_px < h and 0 <= x_px < w:
                            z_depth = self.depth_map[y_px, x_px]

                        z_combined = MP_Z_WEIGTH * lm.z + MIDAS_Z_WEIGTH * z_depth
                        calibration_point += np.array([x_px, y_px, z_combined])
                    print("cp:",calibration_point) 
                    calibration_point /= calibration_lms.__len__()
                    self.calibration_data.append(calibration_point)
                    print("cp/len:",calibration_point)
                    print("cd aft:",self.calibration_data)
                else:
                    print("Impossibile calibrare")
                    return False
                if self.calibration_data.__len__() == CALIBRATION_POINTS:
                    self.touch_plane_calculator()
        else:
            print("Calibrazione giá completata")
            return True 
            
    def touch_plane_calculator(self):

        points = np.array(self.calibration_data, dtype=float)

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
    

    def estimate_depth_map(self, frame):
        """
        Stima mappa di profondità con MiDaS da Hugging Face
        Input: frame (H, W, 3) BGR (OpenCV)
        Output: depth_map (H, W) normalizzato 0-1
        """
        h, w = frame.shape[:2]
        
        try:
            # MiDaS small usa direttamente OpenCV (no PIL!)
            input_batch = self.midas_transform(frame).to(self.device)
            
            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                # Resize alla dimensione originale
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            self.depth_map = depth_normalized
            return self.depth_map
            
        except Exception as e:
            print(f"❌ Errore MiDaS Small: {e}")
            self.depth_map = np.ones((h, w)) * 0.5
            return self.depth_map
    
    def reset(self):
        if self.calibration_data != None:
            self.calibration_data.clear()
        if   self.touch_plane.__len__() > 0:
            self.touch_plane.resize(0)

    def draw_landmark(self, frame, show = None , marks = [8]):
        if self.hands != None and self.hands.hand_landmarks:
            h, w = frame.shape[:2]
            for hand in self.hands.hand_landmarks[: show]:
                for idx,lm in enumerate(hand):
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)

                    # Colore base 
                    color = (0, 255, 0)      # verde
                    radius = 3
                    thickness = -1

                    # Se il landmark è nella lista 'marks', viene evidenziato
                    if idx in marks:
                        color = (0, 0, 255)  # rosso
                        radius = 4

                    cv2.circle(frame, (x_px, y_px), radius, color, thickness)
        return frame