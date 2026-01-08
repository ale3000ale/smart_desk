# pkg/core/handTraker.py
import numpy as np
import cv2, mediapipe as mp
from pkg.config import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
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
        

        # Attributi Funzionali
        self.real_press_threshold = PRESS_TRESHOLD
        self.hands = None

    def get_depth_map(self):
        return self.depth_map


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
            #print(f"Cm puntati {self.depth_map[x_px,y_px]}")

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
    

    def new_estimate_depth_map(self, frame_left, frame_right, stereoSGBM_params = None ,stereo_params = None , verbose=True):
        """
        Stima depth map. Rettifica INTERNAMENTE se necessario.
        """
       
        

        
        # INPUT CHECK
        if frame_left is None or frame_right is None:
            return np.ones((480, 640, 1), dtype=np.float32) * 0.5
        
        h, w = frame_left.shape[:2]
        
        
        
        # PREPROCESSING
        if frame_left.ndim == 3:
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left, gray_right = frame_left.copy(), frame_right.copy()
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #gray_left = clahe.apply(gray_left)
        #gray_right = clahe.apply(gray_right)
        
        # Blur
        #gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        #gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)

        
        window_size = stereoSGBM_params['blockSize']
        min_disp = stereoSGBM_params['minDisparity']
        nDispFactor = stereoSGBM_params['numDisparities']
        num_disp =  16 * nDispFactor 
        num_disp = num_disp if num_disp != 0 else 16
        # STEREO MATCHING
        stereo = cv2.StereoSGBM_create(
            minDisparity= min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 *stereoSGBM_params['P1'] * window_size ** 2,
            P2=32 * stereoSGBM_params['P2']  * window_size ** 2,
            disp12MaxDiff=stereoSGBM_params['disp12MaxDiff'],
            uniquenessRatio=stereoSGBM_params['uniquenessRatio'],
            speckleWindowSize=stereoSGBM_params['speckleWindowSize'],
            speckleRange=stereoSGBM_params['speckleRange'],
            mode=cv2.STEREO_SGBM_MODE_HH
        )
        
        disparity_raw = stereo.compute(gray_left, gray_right)
        disparity_map = disparity_raw.astype(np.float32) / 16.0
        #rimuovere normalizzazione
        #disparity_map = (disparity_map - min_disp) / num_disp




        #return disparity_map

            
        # PARAMETRI CALIBRAZIONE
        BASELINE_MM = 100.0  # Distanza tra telecamere in mm
        FOCAL_LENGTH_PX = 797.0  # Lunghezza focale in pixel (dalla calibrazione)
        MIN_Z = 50.0  # mm
        MAX_Z = 1000.0  # mm
    
        # CALCOLO PROFONDITA' (Z) IN MM
        # Formula: Z = (baseline * focal_length) / disparity
        depth_map_mm = np.zeros((h, w), dtype=np.float32)
        
        # Crea maschera per disparità valida
        mask = disparity_map > 0.5  # Evita divisione per zero
        
        # Applica formula stereo
        depth_map_mm[mask] = (BASELINE_MM * FOCAL_LENGTH_PX) / disparity_map[mask]
        
        # Clipping per rimuovere valori non realistici
        depth_map_mm[mask] = np.clip(depth_map_mm[mask], MIN_Z, MAX_Z)
        
        # Imposta pixel non validi a valore di fallback (e.g., distanza massima)
        depth_map_mm[~mask] = MAX_Z
        
        # CONVERTI IN CM (dividi per 10)
        depth_map_cm = depth_map_mm / 10.0
        
        
        if verbose:
            valid_pixels = np.count_nonzero(mask)
            print(f"[DEBUG DEPTH] Pixel validi: {valid_pixels}/{h*w} ({100*valid_pixels/(h*w):.1f}%)")
            if valid_pixels > 0:
                z_min = depth_map_cm[mask].min()
                z_max = depth_map_cm[mask].max()
                z_mean = depth_map_cm[mask].mean()
                print(f"[DEBUG DEPTH] Range Z: {z_min:.1f} - {z_max:.1f} cm (media: {z_mean:.1f} cm)")
        
        alid_pixels = np.count_nonzero(mask)
        print(f"[DEBUG] Pixel validi: {valid_pixels}/{h*w}")
        if valid_pixels > 0:
            z_min = depth_map_cm[mask].min()
            z_max = depth_map_cm[mask].max()
            z_mean = depth_map_cm[mask].mean()
            print(f"[DEBUG] Range: {z_min:.1f} - {z_max:.1f} cm (media: {z_mean:.1f} cm)")
    

        # Normalizza tra 0 e 1 rispetto al range [MIN_Z, MAX_Z]
        depth_map_normalized = (depth_map_cm - MIN_Z/10.0) / (MAX_Z/10.0 - MIN_Z/10.0)
        depth_map_normalized = np.clip(depth_map_normalized, 0.0, 1.0)
        
        if verbose:
            print(f"[DEBUG] Output NORMALIZZATO 0-1")
        
        return depth_map_normalized
    
        return depth_map_cm

    

    def estimate_depth_map(self, frame_left, frame_right, p2_base = 32):
        """
        Stima mappa di profondità da due frame stereo (left, right) usando StereoSGBM.
        
        Input: 
            frame_left, frame_right = immagini BGR rettificate, stessa dimensione.
        
        Output: 
            self.depth_map = H x W, float32 normalizzata 0-1.
        
        Miglioramenti per ridurre rumore:
        1. Pre-processing: Gaussian blur su input
        2. Parametri SGBM robusti: blockSize=11, uniquenessRatio=15
        3. Post-processing: Median blur + morphological closing (opzionale)
        """
        
        # ========================================================================
        # STEP 0: VALIDAZIONE INPUT
        # ========================================================================
        if frame_left is None or frame_right is None:
            if self.depth_map is None:
                h, w = frame_left.shape[:2] if frame_left is not None else frame_right.shape[:2]
                self.depth_map = np.ones((h, w), dtype=np.float32) * 0.5
            return self.depth_map

        # ========================================================================
        # STEP 1: CONVERSIONE A GRAYSCALE
        # ========================================================================
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # ========================================================================
        # STEP 2: PRE-PROCESSING - RIDUZIONE RUMORE INPUT
        # ========================================================================
        # Gaussian blur (5x5) riduce rumore sensore senza perdere feature
        # Risultato: 40-60% meno rumore nel matching stereo
        #gray_left = cv2.GaussianBlur(gray_left, (5, 5), 1.0)
        #gray_right = cv2.GaussianBlur(gray_right, (5, 5), 1.0)

        # -------------------------------------------------------------
        # MIGLIORAMENTO PRE-PROCESSING (CLAHE invece di Gaussian Blur)
        # -------------------------------------------------------------

        # Crea l'oggetto CLAHE
        # clipLimit: soglia per limitare il contrasto (2.0 è standard, 3-4 per più dettaglio)
        # tileGridSize: dimensione delle celle per l'equalizzazione locale (8x8 è standard)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Applica CLAHE alle immagini grayscale rettificate
        #gray_left = clahe.apply(gray_left)
        #gray_right = clahe.apply(gray_right)

        # ========================================================================
        # STEP 3: CONFIGURAZIONE StereoSGBM CON PARAMETRI ROBUSTI
        # ========================================================================
        window_size = 15                   
        min_disp = 0
        num_disp = 128                      

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,         # 11 = finestra matching robusta
            P1=8 * 3 * window_size ** 2,   # Penalità discontinuità piccola
            P2= p2_base * 3 * window_size ** 2,  # Penalità discontinuità grande
            disp12MaxDiff=1,               # Controlla left-right consistency
            uniquenessRatio=20,            # 15 (era 10) → scarta match deboli/ambigui
            speckleWindowSize=200,         # Rimuove componenti < 100 pixel
            speckleRange=32                # Range massimo variazione
        )

        # ========================================================================
        # STEP 4: CALCOLO DISPARITÀ
        # ========================================================================
        # compute() ritorna int16 scalato x16, converti a float32
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
        # ========================================================================
        # STEP 5: POST-PROCESSING - FILTRO OUTPUT
        # ========================================================================
        
        # 5a) Copia e invalida disparità negativa
        disparity_valid = disparity.copy()
        disparity_valid[disparity_valid <= 0] = np.nan
        
        # 5b) Trova range valido
        disp_min_valid = np.nanmin(disparity_valid)
        disp_max_valid = np.nanmax(disparity_valid)
        
        # 5c) Applica MEDIAN BLUR (rimuove outlier mantenendo edge)
        # Perché median e non gaussian? 
        #   - Median preserva bordi sharp, rimuove SOLO spike isolati
        #   - Gaussian smussa tutto, perdi dettagli
        if (np.isfinite(disp_min_valid) and 
            np.isfinite(disp_max_valid) and 
            disp_max_valid - disp_min_valid > 1e-6):
            
            # Normalizza a [0, 255] per median blur
            disp_norm_temp = (disparity_valid - disp_min_valid) / (disp_max_valid - disp_min_valid)
            disp_uint8 = (disp_norm_temp * 255).astype(np.uint8)
            
            # Median blur: finestra 5x5
            disp_uint8_filtered = cv2.medianBlur(disp_uint8, ksize=5)
            
            # Converti indietro a scale originale
            disparity_filtered = (disp_uint8_filtered.astype(np.float32) / 255.0) * (disp_max_valid - disp_min_valid) + disp_min_valid
        else:
            disparity_filtered = disparity.copy()

        # ========================================================================
        # STEP 6: NORMALIZZAZIONE FINALE
        # ========================================================================
        
        # Invalida disparità negativa
        disparity_filtered[disparity_filtered <= 0] = np.nan

        # Normalizza disparità in [0, 1] ignorando NaN
        disp_min = np.nanmin(disparity_filtered)
        disp_max = np.nanmax(disparity_filtered)

        # Gestisci caso fallback (no disparità valida)
        if not np.isfinite(disp_min) or not np.isfinite(disp_max) or disp_max - disp_min < 1e-6:
            if self.depth_map is None:
                h, w = gray_left.shape
                self.depth_map = np.ones((h, w), dtype=np.float32) * 0.5
            return self.depth_map

        # Normalizza a [0, 1]
        disp_norm = (disparity_filtered - disp_min) / (disp_max - disp_min)

        # Inverti: "più vicino = valore più alto"
        # (Opzionale: commenta se vuoi "più vicino = valore più basso")
        depth = 1.0 - disp_norm

        # Sostituisci eventuali NaN con 0.5 (fallback per pixel invalidi)
        depth = np.nan_to_num(depth, nan=0.5)

        # Salva depth map
        self.depth_map = depth.astype(np.float32)
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