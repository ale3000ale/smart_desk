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
        self.touch_plane_A = [4]                      # coefficiente a,b,c,d del piano A ax+by+cz+d=0 
        self.touch_plane_B = [4]                      # coefficiente a,b,c,d del piano B ax+by+cz+d=0

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
            if self.touch_plane_A.__len__() == 4 and self.touch_plane_B.__len__() == 4:
                point = np.array([tip_index.x, tip_index.y, tip_index.z])   
                p_res_A = point[0]*self.touch_plane_A[0] + \
                        point[1]*self.touch_plane_A[1] + \
                        point[2]*self.touch_plane_A[2] + \
                        self.touch_plane_A[3]
                p_res_B = point[0]*self.touch_plane_B[0] + \
                        point[1]*self.touch_plane_B[1] + \
                        point[2]*self.touch_plane_B[2] + \
                        self.touch_plane_B[3]

                if p_res_A >= 0 and p_res_B >= 0:
                    is_real_press = False          # lato del verso della normale
                else:
                    is_real_press = True          # lato opposto
                print(f"point {point } p_res {p_res_A}  ")
            else:
                print("non calibrato")
        
            
            if draw:
                for lm in lms:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                cv2.circle(frame, hand_pos, 8, (255, 0, 0), -1)
                if self.touch_plane_A.__len__() == 4 and self.touch_plane_B.__len__() == 4:
                    text = "REAL PRESS" if is_real_press else "FAKE / HOVER"
                else: text = "NON CALIBRATO "
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
        #------PIANO A------
        p1 = np.array(self.calibration_data[0], dtype=float)
        p2 = np.array(self.calibration_data[1], dtype=float)
        p3 = np.array(self.calibration_data[2], dtype=float)

        # Vettori nel piano
        v1 = p2 - p1
        v2 = p3 - p1

        # Normale al piano (a, b, c)
        n = np.cross(v1, v2)  # [a, b, c] 

        # Termine d: imponi che il piano passi per p1
        d = -np.dot(n, p1)  # ax1 + by1 + cz1 + d = 0  
        self.touch_plane_A = [n[0], n[1], n[2], d]
        #------PIANO B------
        p1 = np.array(self.calibration_data[0], dtype=float)
        p2 = np.array(self.calibration_data[3], dtype=float)
        p3 = np.array(self.calibration_data[2], dtype=float)

        # Vettori nel piano
        v1 = p2 - p1
        v2 = p3 - p1

        # Normale al piano (a, b, c)
        n = np.cross(v1, v2)  # [a, b, c] 

        # Termine d: imponi che il piano passi per p1
        d = -np.dot(n, p1)  # ax1 + by1 + cz1 + d = 0  
        self.touch_plane_B = [n[0], n[1], n[2], d]

    def reset(self):
        
        self.calibration_data.clear()
        self.touch_plane_A.clear()
        self.touch_plane_B.clear()
