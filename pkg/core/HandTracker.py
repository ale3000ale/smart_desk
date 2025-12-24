# pkg/core/handTraker.py
import numpy as np
import cv2, mediapipe as mp
from pkg.config import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

def callback():
    print("Hand Tracker callback")

class HandTracker:
    def __init__(self,
                 model_path="hand_landmarker.task",
                 num_hands=1,
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
            result_callback = callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.real_press_threshold = real_press_threshold
        self.hover_z_ref = deque(maxlen=10) # buffer per calcolare dinamicamente l'altezza della mano
        self.last_hover_z_ref = None
        self.last_hand_pos = None
        self.last_is_real_press = False

        # dati per la calibrazione del piano
        self.calibration_data = deque(maxlen=10)    # punti di calibrazione
        self.touch_plane = [4]                      # coefficiente a,b,c,d del piano ax+by+cz+d=0

    def get_hand(self,frame,timestamp_ms=0):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.detector.detect_async(mp_image, timestamp_ms)
    

    def process(self, frame, timestamp_ms=0, draw=True):
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        result = self.get_hand(frame, timestamp_ms)

        hand_pos = None
        is_real_press = False

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]

            wrist = lms[0]
            mcp_index = lms[5]
            pip_index = lms[6]
            dip_index = lms[7]
            tip_index = lms[8]
            mcp_middle = lms[9]
            mcp_ring = lms[13]
            mcp_pinky = lms[17]
            
            x_px = int(tip_index.x * w)
            y_px = int(tip_index.y * h)
            hand_pos = (x_px, y_px)

            

            #palm_z = (
            #    wrist.z + mcp_index.z + mcp_middle.z + 
            #    mcp_ring.z + mcp_pinky.z + pip_index.z +
            #    dip_index.z + tip_index.z
            #) / 8.0
            # Calcola la posizione della mano rispetto al piano di tocco
            if self.touch_plane.__len__() == 4:
                point = np.array([dip_index.x, dip_index.y, dip_index.z])   
                p_res = point[0]*self.touch_plane[0] + \
                        point[1]*self.touch_plane[1] + \
                        point[2]*self.touch_plane[2] + \
                        self.touch_plane[3]

                if p_res > 0:
                    is_real_press = False          # lato del verso della normale
                else:
                    is_real_press = True          # lato opposto
                print(f"point {point } p_res {p_res} ")
            else:
                print("non calibrato")
            #if len(self.hover_z_ref) < self.hover_z_ref.maxlen:
            #    self.hover_z_ref.append(palm_z)
                

            
            #z_median = np.median(self.hover_z_ref) 
            #delta_z = palm_z - z_median
            #if delta_z > self.real_press_threshold:
            #    is_real_press = True

            # se hover_z_ref é pieno e last_hover_z esiste e
            # la differenza tra il valore medio e l'ultima posizione é minore della metá della soglia
            # allora aggiorna il buffer per ridefinire la nuova posizione della mano a riposo
            
            #elif    len(self.hover_z_ref) == self.hover_z_ref.maxlen and \
            #        self.last_hover_z_ref is not None and \
            #        abs(self.last_hover_z_ref - z_median) < self.real_press_threshold/2:
            #    self.hover_z_ref.append(palm_z)
            #else:
            #    self.last_hover_z_ref = z_median


            #print("PALM_Z MED = ", f"{z_median:.3f}" ,
            #      "LAST Z MED = ", f"{self.last_hover_z_ref:.3f}",
            #      "delta_z = ",f"{abs(delta_z):.3f} ",
            #      f"X:{tip_index.x:.3f} Y:{tip_index.y:.3f} Z:{tip_index.z:.3f}")
            
            if draw:
                for lm in lms:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                cv2.circle(frame, hand_pos, 8, (255, 0, 0), -1)
                if self.touch_plane.__len__() == 4:
                    text = "REAL PRESS" if is_real_press else "FAKE / HOVER"
                else: text = "NON CALIBRATO "
                text += f" {self.calibration_data.__len__()}/3 "
                color = (0, 255, 0) if is_real_press else (0, 0, 255)
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        #self.last_hand_pos = hand_pos
        #self.last_is_real_press = is_real_press
        return frame, hand_pos, is_real_press
        
    def calibrate_touch_plane(self,frame, timestamp_ms):
        #self.calibration_data = deque(maxlen=10)    # punti di calibrazione
        #self.touch_plane = [4]
        if self.calibration_data.__len__() < 3:
            result = self.get_hand(frame,timestamp_ms)
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]

                wrist = lms[0]
                mcp_index = lms[5]
                mcp_middle = lms[9]
                mcp_ring = lms[13]
                mcp_pinky = lms[17]
                
                
                calibration_dot = ( np.array([wrist.x,wrist.y,wrist.z]) + np.array([mcp_index.x,mcp_index.y,mcp_index.z]) +
                                    np.array([mcp_middle.x,mcp_middle.y,mcp_middle.z]) + np.array([mcp_ring.x,mcp_ring.y,mcp_ring.z]) +
                                    np.array([mcp_pinky.x,mcp_pinky.y,mcp_pinky.z])
                ) / 5.0
                self.calibration_data.append(calibration_dot)
            else:
                print("Impossibile calibrare")
                return False
            if self.calibration_data.__len__() == 3:
                self.touch_plane = self.touch_plane_calculator()
        else:
            print("Calibrazione giá completata")
            return True 
            
    def touch_plane_calculator(self):
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
        return [n[0], n[1], n[2], d]

    def reset(self):
        
        self.hover_z_ref.clear()
        self.calibration_data.clear()
        self.touch_plane.clear()
