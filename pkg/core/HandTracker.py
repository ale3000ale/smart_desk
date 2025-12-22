# pkg/core/handTraker.py
from pkg.ext_import import mp, cv2, np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque


class HandTracker:
    def __init__(self,
                 model_path="hand_landmarker.task",
                 num_hands=1,
                 real_press_threshold=0.014):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  
            num_hands=num_hands ,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.running_mode = vision.RunningMode.VIDEO
        self.real_press_threshold = real_press_threshold
        self.hover_z_ref = deque(maxlen=10) # buffer per calcolare dinamicamente l'altezza della mano
        self.last_hover_z_ref = None
        self.last_hand_pos = None
        self.last_is_real_press = False

    def process(self, frame, timestamp_ms=0, draw=True):
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

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

            

            palm_z = (
                wrist.z + mcp_index.z + mcp_middle.z + 
                mcp_ring.z + mcp_pinky.z + pip_index.z +
                dip_index.z + tip_index.z
            ) / 8.0
            
           
            if len(self.hover_z_ref) < self.hover_z_ref.maxlen:
                self.hover_z_ref.append(palm_z)
                

            
            z_median = np.median(self.hover_z_ref) 
            delta_z = palm_z - z_median
            if delta_z > self.real_press_threshold:
                is_real_press = True
            # se hover_z_ref é pieno e last_hover_z esiste e
            # la differenza tra il valore medio e l'ultima posizione é minore della metá della soglia
            # allora aggiorna il buffer per ridefinire la nuova posizione della mano a riposo
            elif    len(self.hover_z_ref) == self.hover_z_ref.maxlen and \
                    self.last_hover_z_ref is not None and \
                    abs(self.last_hover_z_ref - z_median) < self.real_press_threshold/2:
                self.hover_z_ref.append(palm_z)
            else:
                self.last_hover_z_ref = z_median


            print("PALM_Z MED = ", f"{z_median:.3f}" ,
                  "LAST Z MED = ", f"{self.last_hover_z_ref:.3f}",
                  "delta_z = ",f"{abs(delta_z):.3f}", )
            if draw:
                for lm in lms:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                cv2.circle(frame, hand_pos, 8, (255, 0, 0), -1)
                text = "REAL PRESS" if is_real_press else "FAKE / HOVER"
                color = (0, 255, 0) if is_real_press else (0, 0, 255)
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        self.last_hand_pos = hand_pos
        self.last_is_real_press = is_real_press
        return frame, hand_pos, is_real_press
    
    def calibrate(self, frame, timestamp_ms=0, draw=True):
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

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

            palm_z = (
                wrist.z + mcp_index.z + mcp_middle.z + 
                mcp_ring.z + mcp_pinky.z + pip_index.z +
                dip_index.z + tip_index.z
            ) / 8.0
            
            
            self.hover_z_ref.append(palm_z)
        else:
            print("Impossibile calibrare")
            return False
        return True
