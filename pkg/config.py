
"""
config.py - Configurazioni e costanti globali
"""

import os

# ============= COSTANTI HAND TRACKER =============
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_HANDS_DETECTIONS = 2
PRESS_TRESHOLD = 0.005
CALIBRATION_POINTS = 5
MP_Z_WEIGTH = 0.0
MIDAS_Z_WEIGTH = 1.0


# ============= COSTANTI VIDEOCAMERA =============
CAMERA_DEFAULT_INDEX = 0
CAMERA_DEFAULT_WIDTH = 1280
CAMERA_DEFAULT_HEIGHT = 720
CAMERA_USE_THREADING = True
CAMERA_DEFAULT_FPS = 30

# ============= COSTANTI FINESTRA =============
WINDOW_DEFAULT_NAME = "Smart Desk"
WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 720

# ============= COSTANTI CONTROLLO =============
KEY_QUIT = 'q'              # Tasto per uscire
KEY_CHANGE_CAMERA = 'c'     # Tasto per cambiare videocamera
KEY_CALIBRATION = 't'       # Tasto per calibrare
KEY_RESET = 'r'             # Tasto per resettare la calibrazione
#KEY_SET_720P = 'p'        # Tasto per settare risoluzione 720p

# ============= PERCORSI =============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_PATH = os.path.join(PROJECT_ROOT, 'pkg')
CORE_PATH = os.path.join(PROJECT_ROOT, 'core')

# ============= ESPORTAZIONI =============
__all__ = [
    'CAMERA_DEFAULT_INDEX',  'CAMERA_DEFAULT_WIDTH', 'CAMERA_DEFAULT_HEIGHT',
    'CAMERA_USE_THREADING', 'CAMERA_DEFAULT_FPS', 
	
	'WINDOW_DEFAULT_NAME','WINDOW_DEFAULT_WIDTH', 'WINDOW_DEFAULT_HEIGHT',
	
	'KEY_QUIT', 'KEY_CHANGE_CAMERA','KEY_CALIBRATION',
	'KEY_RESET',
	
	'MIN_HAND_DETECTION_CONFIDENCE',
    'MIN_HAND_PRESENCE_CONFIDENCE','MIN_TRACKING_CONFIDENCE','MAX_HANDS_DETECTIONS',
	'CALIBRATION_POINTS','PRESS_TRESHOLD' ,'PROJECT_ROOT', 'PKG_PATH', 'CORE_PATH',
	'MP_Z_WEIGTH', 'MIDAS_Z_WEIGTH'
]
