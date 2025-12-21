
"""
config.py - Configurazioni e costanti globali
"""

import os

# ============= COSTANTI VIDEOCAMERA =============
CAMERA_DEFAULT_INDEX = 0
CAMERA_DEFAULT_WIDTH = 1080
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

# ============= PERCORSI =============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_PATH = os.path.join(PROJECT_ROOT, 'pkg')
CORE_PATH = os.path.join(PROJECT_ROOT, 'core')

# ============= ESPORTAZIONI =============
__all__ = [
    'CAMERA_DEFAULT_INDEX',  'CAMERA_DEFAULT_WIDTH', 'CAMERA_DEFAULT_HEIGHT',
    'CAMERA_USE_THREADING', 'CAMERA_DEFAULT_FPS', 'WINDOW_DEFAULT_NAME',
    'WINDOW_DEFAULT_WIDTH', 'WINDOW_DEFAULT_HEIGHT', 'KEY_QUIT', 'KEY_CHANGE_CAMERA',
    'PROJECT_ROOT', 'PKG_PATH', 'CORE_PATH'
]
