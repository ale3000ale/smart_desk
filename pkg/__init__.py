"""
pkg - Pacchetto principale di Smart Desk

Esporta:
- Camera: Gestione videocamera
- Window: Gestione finestra
- config: Configurazioni globali
"""
# ============= LIBRERIE ESTERNE =============
import cv2 
import threading
import sys
import os
import numpy as np
import mediapipe as mp


# ============= LIBRERIE INTERNE =============
from .camera.Camera import Camera
from .window.Window import Window
from .core import core 
from .core import Gui
from .core import HandTracker

# ============= CONFIGURAZIONI =============
from . import config




# ============= ESPORTAZIONI =============
# Esporta tutto quello che serve agli altri moduli
__all__ = [
    # Librerie
    'cv2', 'threading', 'sys', 'os',
    # Config
    'config',
	# Classi
	'Camera', 'Window','Gui',
	# Funzioni
	'core'
]

__version__ = '1.0.0'




