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
import numpy 
import mediapipe 




# ============= ESPORTAZIONI =============
# Esporta tutto quello che serve agli altri moduli
__all__ = [
    # Librerie
    'cv2', 'threading', 'sys', 'os','mediapipe', 'numpy',

]




