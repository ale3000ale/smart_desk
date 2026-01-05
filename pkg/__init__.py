"""
pkg - Pacchetto principale di Smart Desk

Esporta:
- Camera: Gestione videocamera
- Window: Gestione finestra
- config: Configurazioni globali
"""


# ============= LIBRERIE INTERNE =============
from .camera.Camera import Camera
from .camera.SteroCamera import StereoCamera
from .window.Window import Window

from .core import core 
from .core import Gui
from .core import HandTracker
from .core import utility






# ============= ESPORTAZIONI =============
# Esporta tutto quello che serve agli altri moduli
__all__ = [

    # Config
    'config',
	# Classi
	'Camera','StereoCamera', 'Window','Gui',
	# librerie
	'core', 'utility'
]

__version__ = '1.0.0'




