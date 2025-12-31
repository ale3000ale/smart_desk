"""
	Core crea e gestisce ML , elaboratore, GUI 
	ML crea eventi in base a ció che la camera vede
	|-->Elaboratore elabora gli eventi ed i dati provenienti da ML e dice alla GUI cosa mostrare
			|-->GUI si occupa solo del rendering 
"""

from pkg import *
from pkg.config import *
from pkg.core.Gui import Gui
from pkg.core.HandTracker import HandTracker
import cv2
import torch

def core():
	wd = Window("Camera Viewer")
	wd_depth = Window("Depth map")
	camera = Camera()
	# SOLO PER QUANDO USO WINDOWS
	camera.set_dimensions(CAMERA_DEFAULT_WIDTH, CAMERA_DEFAULT_HEIGHT)

	#start ML
	tracker = HandTracker()
	#start elaboratore
	#start GUI
	gui = Gui(width=WINDOW_DEFAULT_WIDTH, height=WINDOW_DEFAULT_HEIGHT)    


	print("CUDA disponibile:", torch.cuda.is_available())
	print("Dispositivo predefinito:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	print("Nome GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nessuna") 
	while True:

		#RENDER  GUI
		ret, frame = camera.read() 
		if not ret:
			print("Errore nella lettura del frame")
			break
		
		depth_map = tracker.estimate_depth_map(frame)
		tracker.load_hands(frame)
		frame_tracked, hand_pos, is_real_press = tracker.process(frame)

		tracker.draw_landmark(frame)
		tracker.draw_landmark(depth_map)

		# Rendering GUI sopra il frame tracciato
		gui_frame = gui.render(frame_tracked)

		# Mostra i frame
		wd.show_frame(gui_frame)
		wd_depth.show_frame(depth_map)

		# Listener pulsante GUI
		if gui.consume_button_press():
			print("Pulsante GUI premuto!")

		# Tasti controllo
		key = cv2.waitKey(1) & 0xFF
		if key == ord(config.KEY_QUIT):
			break
		if key == ord(config.KEY_CHANGE_CAMERA):
			camera.change_camera()
		if key == ord(config.KEY_CALIBRATION):
			tracker.calibrate_touch_plane(frame)
		if key == ord(config.KEY_RESET):
			tracker.reset()

		
	
	camera.release()
	cv2.destroyAllWindows()