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
from pkg.ext_import import cv2


def core():
	wd = Window("Camera Viewer")
	camera = Camera()
	# SOLO PER QUANDO USO WINDOWS
	camera.set_dimensions(CAMERA_DEFAULT_WIDTH, CAMERA_DEFAULT_HEIGHT)

	#start ML
	tracker = HandTracker()
	#start elaboratore
	#start GUI
	gui = Gui(width=WINDOW_DEFAULT_WIDTH, height=WINDOW_DEFAULT_HEIGHT)    

	timestamp_ms = 0  
	while True:

		#RENDER  GUI
		ret, frame = camera.read() 
		if not ret:
			print("Errore nella lettura del frame")
			break
		
		
		frame_tracked, hand_pos, is_real_press = tracker.process(
								frame, timestamp_ms=timestamp_ms)
		timestamp_ms += 33  # ~30 FPS

		#if is_real_press and hand_pos is not None:
		#	print("Press reale:", hand_pos)

		#  Rendering GUI sopra il frame tracciato
		gui_frame = gui.render(frame_tracked)

		#  Mostra SOLO questo frame finale
		wd.show_frame(gui_frame)

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
			tracker.calibrate_touch_plane(frame, timestamp_ms=timestamp_ms)
			timestamp_ms += 33
		if key == ord(config.KEY_RESET):
			tracker.reset()

		
	
	camera.release()
	cv2.destroyAllWindows()