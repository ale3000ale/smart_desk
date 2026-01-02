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
from pkg.camera import SteroCamera
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import torch

def core():
	wd = Window("Camera Viewer")
	wd_l = Window("Camera Left")
	wd_r = Window("Camera Rigth")
	wd_depth = Window("Depth map")
	
	#mono_camera = Camera(2)
	stereo_camera = StereoCamera(0,1)
	stereo_camera.load_stereo_calibration("stereo_calib.npz")
	#stereo_camera.set_dimensions(CAMERA_DEFAULT_WIDTH, CAMERA_DEFAULT_HEIGHT)
	#start ML
	tracker = HandTracker()
	#start elaboratore
	#start GUI
	gui = Gui(width=WINDOW_DEFAULT_WIDTH, height=WINDOW_DEFAULT_HEIGHT)    


	while True:

		#RENDER  GUI
		ret_l, ret_r, frame_l, frame_r = stereo_camera.read()
		if not (ret_l and ret_r):
			#print("Errore nella lettura del frame stereo")
			continue

		# Se hai calibrazione, rettifica
		if stereo_camera.stereo_params is not None:
			print("rettifico")
			frame_l, frame_r = stereo_camera.rectify_frames(frame_l, frame_r) 

		depth_map = tracker.estimate_depth_map(frame_l, frame_r)

		tracker.load_hands(frame_l)
		frame_tracked, hand_pos, is_real_press = tracker.process(frame_l)
		#print("Tracking con L")
		tracker.draw_landmark(frame_tracked)
		tracker.draw_landmark(frame_l)
		tracker.draw_landmark(frame_r)
		tracker.draw_landmark(depth_map)

		# Rendering GUI sopra il frame tracciato
		gui_frame = gui.render(frame_tracked)

		# Mostra i frame
		wd.show_frame(gui_frame)
		wd_l.show_frame(frame_l)
		wd_r.show_frame(frame_r)
		wd_depth.show_frame(depth_map)

		# Listener pulsante GUI
		if gui.consume_button_press():
			print("Pulsante GUI premuto!")

		# Tasti controllo
		key = cv2.waitKey(1) & 0xFF
		if key == ord(config.KEY_QUIT):
			break
		if key == ord(config.KEY_CHANGE_CAMERA):
			stereo_camera.change_camera()
		if key == ord(config.KEY_CALIBRATION):
			tracker.calibrate_touch_plane(frame_r)
		if key == ord(config.KEY_RESET):
			tracker.reset()

		
	
	stereo_camera.release()
	cv2.destroyAllWindows()