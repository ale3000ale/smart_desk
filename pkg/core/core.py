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
from pkg.core import utility as ut
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np


stereo = True

colormaps = {
        'TURBO': cv2.COLORMAP_TURBO,
        'JET': cv2.COLORMAP_JET,
        'HOT': cv2.COLORMAP_HOT,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'MAGMA': cv2.COLORMAP_MAGMA,
    }

def core():
	color = cv2.COLORMAP_TURBO
	wd = Window("Camera Viewer")

	wd_depth = Window("Depth map p2 base 32")
	wd_depth_Color = Window("Color Depth map")
	
	if stereo:
		stereo_camera = StereoCamera()
		stereo_camera.load_stereo_calibration("stereo_calib.npz")
	else:
		mono_cameraL = Camera(0)
		mono_cameraR = Camera(1)
		wd_lr = Window("Non Stereo Camera L | R", 1080*2, 720)

	#start ML
	tracker = HandTracker()
	#start elaboratore

	#start GUI
	#gui = Gui(width=WINDOW_DEFAULT_WIDTH, height=WINDOW_DEFAULT_HEIGHT)    


	while True:

		if stereo:
			ret_l, ret_r, frame_l, frame_r = stereo_camera.read()
		else:
			ret_l,frame_l = mono_cameraL.read()
			ret_r,frame_r = mono_cameraR.read()

		if not (ret_l and ret_r):
			#print("Errore nella lettura dei frame")
			continue

		# Se hai calibrazione, rettifica
		if stereo and stereo_camera.stereo_params is not None:
				frame_l, frame_r = stereo_camera.rectify_frames(frame_l, frame_r) 

		depth_map = tracker.new_estimate_depth_map(frame_l, frame_r, stereo_camera.stereo_params)
		#depth_map = tracker.estimate_depth_map(frame_l, frame_r)


		tracker.load_hands(frame_l)
		frame_tracked, hand_pos, is_real_press = tracker.process(frame_l)

		tracker.draw_landmark(frame_tracked)
		#tracker.draw_landmark(depth_map)

		depth_map_uint8 = (depth_map * 255).astype(np.uint8)
		depth_map_colored = cv2.applyColorMap(depth_map_uint8, color)
		wd_depth_Color.show_frame(depth_map_colored)

		# Rendering GUI sopra il frame tracciato
		#gui_frame = gui.render(frame_tracked)

		# Mostra i frame
		
		wd.show_frame(frame_tracked)
		if not stereo:
			wd_lr.show_frame(np.hstack([frame_l, frame_r]))
		wd_depth.show_frame(depth_map)
		
		# Listener pulsante GUI
		#if gui.consume_button_press():
		#	print("Pulsante GUI premuto!")

		# Tasti controllo
		key = cv2.waitKey(1) & 0xFF
		if key == ord(config.KEY_QUIT):
			break
		if key == ord(config.KEY_CHANGE_CAMERA):
			if  stereo:
				stereo_camera.change_cameras()
			else:
				mono_cameraL.change_camera()
				mono_cameraR.change_camera()
		if key == ord(config.KEY_CALIBRATION):
			tracker.calibrate_touch_plane(frame_r)
		if key == ord(config.KEY_RESET):
			tracker.reset()
		if key == ord('s'):
			ut.depth_map_to_csv(depth_map, "assets/depth_map.csv")
			ut.save_depth_map_png(depth_map, "assets/depth_map.png")
			
			tracker.reset()

		if key == ord('1'): color = colormaps['TURBO']    # Miglior scelta
		if key == ord('2'): color = colormaps['JET']
		if key == ord('3'): color = colormaps['HOT']
		if key == ord('4'): color = colormaps['VIRIDIS']
		if key == ord('5'): color = colormaps['MAGMA']

		
	
	stereo_camera.release()
	cv2.destroyAllWindows()