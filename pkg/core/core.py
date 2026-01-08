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


stereoSGBM_params_steps = [['minDisparity', 1],
		['numDisparities', 1],
		['blockSize', 1],
		['P1', 1],
		['P2', 1],
		['disp12MaxDiff', 1],
		['uniquenessRatio', 1],
		['speckleWindowSize', 10],
		['speckleRange', 1]
		]

stereoSGBM_params =  {
		'minDisparity': 94,
		'numDisparities': 2,
		'blockSize':1,
		'P1': 1,
		'P2': 3,
		'disp12MaxDiff': 1,
		'uniquenessRatio': 3,
		'speckleWindowSize': 10,
		'speckleRange': 5
	}

def add_to_param(key):
	stereoSGBM_params[key[0]] += key[1]

def sub_to_param(key):
	stereoSGBM_params[key[0]] -= key[1]


def draw_vertical_dict(frame, data_dict, id, start_pos=(20, 40), font_scale=0.6):
	"""
	Visualizza un dizionario in verticale sul frame.
	
	Args:
		frame: Il frame/immagine cv2 su cui disegnare.
		data_dict: Il dizionario contenente i valori da mostrare.
		start_pos: Tupla (x, y) per la posizione della prima riga.
		font_scale: Dimensione del font.
		color: Colore del testo in BGR (es. (0, 255, 0) per verde).
	"""
	x, y = start_pos
	line_height = int(30 * font_scale * 1.5)  # Calcola interlinea in base al font
	
	# Impostazioni font
	font = cv2.FONT_HERSHEY_SIMPLEX
	thickness = 2

	for key, value in data_dict.items():
		# Formatta il testo: Chiave: Valore
		# Se il valore è float, lo tronca a 2 decimali per pulizia
		if isinstance(value, float):
			text = f"{key}: {value:.2f}"
		else:
			text = f"{key}: {value}"
			
		# Opzionale: Aggiungi un contorno nero per migliorare la leggibilità
		cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
		
		# Disegna il testo vero e proprio
		if id == key:
			cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
		else:
			cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
		# Incrementa la Y per la riga successiva
		y += line_height
	return frame




def core():

	setting_idx = 0
	show_settings = False

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

		depth_map = tracker.new_estimate_depth_map(
			frame_l, frame_r,stereoSGBM_params, stereo_camera.stereo_params, True)
		
		#depth_map = tracker.estimate_depth_map(frame_l, frame_r)


		tracker.load_hands(frame_l)
		frame_tracked, hand_pos, is_real_press = tracker.process(frame_l)

		tracker.draw_landmark(frame_tracked)
		#tracker.draw_landmark(depth_map)

		depth_map_uint8 = (depth_map * 255).astype(np.uint8)
		depth_map_colored = cv2.applyColorMap(depth_map_uint8, color)
		wd_depth_Color.show_frame(depth_map_colored)

		if show_settings:
			frame_tracked = draw_vertical_dict(frame_tracked,stereoSGBM_params, stereoSGBM_params_steps[setting_idx][0])
		

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
		key = cv2.waitKeyEx(1) 
		if key != -1 :
			print(f"Hai premuto il tasto con codice: {key}")
		if key == ord(KEY_QUIT):
			break
		if key == ord(KEY_CHANGE_CAMERA):
			if  stereo:
				stereo_camera.change_cameras()
			else:
				mono_cameraL.change_camera()
				mono_cameraR.change_camera()
		if key == ord(KEY_CALIBRATION):
			tracker.calibrate_touch_plane(frame_r)
		if key == ord(KEY_RESET):
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

		if key == ord(KEY_SETTINGS): show_settings = not show_settings
		if show_settings:
			if key == KEY_DOWN: 
				setting_idx += 1
				if setting_idx == stereoSGBM_params_steps.__len__():
					setting_idx = 0
				print("Hai premuto SU")

			if key == KEY_UP:  
				setting_idx -= 1
				if setting_idx == -1:
					setting_idx = stereoSGBM_params_steps.__len__() - 1
				print("Hai premuto GIÙ")
				
			if key == KEY_LEFT: 
				add_to_param(stereoSGBM_params_steps[setting_idx])
				print("Hai premuto SINISTRA")
				
			if key == KEY_RIGTH:  
				sub_to_param(stereoSGBM_params_steps[setting_idx])
				print("Hai premuto DESTRA")
		
		
	
	stereo_camera.release()
	cv2.destroyAllWindows()