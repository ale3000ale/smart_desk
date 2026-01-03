# pkg/camera/StereoCamera.py

import cv2
import numpy as np
from pkg.config import *
from pkg.window.Window import Window
from threading import Thread, Lock
import time

import pkg.config as const 
import cv2

class StereoCamera:
	"""
	Gestisce due videocamere in stereo (left e right).
	Fornisce frame sincronizzati e supporta la calibrazione stereo.
	"""

	def __init__(self, left_index=0, right_index=1):
		"""
		Inizializza il sistema stereo con due camere.
		
		Args:
			left_index: indice della webcam sinistra (default: 0)
			right_index: indice della webcam destra (default: 1)
			width: larghezza frame
			height: altezza frame
		"""
		self.left_index = left_index
		self.right_index = right_index
		

		# Apri le due videocamere
		self.cap_left = cv2.VideoCapture(self.left_index )
		self.cap_right = cv2.VideoCapture(self.right_index)

		if not self.cap_left.isOpened() or not self.cap_right.isOpened():
			raise RuntimeError(f"Impossibile aprire le videocamere {left_index} e/o {right_index}")
		
		self.width , self.height = self.select_dimensions()
		print(f"W: {self.width}  H: {self.height}")

		self.set_dimensions( self.width, self.height)
	
		
		# Variabili per sincronizzazione thread
		self.lock = Lock()
		self.frame_left = None
		self.frame_right = None
		self.frame_left_ready = False
		self.frame_right_ready = False
		
		# Thread di lettura asincrona (opzionale, per sincronia migliore)
		self.reading = True
		
		self.thread_left = Thread(target=self._read_left_thread, daemon=True)
		self.thread_right = Thread(target=self._read_right_thread, daemon=True)
		self.thread_left.start()
		self.thread_right.start()

		# Parametri calibrazione stereo (verranno caricati da file)
		self.stereo_params = None
		self.map_left = None
		self.map_right = None

		self.wd = Window("Stereo Camera L | R",self.width * 2, self.height)


		print(f"StereoCamera inizializzata: {self.width}x{self.height} (Left: {left_index}, Right: {right_index})")

	def select_dimensions(self):
		Awidth = self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)
		Aheight = self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)
		Bwidth = self.cap_right.get(cv2.CAP_PROP_FRAME_WIDTH)
		Bheight = self.cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT)
		if Awidth * Aheight < Bwidth * Bheight:
			return int(Awidth), int(Aheight)
		else:
			return int(Bwidth), int(Bheight)

		

	def _set_camera_properties(self, cap, width, height):
		"""Imposta proprietà della videocamera."""
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		#cap.set(cv2.CAP_PROP_FPS, CAMERADEFAULTFPS)

	def _read_left_thread(self):
		"""Thread di lettura per camera sinistra."""
		while self.reading:
			ret, frame = self.cap_left.read()
			if ret:
				with self.lock:
					self.frame_left = frame
					self.frame_left_ready = True

	def _read_right_thread(self):
		"""Thread di lettura per camera destra."""
		while self.reading:
			ret, frame = self.cap_right.read()
			if ret:
				with self.lock:
					self.frame_right = frame
					self.frame_right_ready = True

	def read(self):
		"""
		Legge un frame sincronizzato da entrambe le camere.
		
		Returns:
			(ret_left, ret_right, frame_left, frame_right): tuple (bool, bool, ndarray, ndarray)
				- ret_left, ret_right: True se il frame è valido
				- frame_left, frame_right: frame dalle due camere
		"""

		# Leggi dai thread
		with self.lock:
			if self.frame_left_ready and self.frame_right_ready:
				fl = self.frame_left.copy()
				fr = self.frame_right.copy()
				self.frame_left_ready = False
				self.frame_right_ready = False
				self.wd.show_frame(np.hstack([fl, fr]))
				return True, True, fl, fr
		return False, False, None, None


	def get_dimensions(self):
		"""Ritorna (width, height)."""
		return self.width, self.height

	def set_dimensions(self, width, height):
		"""Imposta nuove dimensioni per entrambe le camere."""
		self._set_camera_properties(self.cap_left, width, height)
		self._set_camera_properties(self.cap_right, width, height)
		self.width = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print(f"Dimensioni stereo impostate: {self.width}x{self.height}")

	def load_stereo_calibration(self, calibration_file):
		"""
		Carica i parametri di calibrazione stereo da file .npz.
		
		Args:
			calibration_file: path al file .npz (es. "stereo_calib.npz")
		
		Crea le mappe di rettifica (self.map_left, self.map_right).
		"""
		try:
			data = np.load(calibration_file)
    
			K_left = data['K_left'].astype(np.float64)
			D_left = data['D_left'].astype(np.float64).flatten()  # ← ADD
			K_right = data['K_right'].astype(np.float64)
			D_right = data['D_right'].astype(np.float64).flatten()  # ← ADD
			R = data['R'].astype(np.float64)
			T = data['T'].astype(np.float64)
			
			if T.ndim == 1:
				T = T.reshape(3, 1)
			
			# Calcola le rettifiche
			R1, R2, P1, P2, Q, validRoiL, validRoiR = cv2.stereoRectify(
				K_left, D_left,
				K_right, D_right,
				(self.width, self.height),
				R, T,
				flags=cv2.CALIB_ZERO_DISPARITY,
				alpha=0.0
			)
			
			# Crea le mappe di remap
			self.map_left = cv2.initUndistortRectifyMap(
				K_left, D_left, R1, P1,
				(self.width, self.height),
				cv2.CV_32F
			)
			self.map_right = cv2.initUndistortRectifyMap(
				K_right, D_right, R2, P2,
				(self.width, self.height),
				cv2.CV_32F
			)
			
			# Salva i parametri per usi futuri (Q matrix per triangolazione)
			self.stereo_params = {
				'K_left': K_left,
				'K_right': K_right,
				'R1': R1,
				'R2': R2,
				'P1': P1,
				'P2': P2,
				'Q': Q,
				'validRoiL': validRoiL,
				'validRoiR': validRoiR
			}
			
			print(f"Calibrazione stereo caricata da: {calibration_file}")
			return True
			
		except FileNotFoundError:
			print(f"File di calibrazione non trovato: {calibration_file}")
			return False
		except Exception as e:
			print(f"Errore nel caricamento calibrazione: {e}")
			return False

	def rectify_frames(self, frame_left, frame_right):
		"""
		Applica la rettifica stereo ai frame.
		
		Args:
			frame_left, frame_right: frame grezzi
			
		Returns:
			(frame_left_rect, frame_right_rect): frame rettificati
			
		Nota: richiede load_stereo_calibration() chiamato prima.
		"""
		if self.map_left is None or self.map_right is None:
			print("Attenzione: mappe di rettifica non caricate. Ritorno frame grezzi.")
			return frame_left, frame_right

		map_left_x, map_left_y = self.map_left
		map_right_x, map_right_y = self.map_right

		frame_left_rect = cv2.remap(frame_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
		frame_right_rect = cv2.remap(frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

		return frame_left_rect, frame_right_rect

	def release(self):
		"""Rilascia le videocamere."""
		self.reading = False
		self.thread_left.join(timeout=1.0)
		self.thread_right.join(timeout=1.0)
		self.cap_left.release()
		self.cap_right.release()
		print("StereoCamera rilasciata")


	def scout_cameras(self):						
		cameras_info = []
				
		# Disabilita i log durante la ricerca
		old_log_level = cv2.getLogLevel()
	
		cv2.setLogLevel(0)

		for index in range(10):  # Verifica i primi 10 indici
			cap = cv2.VideoCapture(index)
			
			if cap.isOpened():
				width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				fps = cap.get(cv2.CAP_PROP_FPS)
				
				camera_data = {
					'index': index,
					'width': width,
					'height': height,
					'fps': fps
				}
				
				cameras_info.append(camera_data)
				cap.release()
			else:
				cap.release()
		# Ripristina il livello di log
		cv2.setLogLevel(old_log_level)
		return cameras_info
	
	def change_one_camera(self, cap):
		"""Ritorna informazioni dettagliate su tutte le videocamere disponibili"""
		cameras = self.scout_cameras()
		self.release()
		for cam_info in cameras:
			print(f"Camera {cam_info['index']}: {cam_info['width']}x{cam_info['height']} @ {cam_info['fps']} FPS")
		choice = input("\nScegli la videocamera da usare: ")

		while choice.isdigit() is False or int(choice) < 0 or int(choice) >= len(cameras):
			choice = input("Scelta non valida. Scegli la videocamera da usare: ")

		self.release()
		index = int(choice)
		cap = cv2.VideoCapture(index)
		if cap.isOpened():
			return index
		else:
			print("Impossibile aprire la videocamera scelta. Ritrono alla videocamera predefinita (0).")
			return -1
		

	def change_cameras(self):
		self.release()
		new_id_l = self.change_one_camera(self.cap_left)
		new_id_r = self.change_one_camera(self.cap_right)
		if new_id_l != -1  and new_id_r != -1:
			self.left_index = new_id_l
			self.right_index = new_id_r
			print("Videocamere cambiate")
		else:
			print("Errore nel cambio di una videocamera torno alle precedenti")
			self.cap_left = cv2.VideoCapture(self.left_index )
			self.cap_right = cv2.VideoCapture(self.right_index)

		self.width , self.height = self.select_dimensions()
		print(f"W: {self.width} X H: {self.height}")

		self.set_dimensions(self, self.width, self.height)
		self.wd.change_dimension(self.width , self.height )

		self.thread_left = Thread(target=self._read_left_thread, daemon=True)
		self.thread_right = Thread(target=self._read_right_thread, daemon=True)
		

	def __del__(self):
		"""Distruttore."""
		if hasattr(self, 'cap_left') and self.cap_left is not None:
			self.release()
