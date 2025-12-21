
import pkg.config as const 
from pkg import cv2

class Camera:
	def __init__(self):
		self.index = const.CAMERA_DEFAULT_INDEX
		self.cap = cv2.VideoCapture(self.index)
		if not self.cap.isOpened():
			raise RuntimeError(f"Non è possibile aprire la videocamera {self.index}")
		""" Ottieni dimensioni iniziali attenzione get restituisce float va convertito a int"""
		self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	def get_dimensions(self):
		return self.width, self.height
	
	"""Legge un frame dalla videocamera"""
	def read(self):							
		ret, frame = self.cap.read()
		# ret booleano che indica se la lettura è avvenuta con successo
		# frame è il frame catturato
		return ret, frame			

	"""Rilascia la videocamera  """
	def release(self):								 
		self.cap.release()



	"""	Ritorna una lista di dizionari con informazioni sulle videocamere disponibili"""
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
	
	def change_camera(self):
		"""Ritorna informazioni dettagliate su tutte le videocamere disponibili"""
		cameras = self.scout_cameras()
		for cam_info in cameras:
			print(f"Camera {cam_info['index']}: {cam_info['width']}x{cam_info['height']} @ {cam_info['fps']} FPS")
		choice = input("\nScegli la videocamera da usare: ")

		while choice.isdigit() is False or int(choice) < 0 or int(choice) >= len(cameras):
			choice = input("Scelta non valida. Scegli la videocamera da usare: ")

		self.release()
		self.index = int(choice)
		self.cap = cv2.VideoCapture(self.index)
		if self.cap.isOpened():
			self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		else:
			self.index = 0
			self.release()
			self.cap = cv2.VideoCapture(self.index)
			print("Impossibile aprire la videocamera scelta. Ritrono alla videocamera predefinita (0).")


	def __del__(self):
		self.release()