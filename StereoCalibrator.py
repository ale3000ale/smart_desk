# stereo_calibration.py

import cv2
import numpy as np
import glob
import os
from collections import deque

class StereoCameraCalibrator:
	"""
	Calibra due telecamere stereo e salva i parametri.
	Procedura: scacchiera → foto → calibrazione → salva .npz
	"""
	
	def __init__(self, checkerboard_size=(8, 6), square_size_mm=20.0):
		"""
		Args:
			checkerboard_size: (cols, rows) di angoli interni (es 8×6 = 9×7 quadrati)
			square_size_mm: dimensione di un quadrato in mm
		"""
		self.checkerboard_size = checkerboard_size
		self.square_size_mm = square_size_mm
		
		# Criteri di convergenza per corner refinement
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		
		# Criteri stereo
		self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
		
		# 3D object points (0,0,0), (20,0,0), (40,0,0)...
		objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), dtype=np.float32)
		objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
		objp *= square_size_mm  # scala in mm
		self.objp = objp
		
		self.objpoints = []  # Punti 3D
		self.imgpoints_left = []   # Punti 2D left
		self.imgpoints_right = []  # Punti 2D right
		
	def capture_calibration_images(self, left_camera_idx=0, right_camera_idx=1, num_images=20):
		"""
		Fase 1: Cattura immagini di calibrazione da entrambe le camere.
		
		Procedura:
		1. Mostra la scacchiera alle camere in posizioni diverse
		2. Premi SPAZIO per catturare
		3. Premi ESC per finire
		4. Salva in "calibration_images/left" e "...right"
		"""
		os.makedirs("calibration_images/left", exist_ok=True)
		os.makedirs("calibration_images/right", exist_ok=True)
		
		cap_left = cv2.VideoCapture(left_camera_idx)
		cap_right = cv2.VideoCapture(right_camera_idx)
		
		if not cap_left.isOpened() or not cap_right.isOpened():
			print(f"Errore: impossibile aprire camere {left_camera_idx}, {right_camera_idx}")
			return
		
		Awidth = cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)
		Aheight = cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)
		Bwidth = cap_right.get(cv2.CAP_PROP_FRAME_WIDTH)
		Bheight = cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT)
		if Awidth * Aheight < Bwidth * Bheight:
			width = int(Awidth)
			height = int(Aheight)
			print("dimensioni di left")
		else:
			width = int(Bwidth)
			height = int(Bheight)
			print("dimensioni di rigth")
		print(f"W: {width}  H: {height}")
		# Imposta risoluzione identica
		cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		
		count = 0
		print("=" * 60)
		print("FASE 1: CATTURA IMMAGINI DI CALIBRAZIONE")
		print("=" * 60)
		print(f"Scacchiera: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} angoli interni")
		print(f"Quadrato: {self.square_size_mm} mm")
		print(f"Target: {num_images} immagini")
		print()
		print("Istruzioni:")
		print("1. Mostra la scacchiera alle ENTRAMBE camere")
		print("2. Posizionala in angoli diversi della visione")
		print("3. Premi SPAZIO per catturare")
		print("4. Premi ESC per finire")
		print()
		
		while count < num_images:
			ret_l, frame_l = cap_left.read()
			ret_r, frame_r = cap_right.read()
			
			if not (ret_l and ret_r):
				print("Errore nella lettura dei frame")
				break
			
			# Mostra preview
			h, w = frame_l.shape[:2]
			display_l = cv2.resize(frame_l, (w//2, h//2))
			display_r = cv2.resize(frame_r, (w//2, h//2))
			
			# Combina per visualizzare side-by-side
			combined = np.hstack([display_l, display_r])
			text = f"Immagini catturate: {count}/{num_images} - Premi SPAZIO per catturare, ESC per finire"
			cv2.putText(combined, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			
			cv2.imshow("Cattura Calibrazione (LEFT | RIGHT)", combined)
			
			key = cv2.waitKey(1) & 0xFF
			if key == 27:  # ESC
				print("Cattura interrotta")
				break
			elif key == 32:  # SPAZIO
				cv2.imwrite(f"calibration_images/left/left_{count:03d}.png", frame_l)
				cv2.imwrite(f"calibration_images/right/right_{count:03d}.png", frame_r)
				count += 1
				print(f"✓ Catturate {count} immagini")
		
		cap_left.release()
		cap_right.release()
		cv2.destroyAllWindows()
		print(f"Cattura completata: {count} immagini salvate")
		print()
	
	def detect_corners_in_images(self):
		print("=" * 60)
		print("FASE 2: RILEVAMENTO CORNER SULLA SCACCHIERA")
		print("=" * 60)
		
		images_left = sorted(glob.glob("calibration_images/left/*.png"))
		images_right = sorted(glob.glob("calibration_images/right/*.png"))
		
		if len(images_left) == 0 or len(images_right) == 0:
			print("Errore: nessuna immagine trovata.")
			return False
		
		print(f"Trovate {len(images_left)} immagini left e {len(images_right)} immagini right")
		print()
		
		valid_count = 0
		for img_left, img_right in zip(images_left, images_right):
			frame_l = cv2.imread(img_left)
			frame_r = cv2.imread(img_right)
			
			if frame_l is None or frame_r is None:
				print(f"✗ Errore lettura {os.path.basename(img_left)}")
				continue
			
			gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
			gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
			
			ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.checkerboard_size, None)
			ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.checkerboard_size, None)
			
			if ret_l and ret_r:
				# ✅ NUOVO: Validazione distanza media tra corner
				dist_l = np.mean(np.linalg.norm(np.diff(corners_l.reshape(-1, 2), axis=0), axis=1))
				dist_r = np.mean(np.linalg.norm(np.diff(corners_r.reshape(-1, 2), axis=0), axis=1))
				
				# Se distanze sono troppo diverse = incoerenza
				if abs(dist_l - dist_r) / max(dist_l, dist_r) > 0.3:  # >30% differenza
					print(f"✗ {os.path.basename(img_left)}: distanze corner incoerenti (dist_l={dist_l:.1f}, dist_r={dist_r:.1f})")
					continue
				
				# Affina corner
				corners_l = cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), self.criteria)
				corners_r = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), self.criteria)
				
				self.objpoints.append(self.objp)
				self.imgpoints_left.append(corners_l)
				self.imgpoints_right.append(corners_r)
				
				valid_count += 1
				print(f"✓ {os.path.basename(img_left)}: corner rilevati (dist_l={dist_l:.1f}, dist_r={dist_r:.1f})")
			else:
				print(f"✗ {os.path.basename(img_left)}: corner NON rilevati in una camera")
		
		print()
		print(f"Totale immagini valide: {valid_count}")
		if valid_count < 5:
			print("Errore: servono almeno 5 immagini valide.")
			return False
		
		return True
	
	def calibrate_individual_cameras(self, image_shape):
		"""
		Fase 3: Calibra ogni camera SINGOLARMENTE.
		
		Questo riduce lo spazio dei parametri e migliora convergenza.
		"""
		print("=" * 60)
		print("FASE 3: CALIBRAZIONE SINGOLE CAMERE")
		print("=" * 60)
		
		# Calibra LEFT
		print("Calibrazione camera LEFT...")
		ret_l, self.mtx_left, self.dist_left, rvecs_l, tvecs_l = cv2.calibrateCamera(
			self.objpoints, self.imgpoints_left, image_shape, None, None
		)
		
		error_l = 0
		for i in range(len(self.objpoints)):
			projected, _ = cv2.projectPoints(self.objpoints[i], rvecs_l[i], tvecs_l[i], 
											 self.mtx_left, self.dist_left)
			error_l += cv2.norm(self.imgpoints_left[i] - projected) / len(self.imgpoints_left[i])
		reprojection_error_l = error_l / len(self.objpoints)
		print(f"  Reprojection Error LEFT: {reprojection_error_l:.4f} px")
		
		# Calibra RIGHT
		print("Calibrazione camera RIGHT...")
		ret_r, self.mtx_right, self.dist_right, rvecs_r, tvecs_r = cv2.calibrateCamera(
			self.objpoints, self.imgpoints_right, image_shape, None, None
		)
		
		error_r = 0
		for i in range(len(self.objpoints)):
			projected, _ = cv2.projectPoints(self.objpoints[i], rvecs_r[i], tvecs_r[i], 
											 self.mtx_right, self.dist_right)
			error_r += cv2.norm(self.imgpoints_right[i] - projected) / len(self.imgpoints_right[i])
		reprojection_error_r = error_r / len(self.objpoints)
		print(f"  Reprojection Error RIGHT: {reprojection_error_r:.4f} px")
		print()
		
		# Obiettivo: reprojection error < 0.3 pixel (buono < 0.2)
		if reprojection_error_l > 0.5 or reprojection_error_r > 0.5:
			print("⚠ AVVISO: Errore alto, la qualità della calibrazione potrebbe essere scarsa")
			print("  Suggerimento: ricattura immagini con migliore qualità e angolazione")
		
		return ret_l and ret_r
	
	def calibrate_stereo(self, image_shape):
		"""
		Fase 4: Calibrazione STEREO con intrinseci fissati.
		
		Calcola R (rotazione) e T (traslazione) tra le due camere.
		"""
		print("=" * 60)
		print("FASE 4: CALIBRAZIONE STEREO (R, T)")
		print("=" * 60)
		
		flags = cv2.CALIB_FIX_INTRINSIC  # Fissa intrinseci calcolati nella Fase 3
		
		    # ✅ Criteri più severi
		criteria_stereo_strict = (
			cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
			200,      # max iter (era 100)
			1e-6      # epsilon (era 0.0001)
		)

		print("Calcolo R (rotazione) e T (traslazione)...")
		ret, self.mtx_left, self.dist_left, self.mtx_right, self.dist_right, \
			self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
			self.objpoints, self.imgpoints_left, self.imgpoints_right,
			self.mtx_left, self.dist_left,
			self.mtx_right, self.dist_right,
			image_shape, self.criteria_stereo, flags
		)
		stereo_error = ret  # ← USO DIRETTAMENTE IL VALORE RITORNATO
		# ✅ NUOVO: Validazione con baseline fisico
		baseline_fisico = 90.0  # ← CAMBIA CON LA TUA DISTANZA IN MM
		baseline_calcolato = np.linalg.norm(self.T)
		
		errore_percentuale = abs(baseline_calcolato - baseline_fisico) / baseline_fisico * 100
		
		print(f"Baseline fisico (misurato): {baseline_fisico:.1f} mm")
		print(f"Baseline calcolato: {baseline_calcolato:.1f} mm")
		print(f"Errore: {errore_percentuale:.1f}%")
		if errore_percentuale > 20:  # Se errore > 20%
			print("⚠ AVVISO: Baseline calcolato != fisico. Ricattura con migliore qualità!")
			return False
		
		# Validazione R
		det_R = np.linalg.det(self.R)
		print(f"R determinante: {det_R:.6f} (deve essere ~1.0)")
		
		if det_R < 0.95 or det_R > 1.05:
			print("⚠ R non è una matrice di rotazione valida!")
			return False
		
		print(f"Stereo Reprojection Error: {stereo_error:.4f} px")
		if stereo_error > 0.5:
			print("⚠ AVVISO: Errore stereo elevato.")
		elif stereo_error < 0.25:
			print("✓ Calibrazione eccellente!")
		
		return ret
		
	
	def save_calibration(self, filename="stereo_calib.npz"):
		"""
		Fase 5: Salva tutti i parametri su file .npz per uso runtime.
		"""
		print("=" * 60)
		print("FASE 5: SALVATAGGIO PARAMETRI")
		print("=" * 60)
		
		# Assicura che D siano vettori 1D
		D_left = self.dist_left.flatten()
		D_right = self.dist_right.flatten()
		T_saved = self.T.flatten() if self.T.ndim > 1 else self.T
		
		np.savez(filename,
				K_left=self.mtx_left,
				D_left=D_left,       # ← Flatten
				K_right=self.mtx_right,
				D_right=D_right,     # ← Flatten
				R=self.R,
				T=T_saved,
				E=self.E,
				F=self.F)
		
		print(f"✓ Parametri salvati in: {filename}")
		print()
		print("Contenuto del file:")
		print(f"  K_left:  Matrice intrinseca LEFT {self.mtx_left.shape}")
		print(f"  D_left:  Distorsione LEFT {self.dist_left.shape}")
		print(f"  K_right: Matrice intrinseca RIGHT {self.mtx_right.shape}")
		print(f"  D_right: Distorsione RIGHT {self.dist_right.shape}")
		print(f"  R:       Rotazione stereo {self.R.shape}")
		print(f"  T:       Traslazione stereo {self.T.shape}")
		print()
	
	def run_full_calibration(self, left_idx=0, right_idx=1, num_images=20):
		"""
		Esegui tutte le fasi di calibrazione in sequenza.
		"""
		print("\n")
		print("#" * 60)
		print("# CALIBRAZIONE STEREO COMPLETA")
		print("#" * 60)
		print()
		
		# Fase 1: Cattura
		self.capture_calibration_images(left_idx, right_idx, num_images)
		
		# Fase 2: Rilevamento corner
		if not self.detect_corners_in_images():
			print("Calibrazione fallita nella Fase 2")
			return False
		
		# Fase 3: Calibrazione singole camere
		# Leggi un'immagine per ottenere shape
		sample_img = cv2.imread(glob.glob("calibration_images/left/*.png")[0])
		image_shape = (sample_img.shape[1], sample_img.shape[0])  # (width, height)
		
		if not self.calibrate_individual_cameras(image_shape):
			print("Calibrazione fallita nella Fase 3")
			return False
		
		# Fase 4: Calibrazione stereo
		if not self.calibrate_stereo(image_shape):
			print("Calibrazione fallita nella Fase 4")
			return False
		
		# Fase 5: Salvataggio
		self.save_calibration("stereo_calib.npz")
		
		print("#" * 60)
		print("# CALIBRAZIONE COMPLETATA CON SUCCESSO!")
		print("#" * 60)
		print()
		return True


if __name__ == "__main__":
	# Configura la scacchiera
	CHECKERBOARD_SIZE = (10, 7)  # angoli interni (stampa 9×7 quadrati)
	SQUARE_SIZE_MM = 25.0  # Misura il tuo quadrato!

	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Prova diverse dimensioni
		for size in [(10,7),(9, 7),(8, 6), (7, 5), (6, 4)]:
			ret, corners = cv2.findChessboardCorners(gray, size, None)
			if ret:
				corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), 
										(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
				cv2.drawChessboardCorners(frame, size, corners, ret)
				cv2.putText(frame, f"TROVATO: {size}", (20, 40), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				break
		
		cv2.imshow("Debug Scacchiera", frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

	calibrator = StereoCameraCalibrator(CHECKERBOARD_SIZE, SQUARE_SIZE_MM)
	
	# Esegui calibrazione
	# Cambia indici se le tue camere sono su porte diverse
	calibrator.run_full_calibration(left_idx=0, right_idx=1, num_images=20)
	data = np.load("stereo_calib.npz")

	print("=== Contenuto stereo_calib.npz ===")
	for key in data.files:
		arr = data[key]
		print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
		print(f"  Campione: {arr.flat[:3]}")  # Primi 3 valori
