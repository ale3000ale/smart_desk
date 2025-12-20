from window.window import Window
from camera.Camera import Camera
import cv2

if __name__ == "__main__":

	wd = Window("Camera Viewer")
	camera = Camera()
	
	while True:
		ret, frame = camera.read() 
		if ret:
			wd.show_frame(frame)
		else:
			print("Errore nella lettura del frame")
			break
		
		""" Controlla per uscire o cambiare camera """
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if cv2.waitKey(1) & 0xFF == ord('c'):
				camera.change_camera()
		
	
	camera.release()
	cv2.destroyAllWindows()