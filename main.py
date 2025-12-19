import window.window as window
import cv2
#import mediapipe as mp
#import pyautogui as pyGui

if __name__ == "__main__":

	wd = window.Window("Test", 800, 600)
	cap = cv2.VideoCapture(0)
	

	while True:
		ret, frame = cap.read()
    
		if ret:
			wd.show_frame(frame)
			# Esci con 'q'
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	# 4. Libera risorse
	cap.release()
	cv2.destroyAllWindows()