"""
	Core crea e gestisce ML , elaboratore, GUI 
	ML crea eventi in base a ció che la camera vede
	|-->Elaboratore elabora gli eventi ed i dati provenienti da ML e dice alla GUI cosa mostrare
			|-->GUI si occupa solo del rendering 
"""
from pkg import * 

def core():
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