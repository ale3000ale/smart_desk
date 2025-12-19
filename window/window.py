import cv2

class Window:
	def __init__(self, title="Window", width=640, height=480):
		self.title = title
		self.width = width
		self.height = height
		self._create_window()

	def _create_window(self):
		cv2.namedWindow(self.title)
		cv2.resizeWindow(self.title, self.width, self.height)

	def show_frame(self, frame):
		cv2.imshow(self.title, frame)

	def wait_key(self, delay=1):
		return cv2.waitKey(delay)

	def destroy(self):
		cv2.destroyWindow(self.title)