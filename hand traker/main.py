import cv2
import mediapipe as mp
import pyautogui as pyGui

hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	outputs = hand_detector.process(rgb_frame)
	hands = outputs.multi_hand_landmarks
	screen_width, screen_height = pyGui.size()

	if hands:
		print(hands.length)
		for hand in hands:
			#drawing_utils.draw_landmarks(frame, hand)
			landmarks = hand.landmark
			index = [0,0]
			thumb = [0,0]
			
			for id, lm in enumerate(landmarks):
				x = int(lm.x * frame.shape[1])
				y = int(lm.y * frame.shape[0])
				if id == 8:  # Index finger tip
					index = [x, y]
					cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=cv2.FILLED)
					pyGui.moveTo(screen_width / frame.shape[1] * x, screen_height / frame.shape[0] * y)
				if id == 4:  # Index finger tip
					thumb = [x, y]
					cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=cv2.FILLED)
					#pyGui.moveTo(screen_width / frame.shape[1] * x, screen_height / frame.shape[0] * y)
				
			distance = ((index[0]-thumb[0])**2 + (index[1]-thumb[1])**2 )**0.5
			print(distance)
			if distance < 40:
				pyGui.click()
				#cv2.circle(img=frame, center=((index[0]+thumb[0])//2, (index[1]+thumb[1])//2), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
	if not ret:
		break
	#cv2.imshow('Normal Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
