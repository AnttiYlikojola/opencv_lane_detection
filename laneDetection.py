#from picamera import PiCamera
from time import sleep

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if cap.isOpened() == False:# tarkastetaan onnistuiko videon avaaminen
    print ("VideoCapture failed")
    
width = None
height = None

region_of_interest_vertices = None

def region_of_interest(img, vertices):
	
	# maaritellaan tyhja matriisi joka tasmaa kuvan korkeuteen/leveyteen
	mask = np.zeros_like(img)
	
	# haetaan kuvan varikanavat
	channel_count = 1
	
	# luodaan tasmaava vari samoilla varikanavilla
	match_mask_color = (255,) * channel_count
	
	# taytetaan polygoni
	cv2.fillPoly(mask, vertices, match_mask_color)
		
	# palautetaan kuva jossa vain maskatut pikselit tasmaavat
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, vertices):
	# luodaan kaistan piirrolle musta kuva
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	# piirretaan kaistaviivat
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(line_img,(x1,y1), (x2, y2), [250, 250, 0], 5)
			
	# lisataan alkuperaiseen kuvaan kaistaviivat
	img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
	
	return img

while True:
	_, frame = cap.read()
        
	if not width or not height:
		width = frame.shape[1]
		height = frame.shape[0]
		region_of_interest_vertices = [(0, height),(width / 2, height / 2),(width, height)]
		print(width, height)
		
	gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	canny_image = cv2.Canny(gray_image,100,200)
	cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

	lines = cv2.HoughLinesP(
		cropped_image,
		rho=6,
		theta = np.pi/60,
		threshold=160,
		lines=np.array([]),
		minLineLength=40,
		maxLineGap=25
		)
		
	if lines is not None:
		# piirra viivat
		image_with_lines = draw_lines(frame, frame)
		cv2.imshow("frame", image_with_lines)
			    
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
		

	cv2.imshow("cropper", cropped_image)
	
cap.release()
cv2.destroyAllWindows()

