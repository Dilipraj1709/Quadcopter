import argparse
import cv2
import numpy as np
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from dronekit import *
import droneapi
import gps
import socket
import sys


#connects to the quadcopter
def connect2copter():
    parser = argparse.ArgumentParser(description='Print out vehicle state information. Connects to SITL on local PC by default.')
    parser.add_argument('--connect', default='115200', help="vehicle connection target. Default '57600'")
    args = parser.parse_args()
    vehicle = connect('/dev/serial/by-id/usb-3D_Robotics_PX4_FMU_v2.x_0-if00', baud = 115200, rate=6)
    return vehicle

#wait until the vehicle is in guided mode
def wait4guided(vehicle):
    while (vehicle.mode.name != 'GUIDED'):
        time.sleep(2)
    return

#arm the vehicle if armable
def checkArm(vehicle):
    #uncomment for real world test
   while not vehicle.is_armable:
#       print "waiting for vehicle to become armable"
        time.sleep(1)
 #  print "Contact!"
    vehicle.armed = True
    while not vehicle.armed:      
 ##       print "Waiting for arming..."
        time.sleep(1)
    return

#takeoff
def TakeoffandClimb (vehicle, alt):
##    print "rotate!"
    vehicle.simple_takeoff(alt)
    while not (vehicle.location.global_frame.alt>=alt*0.95):
##        print " Altitude: ", vehicle.location.global_frame.alt
        time.sleep(1)
    return

#initialize the camera
def initcam():
        #initialize camera
        camera = PiCamera()
        camera.resolution=(640,480)
        camera.framerate=50
        rawCapture=PiRGBArray(camera, size=(640,480))
        time.sleep(0.1)
        return camera, rawCapture


def main():
        #variable init
        RX = 0
        RY = 0
        mode = "N/A"
        alt = 1.5 #altitude (meters)

        #connect to copter
        vehicle = connect2copter()

        #init camera
        (camera, rawCapture) = initcam()

        #wait 4 guided mode
        wait4guided(vehicle)

        #check armable and arm
        checkArm(vehicle)

        #Takeoff
        TakeoffandClimb(vehicle, alt)
	
	# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
	for c in cnts:
                # approximate the contour
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                                        # ensure that the approximated contour is "roughly" rectangular
                        if len(approx) >= 4 and len(approx) <= 6:
                        # compute the bounding box of the approximated contour and
                        # use the bounding box to compute the aspect ratio
                                (x, y, w, h) = cv2.boundingRect(approx)
                                aspectRatio = w / float(h)
                                # compute the solidity of the original contour
                                area = cv2.contourArea(c)
                                hullArea = cv2.contourArea(cv2.convexHull(c))
                                solidity = area / float(hullArea)
                                # compute whether or not the width and height, solidity, and
                                # aspect ratio of the contour falls within appropriate bounds
                                keepDims = w > 25 and h > 25
                                keepSolidity = solidity > 0.9
                                keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2
                                # ensure that the contour passes all our tests
                                if keepDims and keepSolidity and keepAspectRatio:
                                        # draw an outline around the target and update the status
                                        # text
                                        cv2.drawContours(imgOriginal, [approx], -1, (0, 0, 255), 4)
                                        status = "Target(s) Acquired"
                                        
                                        # compute the center of the contour region and draw the
                                        # crosshairs
                                        M = cv2.moments(approx)
                                        (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                                        (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
                                        (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
                                        cv2.line(imgOriginal, (startX, cY), (endX, cY), (0, 0, 255), 3)
                                        cv2.line(imgOriginal, (cX, startY), (cX, endY), (0, 0, 255), 3)
                                        (DX,DY) = (cX-320,cY-230)
                                        (AX,AY) = (DX*(.0016*alt+.00003), DY*(.0016*alt+.00003))
                                        if not (RY>-60 and (-60<RX<60)):
                                                #fly to mode
                                                if (DY<-30):#check this it could be backwards
                                                        center = "Center Coordinates: ({centerx}, {centery})".format(centerx=str(cX),centery=str(cY))
                                                        Deltas = "Center Deltas: ({deltaX}, {deltaY})".format(deltaX=str(DX),deltaY=str(DY))
                                                        # draw the status text on the frame
                                                        if DX>0:
                                                            if DY>0: #bottom right of screen
                                                                pathcorrection = "Move Right {magdelx} and Backward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                            else: #top right of screen
                                                              pathcorrection = "Move Right {magdelx} and Forward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                        else:
                                                            if DY>0: #bottom left of screen
                                                                pathcorrection = "Move Left {magdelx} and Backward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                            else: #top left of screen
                                                                 pathcorrection = "Move Left {magdelx} and Forward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                        RX = DX
                                                        RY = DY
                                                        mode = "Fly to Mode"

                                        elif RY>-60:
                                                #Search/hover mode
                                                if (DX<-30 and DY<-30) or (DX<-30 and DY>30) or (DX>30 and DY<-30) or (DX>30 and DY>30) or (DX<30 and DX>-30 and DY>30) or (DX<30 and DX>-30 and DY<-30) or (DY<30 and DY>-30 and DX>30) or (DY<30 and DY>-30 and DX<-30):
                                                        center = "Center Coordinates: ({centerx}, {centery})".format(centerx=str(cX),centery=str(cY))
                                                        Deltas = "Center Deltas: ({deltaX}, {deltaY})".format(deltaX=str(DX),deltaY=str(DY))
                                                        # draw the status text on the frame
                                                        if DX>0:
                                                            if DY>0: #bottom right of screen
                                                                pathcorrection = "Move Right {magdelx} and Backward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                            else: #top right of screen
                                                              pathcorrection = "Move Right {magdelx} and Forward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                        else:
                                                            if DY>0: #bottom left of screen
                                                                pathcorrection = "Move Left {magdelx} and Backward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                            else: #top left of screen
                                                                 pathcorrection = "Move Left {magdelx} and Forward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                        RX = DX
                                                        RY = DY
                                                        mode = "Search Mode"
                                        else:
                                              #Search/hover mode
                                                if (DX<-30 and DY<-30) or (DX<-30 and DY>30) or (DX>30 and DY<-30) or (DX>30 and DY>30) or (DX<30 and DX>-30 and DY>30) or (DX<30 and DX>-30 and DY<-30) or (DY<30 and DY>-30 and DX>30) or (DY<30 and DY>-30 and DX<-30):
                                                        center = "Center Coordinates: ({centerx}, {centery})".format(centerx=str(cX),centery=str(cY))
                                                        Deltas = "Center Deltas: ({deltaX}, {deltaY})".format(deltaX=str(DX),deltaY=str(DY))
                                                        # draw the status text on the frame
                                                        if DX>0:
                                                            if DY>0: #bottom right of screen
                                                                pathcorrection = "Move Right {magdelx} and Backward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                            else: #top right of screen
                                                              pathcorrection = "Move Right {magdelx} and Forward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                        else:
                                                            if DY>0: #bottom left of screen
                                                                pathcorrection = "Move Left {magdelx} and Backward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                            else: #top left of screen
                                                                 pathcorrection = "Move Left {magdelx} and Forward {magdelY}".format(magdelx=str(abs(AX)),magdelY=str(abs(AY)))
                                                        RX = DX
                                                        RY = DY

	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()