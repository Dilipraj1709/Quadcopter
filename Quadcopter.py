# import the necessary packages
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

        
        #begin mission
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                imgOriginal = frame.array
                status = "No Targets"
                center = "N/A"
                Deltas = "N/A"
                pathcorrection = "N/A"
                                       #make gray
                gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

                                #blur
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)

                                #edges
                edged = cv2.Canny(blurred, 50, 150)

                        # find contours in the edge map
                (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE)

                        # loop over the contours
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
                                                        mode = "Search Mode"



                cv2.putText(imgOriginal, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                cv2.putText(imgOriginal, center, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                cv2.putText(imgOriginal, Deltas, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                cv2.putText(imgOriginal, pathcorrection, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                cv2.putText(imgOriginal, mode, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                                # show the frame and record if a key is pressed
                cv2.rectangle(imgOriginal,(290,200),(350,260),(0,0,225),2)
                cv2.imshow("Frame", imgOriginal)
                key = cv2.waitKey(1) & 0xFF
                rawCapture.truncate(0)
        cv2.destroyAllWindows()

        return

if __name__ == "__main__":
        main()
