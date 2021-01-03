# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:38:26 2020

@author: Nikhil
"""

import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time

parser=argparse.ArgumentParser()
parser.add_argument('--connect',default='127.0.0.1:14550')
args=parser.parse_args()
vehicle=connect(args.connect, baud=57600,wait_ready=True )

def arm_and_takeoff(aTargetAltitude):

  print ("Basic pre-arm checks")
  # Don't let the user try to arm until autopilot is ready
  while not vehicle.is_armable:
    print (" Waiting for vehicle to initialise...")
    time.sleep(1)
        
  print ("Arming motors")
  # Copter should arm in GUIDED mode
  vehicle.mode    = VehicleMode("GUIDED")
  vehicle.armed   = True

  while not vehicle.armed:
    print (" Waiting for arming...")
    time.sleep(1)

  print "Taking off!"
  arm_and_takeoff(1) # Take off to target altitude
  vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).
  while True:
      print (" Altitude: ", vehicle.location.global_relative_frame.alt)
      #Break and return from function just below target altitude.
      if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
          print ("Reached target altitude")
          break
      time.sleep(1)

cap = cv2.VideoCapture(0)

#setting the frame size
cap.set(3, 320) #setting the horizontal frame size 
cap.set(4, 240) #setting the vertical frame size

hasFrame, frame = cap.read()
  
classes = None
with open("coco.names", "r") as f:
 classes = [line.strip() for line in f.readlines()]
 
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')           

vehicle.simple_takeoff(aTargetAltitude)

while cv2.waitKey(1) < 0:
    sensor_left = "give the left sensor values here with a delay"
    sensor_right = "give the right sensor values here with a delay"
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    Width = frame.shape[1]
    Height = frame.shape[0]
    class_ids = []
    confidences = []
    boxes = []
    net.setInput(cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    for i in indices:
        i = i[0]
        box = boxes[i]
        if class_ids[i]==0:
            label = str(classes[class_id]) 
            cv2.rectangle(frame, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
            cv2.putText(frame, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print("x = ",((box[0])+(box[2])/2))
            print("y = ",((box[1])+(box[3])/2))
            if sensor_right < 50 and sensor_left < 50:
                V_x = 0 #m/s
                V_y = 0 #m/s
                V_z = 0 #m/s
            if sensor_right < 50:
                V_x = 0 #m/s
                V_y = -0.5 #m/s
                V_z = 0 #m/s
            if sensor_left < 50:
                V_x = 0 #m/s
                V_y = 0.5 #m/s
                V_z = 0 #m/s
            if ((box[0])+(box[2])/2) > 200 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) > 180:
                V_x = 0 #m/s
                V_y = 0.5 #m/s
                V_z = 0 #m/s
            elif ((box[0])+(box[2])/2) > 200 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) < 180:
                V_x = 0.5 #m/s
                V_y = 0.5 #m/s
                V_z = 0 #m/s
            elif ((box[0])+(box[2])/2) < 120 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) > 180:
                V_x = 0 #m/s
                V_y = -0.5 #m/s
                V_z = 0 #m/s
            elif ((box[0])+(box[2])/2) > 120 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) < 180:
                V_x = 0.5 #m/s
                V_y = -0.5 #m/s
                V_z = 0 #m/s
            elif (round(box[1]+box[3])) < 180:
                V_x = 0.4 #m/s
                V_y = 0 #m/s
                V_z = 0 #m/s
            else:
                V_x = 0 #m/s
                V_y = 0 #m/s
                V_z = 0 #m/s
    cv2.imshow("output",frame)
    
    
    
    
    
    
    
