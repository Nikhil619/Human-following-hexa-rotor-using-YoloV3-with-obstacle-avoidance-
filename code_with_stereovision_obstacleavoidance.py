import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from matplotlib import pyplot as plt
import StereoVisionLibrary as svl
import ObjectDetectionLibrary as odl 
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--connect',default='127.0.0.1:14550')
args=parser.parse_args()
vehicle=connect(args.connect, baud=57600,wait_ready=True )

def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 

    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

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

  print ("Taking off!")
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

#front cams
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

#leftside cams
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(4)

#rightside cams
cap5 = cv2.VideoCapture(5)
cap6 = cv2.VideoCapture(6)


#setting the frame size
cap.set(3, 320) #setting the horizontal frame size 
cap.set(4, 240) #setting the vertical frame size

hasFrame, frame = cap.read()
  
classes = None
with open("coco.names", "r") as f:
 classes = [line.strip() for line in f.readlines()]
 
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')           

arm_and_takeoff(1)

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    hasFrame, frame1 = cap1.read()
    hasFrame, frame2 = cap2.read()
    hasFrame, frame3 = cap3.read()
    hasFrame, frame4 = cap4.read()
    hasFrame, frame5 = cap5.read()
    hasFrame, frame6 = cap6.read()

    frameCopy = np.copy(frame)
    IMG_left1 = frame1
    IMG_right1 = frame2
    IMG_left2 = frame3
    IMG_right2 = frame4
    IMG_left3 = frame5
    IMG_right3 = frame6

    
    #Finding disparity map
    disparity1 = svl.compute_disparity(IMG_left1, IMG_right1)
    disparity2 = svl.compute_disparity(IMG_left2, IMG_right2)
    disparity3 = svl.compute_disparity(IMG_left3, IMG_right3)
    
    #object detection without using disparity
    IMG_detected_11 = odl.find_object_rect(IMG_left1, IMG_right2)
    IMG_detected_12 = odl.find_object_rect(IMG_left2, IMG_right2)
    IMG_detected_13 = odl.find_object_rect(IMG_left3, IMG_right3)
    
    #object detection using disparity
    IMG_detected_21 = odl.find_object_mask(IMG_left1, IMG_right1, disparity1)
    IMG_detected_22 = odl.find_object_mask(IMG_left2, IMG_right2, disparity2)
    IMG_detected_23 = odl.find_object_mask(IMG_left3, IMG_right3, disparity3)
    
    im21 = np.copy(IMG_detected_21)
    im22 = np.copy(IMG_detected_22)
    im23 = np.copy(IMG_detected_23)
    
    #depth/distance of object
    distance1 = svl.compute_depth(im21, disparity1)
    distance2 = svl.compute_depth(im22, disparity2)
    distance3 = svl.compute_depth(im23, disparity3)
    
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
            if distance1 < 50:
                V_x = 0 #m/s
                V_y = 0 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            if distance3 < 50:
                V_x = 0 #m/s
                V_y = -0.5 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            if distance2 < 50:
                V_x = 0 #m/s
                V_y = 0.5 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            if ((box[0])+(box[2])/2) > 200 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) > 180:
                V_x = 0 #m/s
                V_y = 0.5 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            elif ((box[0])+(box[2])/2) > 200 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) < 180:
                V_x = 0.5 #m/s
                V_y = 0.5 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            elif ((box[0])+(box[2])/2) < 120 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) > 180:
                V_x = 0 #m/s
                V_y = -0.5 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            elif ((box[0])+(box[2])/2) > 120 and ((box[1])+(box[3])/2) < 160 and ((box[1])+(box[3])/2) > 80 and (round(box[1]+box[3])) < 180:
                V_x = 0.5 #m/s
                V_y = -0.5 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            elif (round(box[1]+box[3])) < 180:
                V_x = 0.4 #m/s
                V_y = 0 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
            else:
                V_x = 0 #m/s
                V_y = 0 #m/s
                V_z = 0 #m/s
                send_ned_velocity(V_x, V_y, V_z, 1)
    cv2.imshow("output",frame)

