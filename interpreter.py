import cv2
import numpy as np

#Load YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layers_names = net.getUnconnectedOutLayersNames()


#Open a connection to the camera (0 represents the default camera, change it if you have multiple cameras)
cap = cv2.VideoCapture(0)

#check if the camera is open
if not cap.isOpened():
    print('Error:could not open the camera..')
    exit()
    
while True:
    #read the framec;
    ret, frame = cap.read()
    
    #Prepare the frame for YOLO Input
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (220,220),(0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layers_names)
    
    #Get hand detection results
    class_ids = []
    confidences = []
    boxes = []
   
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 0:
                #Object detected
                center_x = int(detection[0]*frame.shape[1])
                center_y = int(detection[1]*frame.shape[0])
                w = int(detection[2]*frame.shape[1])
                h = int(detection[3]*frame.shape[0])
                
                #Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    #Non-maximum supression to remove redundant bounding boxes 
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  
    
    #draw bounding boxes on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                     
            
            
    
    #check if the frame is read successfully
    if not ret:
        print('Error:could not read the frame..')
        break
    
    #display the captured frame
    cv2.imshow('camera feed', frame)
    
    #break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()