import numpy as np
import cv2 as cv2
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load video or camera
cap = cv2.VideoCapture('Campanario-N-S-sab-4pm.mp4')
#cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

# Funcion conteo
def contar_vehiculos(img):
    cars = trucks = 0
    # Loading image
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1/225, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img, label, (x, y -10), font, 0.6, (0, 0, 0), 1)
            #conteo carros
            if "car" == label:
                cars += 1
            if "truck" == label:
                trucks += 1
    print(f"hay {cars} carros y {trucks} camiones")

tt = 20
# Abrir video
while cap.isOpened():

    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    #cada 20 seg cuenta vehiculos
    if tt >= 20:
        contar_vehiculos(frame)
        t0 = time.time()

    t1 = time.time()
    tt = t1 - t0

cap.release()
cv2.destroyAllWindows()

