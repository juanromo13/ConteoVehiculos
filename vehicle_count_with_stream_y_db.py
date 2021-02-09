import numpy as np
import cv2 as cv2
import time
import psycopg2
from flask import Flask, Response

# Conexion DB
PSQL_HOST = ""
PSQL_PORT = ""
PSQL_USER = ""
PSQL_PASS = ""
PSQL_DB = ""

connection_address= """
host=%s port=%s user=%s password=%s dbname=%s
""" % (PSQL_HOST, PSQL_PORT, PSQL_USER, PSQL_PASS, PSQL_DB)
connection = psycopg2.connect(connection_address)

cursor = connection.cursor()

sql = "select tiempo from cruce where id='1';"
cursor.execute(sql)
rows = cursor.fetchall()
for row in rows:
    tiempo = row[0]

# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load video or camera
app = Flask(__name__)
video = cv2.VideoCapture(0)
# video = cv2.VideoCapture('Campanario-N-S-sab-4pm.mp4')

if not video.isOpened():
    print("No se pudo abrir el video.")
    exit()

# Counting function
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
    # Upload data to DB
    total_carros = cars + trucks
    sql = "UPDATE colas_reales set nortsouth=%s WHERE id='1';"
    val = str(total_carros)
    cursor.execute(sql,val)
    connection.commit()

# video streaming
@app.route('/')
def index():
    return "Bienvenido ve a /video_feed"

def gen(video):
    # tt = 20
    tt = tiempo
    while True:
        success, image = video.read()
        
        if tt >= tiempo: 
            contar_vehiculos(image)
            t0 = time.time()

        t1 = time.time()
        tt = t1 - t0
        
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, threaded=True)


