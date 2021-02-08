# files
1. Campanario---.mp4 (test videos)
2. coco.names (list of object names that the algorithm can recognize)
3. requeriments.txt (requeriments python requirements)
4. yolov3-tiny.cfg (yolo configuration file)
5. yolov3-tiny.weights (yolo pre-trained weight file)
 
# How to run:
python code for count vehicles.

1. install requeriments, run "pip install -r requirements.txt"
2. download ngrok
3. run ngrok "./ngrok http 9999"
4. conteo_vehiculos_con_stream_y_db.py use a postgres db, you need configure this file with your db.

# Note:
if you want a better recognition, you can download another weights and cfg from https://pjreddie.com/darknet/yolo/ then you must change the code in this line:

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

Example if you gonna use YOLOv3-320:

net = cv2.dnn.readNet("yolov3-320.weights", "yolov3-320.cfg")
