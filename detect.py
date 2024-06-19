import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
import argparse
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights","yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')

# Load CSV file mapping number plates to email addresses
def load_number_plate_emails(filename):
    number_plate_emails = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # Check if row has at least two elements
                number_plate_emails[row[0]] = row[1]
            else:
                print("Invalid row:", row)
    return number_plate_emails


number_plate_emails = load_number_plate_emails('number_plate_emails.csv')

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input video file")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["input"])  # Use the video file specified by the user
COLORS = [(0,255,0),(0,0,255)]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi,dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi/255.0
        return int(model.predict(helmet_roi)[0][0])
    except:
        pass

ret = True

while ret:
    ret, img = cap.read()
    if img is None:
        break
    
    img = imutils.resize(img,height=500)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            if classIds[i] == 0: # Bike
                helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
            else: # Number plate
                x_h = x-60
                y_h = y-350
                w_h = w+100
                h_h = h+100
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                if y_h>0 and x_h>0:
                    h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                    c = helmet_or_nohelmet(h_r)
                    cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                    cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)
                    
                    # OCR for number plate
                    number_plate = img[y_h:y_h+h_h, x_h:x_h+w_h]
                    gray_plate = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
                    gray_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    plate_text = pytesseract.image_to_string(gray_plate, config='--psm 8 --oem 3')
                    cv2.putText(img, "Number Plate: " + plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
                    # Check if number plate is in the list and send email
                    if plate_text in number_plate_emails:
                        receiver_email = number_plate_emails[plate_text]
                        sender_email = "asmath0809@gmail.com"
                        password = "jzfk gauh fgwe csnr"
                        subject = "Helmet Violation Detected"
                        body = "Dear Driver,\n\nYou have been detected without a helmet. Please ensure safety while driving."

                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = receiver_email
                        msg['Subject'] = subject
                        msg.attach(MIMEText(body, 'plain'))

                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login(sender_email, password)
                        text = msg.as_string()
                        server.sendmail(sender_email, receiver_email, text)
                        server.quit()

    writer.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
