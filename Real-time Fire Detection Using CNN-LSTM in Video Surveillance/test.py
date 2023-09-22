import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import cv2
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import smtplib
import os
import sys
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from PIL import ImageGrab
lb=['Forest fire', 'Match stick','Stove']
# Load the saved model
model = load_model('./new_model.h5')
#model.summary()
#lb = pickle.loads(open('./new_model_leaves.pkl', "rb").read())
runOnce = False # created boolean
def send_mail_function(): # defined function to send mail post fire detection using threading
    
    sender_email = ''
    sender_password = ''
    recipient_email = ''
    subject = 'Fire Alert'
    message = 'ALERT: Fire Detected!\n'
    'location: https://goo.gl/maps/4gxSTwBMWnmD2zAJ9'

    # Take a screenshot of the current screen
    screenshot = ImageGrab.grab()

    # Save the screenshot as a temporary file
    screenshot_path = 'screenshot.png'
    screenshot.save(screenshot_path)

    # Create a multipart message object and add the email body
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message))

    # Attach the screenshot to the email as an image
    with open(screenshot_path, 'rb') as f:
        screenshot_data = f.read()
        screenshot_image = MIMEImage(screenshot_data)
        screenshot_image.add_header('Content-Disposition', 'attachment', filename='screenshot.png')
        msg.attach(screenshot_image)

    # Send the email using SMTP
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.close()
        print('Email sent successfully')
    except Exception as e:
        print('Email could not be sent:', str(e))

    # Delete the temporary screenshot file
    os.remove(screenshot_path)

cap=cv2.VideoCapture(0)
while True:
    ret,img_path=cap.read()
    # Define datagen
    from tensorflow.keras.models import load_model
    from collections import deque
    import numpy as np
    import argparse
    import pickle
    import cv2
    #button2.destroy()
    img_dims = 64
    batch_size = 16

    print("[INFO] loading model and label binarizer...")

    #img = cv2.imread("img\\2.jpeg")

    output = img_path.copy()
    output = cv2.resize(output, (300, 300))
    img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100)).astype("float32")
    preds = model.predict(np.expand_dims(img, axis=0))[0]
    confidence = preds[np.argmax(preds)]

    i = np.argmax(preds)

    if 0==int(i):
        classes='Major Fire'
    else:
        classes='Minor Fire'

    print(confidence)
    if confidence>0.77:
        
        #imgs=cv2.imread(img_path)
        cv2.putText(output, str(classes), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
        cv2.imshow('output', output)
        if classes=='Major Fire':
            if runOnce == False:
                print("Mail send initiated")
                threading.Thread(target=send_mail_function).start() # To call alarm thread
                runOnce = True
            if runOnce == True:
                print("Mail is already sent once")
                runOnce = True
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    cv2.putText(output, 'No fire', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
    cv2.imshow('output', output)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

  

