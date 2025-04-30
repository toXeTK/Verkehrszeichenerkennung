import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

chptfile = 38

# Lade das trainierte Modell
MODEL_PATH = '/home/pi/Documents/Verkehrszeichenerkennung/projekt_hk_kt/chpt/' + f'{chptfile}/' +  f'{chptfile}-chpt.model.keras' 
model = load_model(MODEL_PATH)
 
print("imputshape:" ,model.input_shape)
 
 
# Definiere Bildparameter
IMG_SIZE = (64,64)
 
# Starte die USB-Kamera
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    print("Fehler: Konnte Kamera nicht öffnen!")
    exit()
 
while True:
    # Aufnahme eines Frames
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Kamerabildes!")
        break
    # Preprocessing: Bild in richtige Größe umwandeln
    image = cv2.resize(frame, IMG_SIZE)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    image = image.astype('float32') / 255.0 # image normalization
    #image = image.flatten()
    image = np.expand_dims(image, axis=0)  
 
    print("imagesize:" ,image.shape)
 
    
    # Vorhersage mit dem Modell
    prediction = model.predict(image)
    class_id = np.argmax(prediction)  # Nimm die Klasse mit der höchsten Wahrscheinlichkeit
    confidence = np.max(prediction)  # Hole die höchste Wahrscheinlichkeit
    print("Klassen:", prediction)
    # Ergebnisse auf dem Bild anzeigen
    label = f"Class: {class_id}, Confidence: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Classification", frame)
    # Beenden mit der 'q'-Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Aufräumen
cap.release()
cv2.destroyAllWindows()
