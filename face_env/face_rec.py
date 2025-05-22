import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os

# Initialize face detector and embedding model
detector = MTCNN()
embedder = FaceNet()

# Load known face embeddings
known_embeddings = []
known_names = []

def register_known_faces(folder="known_faces"):
    for filename in os.listdir(folder):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        if faces:
            x, y, w, h = faces[0]['box']
            face = rgb[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face])[0]
            known_embeddings.append(embedding)
            known_names.append(name)
    print(f"Registered {len(known_names)} known faces.")

def recognize_face(embedding, threshold=0.8):
    if not known_embeddings:
        return "Unknown"
    distances = [np.linalg.norm(embedding - known) for known in known_embeddings]
    min_dist = min(distances)
    if min_dist < threshold:
        return known_names[distances.index(min_dist)]
    return "Unknown"

# Register known faces
register_known_faces()

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    for det in detections:
        x, y, w, h = det['box']
        face = rgb[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face])[0]
            name = recognize_face(embedding)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        except:
            continue

    cv2.imshow('Real-Time Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
