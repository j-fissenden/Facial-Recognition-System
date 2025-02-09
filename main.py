import face_recognition
import cv2
import os
import numpy as np

# Path to the known faces directory
KNOWN_FACES_DIR = "known_faces"
TEST_IMAGE_PATH = "test.jpg"

# Load known faces
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        
        # Load and encode the image
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]  # Remove file extension for name
            known_face_names.append(name)

print(f"Loaded {len(known_face_encodings)} known faces.")

# Load test image
print("Processing test image...")
test_image = cv2.imread(TEST_IMAGE_PATH)

if test_image is None:
    print("Error: Test image not found.")
    exit()

rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Detect faces in the test image
face_locations = face_recognition.face_locations(rgb_test_image)
face_encodings = face_recognition.face_encodings(rgb_test_image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Compare detected face with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown Person"

    # Find the closest match
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

    if best_match_index is not None and matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face and label it
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the output
cv2.imshow("Face Recognition", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()