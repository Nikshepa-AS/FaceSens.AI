import cv2
import numpy as np

MODEL_PATH = "models/sface.onnx"

# Load SFace
recognizer = cv2.FaceRecognizerSF.create(
    MODEL_PATH,
    ""
)

print("✅ SFace model loaded successfully!")


def get_embedding(face_image):
    """
    Convert a face image into a normalized SFace embedding.
    """

    # SFace expects a 112 x 112 aligned face
    face = cv2.resize(face_image, (112, 112))

    # Generate face feature
    embedding = recognizer.feature(face)

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# Test with webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened.")
    exit()

print("📷 Camera started.")
print("Press Q to quit.")

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        break

    # Display camera
    cv2.imshow("FaceSense AI - SFace Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()