import cv2
import numpy as np

# -----------------------------
# Model paths
# -----------------------------
YUNET_MODEL = "models/yunet.onnx"
SFACE_MODEL = "models/sface.onnx"

# -----------------------------
# Load YuNet
# -----------------------------
detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# -----------------------------
# Load SFace
# -----------------------------
recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

print("✅ YuNet loaded")
print("✅ SFace loaded")


def get_face_embedding(frame):
    """
    Detect the largest face in a frame
    and generate its SFace embedding.
    """

    height, width = frame.shape[:2]

    # YuNet needs the current image size
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is None:
        return None, None

    # Select largest face
    largest_face = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # --------------------------------
    # Align face using YuNet landmarks
    # --------------------------------
    aligned_face = recognizer.alignCrop(
        frame,
        largest_face
    )

    # --------------------------------
    # Generate SFace feature vector
    # --------------------------------
    embedding = recognizer.feature(
        aligned_face
    )

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding, largest_face


# -----------------------------
# Webcam test
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera")
    exit()

print("📷 Camera started")
print("Press Q to quit")

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera frame")
        break

    embedding, face = get_face_embedding(frame)

    if face is not None:

        x, y, w, h = face[:4].astype(int)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if embedding is not None:

            cv2.putText(
                frame,
                f"Embedding: {embedding.shape}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "FaceSense AI - Face Embedding",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()