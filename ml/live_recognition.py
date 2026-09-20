import cv2
import numpy as np
import pickle
import os


# --------------------------------
# Configuration
# --------------------------------

YUNET_MODEL = "models/yunet.onnx"
SFACE_MODEL = "models/sface.onnx"
DATABASE_FILE = "data/face_database.pkl"

THRESHOLD = 0.41


# --------------------------------
# Load database
# --------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Face database not found.")
    exit()

with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


print("✅ Face database loaded")

for name in database:
    print(f"   👤 {name}")


# --------------------------------
# Load models
# --------------------------------

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

print("✅ YuNet loaded")
print("✅ SFace loaded")
print(f"✅ Validation threshold: {THRESHOLD}")


# --------------------------------
# Cosine similarity
# --------------------------------

def cosine_similarity(a, b):

    a = a.flatten()
    b = b.flatten()

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# --------------------------------
# Find best identity
# --------------------------------

def find_best_match(query_embedding):

    best_name = "Unknown"
    best_score = -1

    for name, embeddings in database.items():

        for stored_embedding in embeddings:

            score = cosine_similarity(
                query_embedding,
                stored_embedding
            )

            if score > best_score:

                best_score = score
                best_name = name

    return best_name, best_score


# --------------------------------
# Camera
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open camera.")
    exit()


print()
print("======================================")
print("FaceSense AI - LIVE RECOGNITION")
print("======================================")
print("Press Q to quit.")


while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera.")
        break


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    # --------------------------------
    # Detect ALL faces
    # --------------------------------

    _, faces = detector.detect(frame)


    if faces is not None:

        for face in faces:

            # Face bounding box
            x, y, w, h = face[:4].astype(int)


            # --------------------------------
            # Align face
            # --------------------------------

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )


            # --------------------------------
            # Generate embedding
            # --------------------------------

            embedding = recognizer.feature(
                aligned_face
            )


            # Normalize
            embedding = (
                embedding /
                np.linalg.norm(embedding)
            )


            # --------------------------------
            # Match
            # --------------------------------

            name, score = find_best_match(
                embedding
            )


            # --------------------------------
            # Unknown rejection
            # --------------------------------

            if score >= THRESHOLD:

                label = (
                    f"{name} | "
                    f"{score:.3f}"
                )

                box_color = (0, 255, 0)

            else:

                label = (
                    f"UNKNOWN | "
                    f"{score:.3f}"
                )

                box_color = (0, 0, 255)


            # --------------------------------
            # Draw result
            # --------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )


            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )


    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------
    # Display threshold
    # --------------------------------

    cv2.putText(
        frame,
        f"Threshold: {THRESHOLD:.2f}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "FaceSense AI - Live Recognition",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()