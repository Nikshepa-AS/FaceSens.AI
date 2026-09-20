import cv2
import numpy as np
import pickle
import os


# --------------------------------
# Paths
# --------------------------------

YUNET_MODEL = "models/yunet.onnx"
SFACE_MODEL = "models/sface.onnx"
DATABASE_FILE = "data/face_database.pkl"


# --------------------------------
# Validated threshold
# --------------------------------

THRESHOLD = 0.41


# --------------------------------
# Load face database
# --------------------------------

if not os.path.exists(DATABASE_FILE):
    raise FileNotFoundError(
        "Face database not found."
    )


with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


# --------------------------------
# Load YuNet
# --------------------------------

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


# --------------------------------
# Load SFace
# --------------------------------

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)


print("✅ Recognition service loaded")
print("✅ YuNet loaded")
print("✅ SFace loaded")
print(f"✅ Threshold: {THRESHOLD}")


# --------------------------------
# Cosine similarity
# --------------------------------

def cosine_similarity(a, b):

    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
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

    best_name = None
    best_score = -1.0

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
# Identify faces in image
# --------------------------------

def identify_image(image_bytes):

    # Convert uploaded bytes → image
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise ValueError(
            "Could not read uploaded image."
        )


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    # --------------------------------
    # Detect faces
    # --------------------------------

    _, faces = detector.detect(frame)


    if faces is None:

        return {
            "face_count": 0,
            "results": []
        }


    results = []


    # --------------------------------
    # Process every detected face
    # --------------------------------

    for face in faces:

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


        embedding = (
            embedding /
            np.linalg.norm(embedding)
        )


        # --------------------------------
        # Find best match
        # --------------------------------

        name, score = find_best_match(
            embedding
        )


        # --------------------------------
        # Unknown rejection
        # --------------------------------

        if score >= THRESHOLD:

            status = "known"

            identified_name = name

        else:

            status = "unknown"

            identified_name = None


        results.append({

            "name": identified_name,

            "similarity": round(
                float(score),
                4
            ),

            "status": status,

            "bounding_box": {

                "x": int(x),

                "y": int(y),

                "width": int(w),

                "height": int(h)

            }

        })


    return {

        "face_count": len(results),

        "threshold": THRESHOLD,

        "results": results

    }