import cv2
import numpy as np
import pickle
import os

from embeddings import get_face_embedding


DATABASE_FILE = "data/face_database.pkl"


# --------------------------------
# Load registered faces
# --------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Face database not found.")
    print("Please register a person first.")
    exit()


with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


print("✅ Face database loaded")

print("Registered people:")

for name in database:
    print(f"   👤 {name}")


# --------------------------------
# Cosine similarity
# --------------------------------

def cosine_similarity(embedding1, embedding2):

    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    similarity = np.dot(
        embedding1,
        embedding2
    ) / (
        np.linalg.norm(embedding1)
        *
        np.linalg.norm(embedding2)
    )

    return float(similarity)


# --------------------------------
# Find best match
# --------------------------------

def find_best_match(query_embedding):

    best_name = None
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

    print("❌ Could not open camera")
    exit()


print()
print("📷 Recognition started")
print("Press Q to quit")


while True:

    success, frame = camera.read()

    if not success:
        break


    embedding, face = get_face_embedding(frame)


    if face is not None and embedding is not None:

        x, y, w, h = face[:4].astype(int)


        name, score = find_best_match(
            embedding
        )


        # -------------------------
        # Temporary threshold
        # -------------------------
        #
        # IMPORTANT:
        # This is NOT our final threshold.
        # We will calibrate it later.
        #

        threshold = 0.50


        if score >= threshold:

            label = f"{name} | {score:.3f}"

        else:

            label = f"UNKNOWN | {score:.3f}"


        # Draw face box

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            label,
            (x, y - 10),
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
            0.8,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "FaceSense AI - Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()