import cv2
import numpy as np
import pickle
import os

from embeddings import get_face_embedding


DATABASE_FILE = "data/face_database.pkl"

# Person whose identity we are testing against
TARGET_NAME = "Nikshepa"

SAMPLES_REQUIRED = 10


# --------------------------------
# Load database
# --------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Database not found.")
    exit()

with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


if TARGET_NAME not in database:
    print(f"❌ {TARGET_NAME} is not registered.")
    exit()


# --------------------------------
# Cosine similarity
# --------------------------------

def cosine_similarity(a, b):

    a = a.flatten()
    b = b.flatten()

    return float(
        np.dot(a, b) /
        (
            np.linalg.norm(a) *
            np.linalg.norm(b)
        )
    )


# --------------------------------
# Compare ONLY with Nikshepa
# --------------------------------

def compare_with_target(query_embedding):

    best_score = -1

    for stored_embedding in database[TARGET_NAME]:

        score = cosine_similarity(
            query_embedding,
            stored_embedding
        )

        if score > best_score:
            best_score = score

    return best_score


# --------------------------------
# Camera
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened.")
    exit()


print()
print("======================================")
print("UNKNOWN PERSON TEST")
print("======================================")
print(f"Comparing face against: {TARGET_NAME}")
print()
print("Ask the OTHER registered person")
print("to stand in front of the camera.")
print()
print("Collecting 10 samples...")
print()


scores = []


while len(scores) < SAMPLES_REQUIRED:

    success, frame = camera.read()

    if not success:
        break

    embedding, face = get_face_embedding(frame)

    if face is not None and embedding is not None:

        score = compare_with_target(embedding)

        scores.append(score)

        x, y, w, h = face[:4].astype(int)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Similarity: {score:.3f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        print(
            f"Sample {len(scores)}/10: "
            f"{score:.3f}"
        )

        cv2.waitKey(500)


    cv2.imshow(
        "FaceSense AI - Unknown Test",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()


# --------------------------------
# Results
# --------------------------------

if scores:

    print()
    print("======================================")
    print("UNKNOWN PERSON RESULTS")
    print("======================================")

    print(f"Samples       : {len(scores)}")
    print(f"Minimum score : {min(scores):.3f}")
    print(f"Maximum score : {max(scores):.3f}")
    print(f"Average score : {np.mean(scores):.3f}")

else:

    print("❌ No valid samples collected.")
    