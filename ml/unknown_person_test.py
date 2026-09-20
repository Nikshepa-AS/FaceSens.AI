import cv2
import numpy as np
import pickle
import os

from embeddings import get_face_embedding


DATABASE_FILE = "data/face_database.pkl"
SAMPLES_REQUIRED = 10


# --------------------------------
# Load database
# --------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Face database not found.")
    exit()

with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


print("Registered identities:")

for name in database:
    print(f"  - {name}")


# --------------------------------
# Cosine similarity
# --------------------------------

def cosine_similarity(a, b):

    a = a.flatten()
    b = b.flatten()

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


# --------------------------------
# Compare against ALL identities
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

    print("❌ Could not open camera.")
    exit()


print()
print("========================================")
print("TRUE UNKNOWN PERSON TEST")
print("========================================")
print()
print("IMPORTANT:")
print("The person in front of the camera")
print("must NOT be registered in the database.")
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

        best_name, best_score = find_best_match(
            embedding
        )

        scores.append(best_score)

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
            f"Best match: {best_name}",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Similarity: {best_score:.3f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


        print(
            f"Sample {len(scores)}/10: "
            f"{best_name} = {best_score:.3f}"
        )

        cv2.waitKey(500)


    cv2.imshow(
        "FaceSense AI - Unknown Person Test",
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
    print("========================================")
    print("UNKNOWN PERSON RESULTS")
    print("========================================")

    print(f"Samples       : {len(scores)}")
    print(f"Minimum score : {min(scores):.3f}")
    print(f"Maximum score : {max(scores):.3f}")
    print(f"Average score : {np.mean(scores):.3f}")

else:

    print("❌ No valid samples collected.")