import cv2
import numpy as np
import pickle
import os

from embeddings import get_face_embedding


DATABASE_FILE = "data/face_database.pkl"


# ---------------------------------------
# Load database
# ---------------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Database not found.")
    exit()

with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


names = list(database.keys())

if len(names) < 2:
    print("❌ Register at least TWO people first.")
    exit()


print("Registered people:")
for name in names:
    print(f"  - {name}")


# ---------------------------------------
# Cosine similarity
# ---------------------------------------

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


# ---------------------------------------
# Compare query with database
# ---------------------------------------

def best_match(query_embedding):

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


# ---------------------------------------
# Choose test person
# ---------------------------------------

print()
print("Select a registered person for genuine-match testing:")

for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")

choice = int(input("Enter number: ")) - 1

if choice < 0 or choice >= len(names):
    print("❌ Invalid choice.")
    exit()

target_name = names[choice]

print()
print(f"Testing identity: {target_name}")

print()
print("Look at the camera.")
print("We will collect 10 similarity measurements.")
print("Press Q to stop.")


# ---------------------------------------
# Camera
# ---------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened.")
    exit()


scores = []


while len(scores) < 10:

    success, frame = camera.read()

    if not success:
        break

    embedding, face = get_face_embedding(frame)

    if face is not None and embedding is not None:

        name, score = best_match(embedding)

        scores.append(score)

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
            f"{name}: {score:.3f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(
            f"Sample {len(scores)}/10: "
            f"{name} = {score:.3f}"
        )

        cv2.waitKey(500)

    cv2.imshow(
        "FaceSense AI - Threshold Testing",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()


# ---------------------------------------
# Results
# ---------------------------------------

if scores:

    print()
    print("===================================")
    print("THRESHOLD TEST RESULTS")
    print("===================================")

    print(f"Target person : {target_name}")
    print(f"Samples       : {len(scores)}")
    print(f"Minimum score : {min(scores):.3f}")
    print(f"Maximum score : {max(scores):.3f}")
    print(f"Average score : {np.mean(scores):.3f}")

else:

    print("❌ No valid face samples collected.")