import cv2
import numpy as np
import os
import pickle

from embeddings import get_face_embedding


# --------------------------------
# Configuration
# --------------------------------

DATA_FOLDER = "data"
DATABASE_FILE = os.path.join(
    DATA_FOLDER,
    "face_database.pkl"
)

SAMPLES_REQUIRED = 5


# --------------------------------
# Create data folder
# --------------------------------

os.makedirs(DATA_FOLDER, exist_ok=True)


# --------------------------------
# Load existing database
# --------------------------------

if os.path.exists(DATABASE_FILE):

    with open(DATABASE_FILE, "rb") as file:
        database = pickle.load(file)

else:

    database = {}


# --------------------------------
# Get person's name
# --------------------------------

name = input("Enter person's name: ").strip()

if not name:

    print("❌ Name cannot be empty.")
    exit()


# --------------------------------
# Start camera
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open camera.")
    exit()


print()
print(f"👤 Registering: {name}")
print(f"📷 Capture {SAMPLES_REQUIRED} face samples")
print("Please look at the camera.")
print()


embeddings = []


# --------------------------------
# Capture samples
# --------------------------------

while len(embeddings) < SAMPLES_REQUIRED:

    success, frame = camera.read()

    if not success:

        print("❌ Could not read camera.")
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


        # Capture embedding
        if embedding is not None:

            embeddings.append(
                embedding.flatten()
            )

            print(
                f"✅ Sample {len(embeddings)}/{SAMPLES_REQUIRED} captured"
            )


            # Small delay between captures
            cv2.waitKey(500)


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


    cv2.putText(
        frame,
        f"Samples: {len(embeddings)}/{SAMPLES_REQUIRED}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "FaceSense AI - Registration",
        frame
    )


    # Press Q to cancel
    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("❌ Registration cancelled.")
        camera.release()
        cv2.destroyAllWindows()
        exit()


# --------------------------------
# Save registration
# --------------------------------

camera.release()
cv2.destroyAllWindows()


if len(embeddings) == SAMPLES_REQUIRED:

    database[name] = embeddings

    with open(DATABASE_FILE, "wb") as file:

        pickle.dump(
            database,
            file
        )

    print()
    print("================================")
    print("✅ REGISTRATION SUCCESSFUL")
    print("================================")
    print(f"Name     : {name}")
    print(f"Samples  : {len(embeddings)}")
    print(f"Database : {DATABASE_FILE}")


else:

    print()
    print("❌ Registration failed.")