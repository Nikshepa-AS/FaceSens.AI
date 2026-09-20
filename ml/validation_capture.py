import cv2
import pickle
import os

from embeddings import get_face_embedding


OUTPUT_FILE = "results/validation_data.pkl"
DATABASE_FILE = "data/face_database.pkl"

SAMPLES_REQUIRED = 20


# --------------------------------
# Create results folder
# --------------------------------

os.makedirs("results", exist_ok=True)


# --------------------------------
# Load registered identities
# --------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Face database not found.")
    exit()

with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


names = list(database.keys())

print("\nRegistered identities:")

for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")


# --------------------------------
# Select person
# --------------------------------

try:
    choice = int(input("\nSelect person for validation: "))
except ValueError:
    print("❌ Please enter a number.")
    exit()


if choice < 1 or choice > len(names):
    print("❌ Invalid choice.")
    exit()


name = names[choice - 1]


# --------------------------------
# Load existing validation data
# --------------------------------

if os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE, "rb") as file:
        validation_data = pickle.load(file)

else:

    validation_data = {}


if name not in validation_data:
    validation_data[name] = []


# Start fresh for this person
validation_data[name] = []


# --------------------------------
# Start camera
# --------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera.")
    exit()


print()
print("======================================")
print(f"VALIDATION CAPTURE: {name}")
print("======================================")
print()
print("C = Capture sample")
print("Q = Quit")
print()
print("Capture 20 NEW samples.")
print()


# --------------------------------
# Capture loop
# --------------------------------

while len(validation_data[name]) < SAMPLES_REQUIRED:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera.")
        break


    embedding, face = get_face_embedding(frame)


    # --------------------------------
    # Face detected
    # --------------------------------

    if face is not None and embedding is not None:

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
            "C = Capture | Q = Quit",
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


    cv2.putText(
        frame,
        f"Samples: {len(validation_data[name])}/{SAMPLES_REQUIRED}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "FaceSense AI - Validation Capture",
        frame
    )


    # --------------------------------
    # Keyboard input
    # --------------------------------

    key = cv2.waitKey(1) & 0xFF


    # C = capture
    if key == ord("c") or key == ord("C"):

        if embedding is not None:

            validation_data[name].append(
                embedding.flatten()
            )

            print(
                f"✅ Captured "
                f"{len(validation_data[name])}/"
                f"{SAMPLES_REQUIRED}"
            )

        else:

            print("❌ No face detected. Try again.")


    # Q = quit
    elif key == ord("q") or key == ord("Q"):

        print("⚠️ Capture stopped.")
        break


# --------------------------------
# Release camera
# --------------------------------

camera.release()
cv2.destroyAllWindows()


# --------------------------------
# Save validation data
# --------------------------------

with open(OUTPUT_FILE, "wb") as file:

    pickle.dump(
        validation_data,
        file
    )


print()
print("======================================")
print("VALIDATION SAVED")
print("======================================")
print(f"Person  : {name}")
print(f"Samples : {len(validation_data[name])}")
print(f"File    : {OUTPUT_FILE}")