import cv2
import numpy as np
import pickle
import os


# ============================================================
# PATHS
# ============================================================

YUNET_MODEL = "models/yunet.onnx"

SFACE_MODEL = "models/sface.onnx"

DATABASE_FILE = "data/face_database.pkl"


# ============================================================
# VALIDATED THRESHOLD
# ============================================================

THRESHOLD = 0.41


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    if not os.path.exists(
        DATABASE_FILE
    ):

        print(
            "⚠️ Face database not found."
        )

        return {}


    try:

        with open(
            DATABASE_FILE,
            "rb"
        ) as file:

            database = pickle.load(
                file
            )


        return database


    except Exception as error:

        print(
            "❌ Database loading error:",
            error
        )

        return {}


# ============================================================
# LOAD YUNET
# ============================================================

if not os.path.exists(
    YUNET_MODEL
):

    raise FileNotFoundError(
        f"YuNet model not found: {YUNET_MODEL}"
    )


detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


# ============================================================
# LOAD SFACE
# ============================================================

if not os.path.exists(
    SFACE_MODEL
):

    raise FileNotFoundError(
        f"SFace model not found: {SFACE_MODEL}"
    )


recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)


# ============================================================
# STARTUP INFORMATION
# ============================================================

print(
    "✅ Recognition service loaded"
)

print(
    "✅ YuNet loaded"
)

print(
    "✅ SFace loaded"
)

print(
    f"✅ Threshold: {THRESHOLD}"
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b
):

    a = np.asarray(
        a,
        dtype=np.float32
    ).flatten()

    b = np.asarray(
        b,
        dtype=np.float32
    ).flatten()


    denominator = (

        np.linalg.norm(a)
        *
        np.linalg.norm(b)

    )


    if denominator == 0:

        return 0.0


    score = (

        np.dot(a, b)
        /
        denominator

    )


    return float(
        score
    )


# ============================================================
# FIND BEST MATCH
# ============================================================

def find_best_match(
    query_embedding,
    database
):

    best_name = None

    best_score = -1.0


    # --------------------------------------------------------
    # No registered users
    # --------------------------------------------------------

    if not database:

        return (
            None,
            0.0
        )


    # --------------------------------------------------------
    # Compare with every registered person
    # --------------------------------------------------------

    for name, embeddings in database.items():

        # Safety check

        if embeddings is None:

            continue


        # Make sure embeddings is iterable

        for stored_embedding in embeddings:

            try:

                score = cosine_similarity(
                    query_embedding,
                    stored_embedding
                )


                if score > best_score:

                    best_score = score

                    best_name = name


            except Exception as error:

                print(
                    "⚠️ Embedding comparison error:",
                    error
                )


    return (
        best_name,
        best_score
    )


# ============================================================
# IDENTIFY IMAGE
# ============================================================

def identify_image(
    image_bytes
):

    # --------------------------------------------------------
    # Load latest database
    #
    # Important:
    # This allows newly registered people to be recognized
    # without requiring a server restart.
    # --------------------------------------------------------

    database = load_database()


    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # Set detector input size
    # --------------------------------------------------------

    detector.setInputSize(
        (
            width,
            height
        )
    )


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    _, faces = detector.detect(
        frame
    )


    # --------------------------------------------------------
    # No face
    # --------------------------------------------------------

    if faces is None:

        return {

            "face_count":
                0,

            "threshold":
                THRESHOLD,

            "message":
                "No face detected.",

            "results":
                []

        }


    # --------------------------------------------------------
    # Process faces
    # --------------------------------------------------------

    results = []


    for face in faces:

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, w, h = (
            face[:4]
            .astype(int)
        )


        # ----------------------------------------------------
        # Align face
        # ----------------------------------------------------

        aligned_face = (
            recognizer.alignCrop(
                frame,
                face
            )
        )


        # ----------------------------------------------------
        # Generate SFace embedding
        # ----------------------------------------------------

        embedding = (
            recognizer.feature(
                aligned_face
            )
        )


        # ----------------------------------------------------
        # Normalize embedding
        # ----------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )


        if norm != 0:

            embedding = (
                embedding /
                norm
            )


        # ----------------------------------------------------
        # Find best identity
        # ----------------------------------------------------

        name, score = (
            find_best_match(
                embedding,
                database
            )
        )


        # ----------------------------------------------------
        # Unknown rejection
        # ----------------------------------------------------

        if (
            name is not None
            and
            score >= THRESHOLD
        ):

            status = "known"

            identified_name = name

        else:

            status = "unknown"

            identified_name = None


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append({

            "name":
                identified_name,

            "similarity":
                round(
                    float(score),
                    4
                ),

            "status":
                status,

            "bounding_box": {

                "x":
                    int(x),

                "y":
                    int(y),

                "width":
                    int(w),

                "height":
                    int(h)

            }

        })


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "face_count":
            len(results),

        "threshold":
            THRESHOLD,

        "message":
            "Recognition completed.",

        "results":
            results

    }
    
    
def register_person(name, image_bytes):
    """
    Register a person using one camera/image sample.
    The person's face embedding is stored in face_database.pkl.
    """

    if not name or not name.strip():
        raise ValueError("Person name is required.")

    name = name.strip()

    # Read image
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
            "Could not read registration image."
        )

    height, width = frame.shape[:2]

    # Detect faces
    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        raise ValueError(
            "No face detected. Please keep your face clearly visible."
        )

    if len(faces) > 1:
        raise ValueError(
            "Multiple faces detected. Please keep only one person in the frame."
        )

    # Get the detected face
    face = faces[0]

    # Align face
    aligned_face = recognizer.alignCrop(
        frame,
        face
    )

    # Generate SFace embedding
    embedding = recognizer.feature(
        aligned_face
    )

    # Normalize embedding
    norm = np.linalg.norm(embedding)

    if norm != 0:
        embedding = embedding / norm

    embedding = embedding.astype(
        np.float32
    )

    # Load existing database
    database = load_database()

    if name not in database:
        database[name] = []

    # Store embedding
    database[name].append(
        embedding
    )

    # Save database
    os.makedirs(
        os.path.dirname(DATABASE_FILE),
        exist_ok=True
    )

    with open(
        DATABASE_FILE,
        "wb"
    ) as file:

        pickle.dump(
            database,
            file
        )

    return {
        "success": True,
        "name": name,
        "samples": len(
            database[name]
        ),
        "message": (
            f"{name} registered successfully."
        )
    }