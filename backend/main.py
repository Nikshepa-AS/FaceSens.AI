from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form
)

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime

from backend.recognition_service import (
    identify_image,
    register_person
)

from backend.database import (
    initialize_database,
    get_connection,
    log_recognition,
    add_user
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="FaceSense AI",
    description=(
        "Intelligent Face Identification and "
        "Unknown-Person Rejection System"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/app",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "FaceSense AI",
        "status": "running",
        "message": "Face identification API is online"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# USERS
# ============================================================

@app.get("/users")
def get_users():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                created_at
            FROM users
            ORDER BY id
            """
        )

        rows = cursor.fetchall()

    finally:

        connection.close()


    users = []

    for row in rows:

        users.append(
            {
                "id": row[0],
                "name": row[1],
                "created_at": row[2]
            }
        )


    return {
        "count": len(users),
        "users": users
    }


# ============================================================
# RECOGNITION HISTORY
# ============================================================

@app.get("/history")
def get_history():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                similarity,
                status,
                timestamp
            FROM recognition_history
            ORDER BY id DESC
            LIMIT 50
            """
        )

        rows = cursor.fetchall()

    finally:

        connection.close()


    history = []

    for row in rows:

        history.append(
            {
                "id": row[0],
                "name": row[1],
                "similarity": row[2],
                "status": row[3],
                "timestamp": row[4]
            }
        )


    return {
        "count": len(history),
        "history": history
    }


# ============================================================
# REGISTER PERSON
# ============================================================

@app.post("/register")
async def register(
    name: str = Form(...),
    file: UploadFile = File(...)
):

    try:

        # Validate name
        name = name.strip()

        if not name:

            raise ValueError(
                "Person name is required."
            )


        # Validate file
        if not file.content_type:

            raise ValueError(
                "File type could not be detected."
            )


        if not file.content_type.startswith("image/"):

            raise ValueError(
                "Please upload an image file."
            )


        # Read image
        image_bytes = await file.read()


        if not image_bytes:

            raise ValueError(
                "Uploaded image is empty."
            )


        # Run face registration
        result = register_person(
            name,
            image_bytes
        )


        # Add identity to SQLite
        add_user(name)


        return result


    except Exception as error:

        print(
            "❌ Registration error:",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# IDENTIFY FACE
# ============================================================

@app.post("/identify")
async def identify(
    file: UploadFile = File(...)
):

    try:

        # Validate file
        if not file.content_type:

            raise ValueError(
                "File type could not be detected."
            )


        if not file.content_type.startswith("image/"):

            raise ValueError(
                "Please upload an image file."
            )


        # Read image
        image_bytes = await file.read()


        if not image_bytes:

            raise ValueError(
                "Uploaded image is empty."
            )


        # FaceSense AI recognition
        result = identify_image(
            image_bytes
        )


        # Save recognition history
        for face_result in result.get(
            "results",
            []
        ):

            name = face_result.get(
                "name"
            )

            similarity = face_result.get(
                "similarity",
                0.0
            )

            status = face_result.get(
                "status",
                "unknown"
            )


            history_name = (
                name
                if name
                else "Unknown"
            )


            log_recognition(
                history_name,
                similarity,
                status
            )


        return result


    except Exception as error:

        print(
            "❌ Identification error:",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# IDENTIFY IMAGE ALIAS
# ============================================================

@app.post("/identify/image")
async def identify_image_upload(
    file: UploadFile = File(...)
):

    return await identify(file)


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api/info")
def api_info():

    return {

        "project":
            "FaceSense AI",

        "version":
            "1.0.0",

        "engine":
            "YuNet + SFace",

        "threshold":
            0.41,

        "features": [

            "Face Detection",
            "Face Embeddings",
            "Similarity Matching",
            "Unknown Rejection",
            "Multi-Face Recognition",
            "Live Recognition",
            "Recognition History",
            "Person Registration"

        ],

        "endpoints": {

            "health":
                "/health",

            "users":
                "/users",

            "history":
                "/history",

            "register":
                "/register",

            "identify":
                "/identify",

            "identify_image":
                "/identify/image",
            "evaluation":
                "/evaluation"

        }

    }
@app.get("/evaluation")
def get_evaluation():

    import os
    import json

    evaluation_file = (
        "results/evaluation_summary.json"
    )

    if not os.path.exists(evaluation_file):
        raise HTTPException(
            status_code=404,
            detail=(
                "Evaluation results not found. "
                "Run: python ml/evaluator.py"
            )
        )

    with open(
        evaluation_file,
        "r"
    ) as file:

        evaluation = json.load(file)

    return evaluation