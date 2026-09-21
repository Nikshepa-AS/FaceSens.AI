# FaceSense AI

## Intelligent Face Identification, Registration & Unknown-Person Rejection System

FaceSense AI is an end-to-end AI/ML face identification system that detects faces, generates face embeddings, compares them with registered identities using similarity matching, and rejects faces that do not sufficiently match the registered database.

The project combines pretrained deep-learning models with a custom recognition pipeline, FastAPI backend, SQLite database, and a browser-based frontend.

---

## 1. Project Overview

FaceSense AI demonstrates a practical face-recognition workflow:

```text
Input Image / Camera Frame
          |
          v
    Face Detection
        YuNet
          |
          v
    Detected Face
          |
          v
   Face Embedding
        SFace
          |
          v
 Numerical Embedding
          |
          v
 Cosine Similarity
          |
          v
 Best Matching Identity
          |
          v
 Threshold Decision
       /       \
    Known     Unknown
```

The system supports person registration, image-based recognition, live camera recognition, multiple-face recognition, person management, recognition history, and evaluation of the matching threshold.

The project focuses on using pretrained deep-learning models effectively rather than training a deep neural network from scratch.

---

## 2. Objectives

The main objectives of FaceSense AI are:

- Detect faces from images and camera frames.
- Generate numerical face embeddings.
- Register known individuals.
- Compare new face embeddings with registered embeddings.
- Identify registered individuals using similarity matching.
- Reject unknown people when the similarity score is below the configured threshold.
- Support multiple faces in a single image.
- Provide live camera recognition.
- Maintain recognition history.
- Provide registered-person management.
- Evaluate similarity thresholds using genuine and impostor comparisons.
- Test the system under different practical failure conditions.

---

## 3. Key Features

| Feature | Description |
|---|---|
| Face Detection | Detects faces using the YuNet face detector |
| Face Embeddings | Generates numerical facial representations using SFace |
| Person Registration | Registers an identity and stores its face embedding |
| Face Identification | Compares a new face against registered identities |
| Unknown Rejection | Rejects faces that do not satisfy the matching threshold |
| Similarity Matching | Uses cosine similarity between face embeddings |
| Image Recognition | Recognizes faces from uploaded images |
| Live Recognition | Processes frames from the browser camera |
| Multiple-Face Recognition | Processes multiple detected faces independently |
| Person Management | Displays, searches, and deletes registered people |
| Recognition History | Stores recognition results, similarity scores, status, and timestamps |
| Evaluation | Evaluates genuine/impostor similarity and threshold performance |
| Web Interface | Provides a browser-based interface for interacting with the system |
| FastAPI Backend | Provides REST API endpoints for frontend and ML operations |

---

## 4. Technology Stack

| Technology | Role |
|---|---|
| Python | Main programming language |
| OpenCV | Computer vision and ONNX model inference |
| YuNet | Face detection |
| SFace | Face embedding generation and face recognition |
| NumPy | Numerical operations and similarity calculations |
| Pandas | Evaluation data processing |
| scikit-learn | Evaluation and metric calculations |
| Matplotlib | Evaluation/visualization support |
| FastAPI | Backend REST API |
| Uvicorn | ASGI application server |
| HTML | Frontend structure |
| CSS | Frontend styling and animations |
| JavaScript | Frontend interaction and API communication |
| SQLite | User and recognition-history storage |
| ONNX | Model format for YuNet and SFace |

---

## 5. Machine Learning Models

FaceSense AI uses two pretrained deep-learning computer-vision models.

### 5.1 YuNet Face Detector

YuNet is used for face detection.

Its purpose is to locate faces in an input image or camera frame.

```text
Input Image
     |
     v
   YuNet
     |
     v
Face Bounding Boxes
     |
     v
Detected Face Regions
```

The detected face bounding boxes are used to identify the individual face regions that are subsequently processed by the recognition model.

Model file:

```text
models/yunet.onnx
```

### 5.2 SFace Face Recognition Model

SFace is used to generate numerical face embeddings from detected face regions.

```text
Detected Face
      |
      v
    SFace
      |
      v
Face Embedding
```

The embedding represents facial features in numerical form and is used for similarity-based recognition.

Model file:

```text
models/sface.onnx
```

### Face Detection vs Face Recognition

These two models perform different tasks:

```text
YuNet
  |
  +--> Detects WHERE the face is

SFace
  |
  +--> Generates a numerical representation of the face
```

YuNet does not generate the face embeddings used for identity matching.

---

## 6. Face Embeddings

A face embedding is a numerical representation of facial features.

Instead of directly comparing two images pixel by pixel, FaceSense AI converts detected faces into numerical embeddings.

```text
Face Image
    |
    v
SFace Model
    |
    v
Numerical Face Embedding
```

### Registration

During registration:

```text
Person Image
     |
     v
Face Detection
     |
     v
SFace Embedding
     |
     v
Store Embedding
```

### Identification

During identification:

```text
New Face
     |
     v
Face Detection
     |
     v
SFace Embedding
     |
     v
Compare with Registered Embeddings
```

The registered embeddings are stored locally in the application's face database.

This allows the system to compare a new face with previously registered identities without storing the complete input image as the matching representation.

---

## 7. Face Recognition Pipeline

The complete recognition pipeline is:

```text
Input Image / Camera Frame
          |
          v
   Face Detection
       (YuNet)
          |
          v
    Detected Face
          |
          v
   Face Embedding
       (SFace)
          |
          v
 Numerical Embedding Vector
          |
          v
    Cosine Similarity
          |
          v
  Best Matching Identity
          |
          v
    Threshold Decision
       /          \
      /            \
   Known          Unknown
```

### Pipeline Steps

1. The application receives an image or camera frame.
2. YuNet detects one or more faces.
3. Each detected face is processed by SFace.
4. SFace generates a numerical face embedding.
5. The embedding is compared with registered embeddings.
6. Cosine similarity is calculated for the comparisons.
7. The highest similarity score is selected.
8. The score is compared with the configured threshold.
9. If the threshold is satisfied, the corresponding registered identity is returned.
10. Otherwise, the face is rejected as `Unknown`.

---

## 8. Similarity Matching

FaceSense AI uses cosine similarity for comparing face embeddings.

The cosine similarity formula is:

```text
                A · B
Similarity = -----------
             ||A|| × ||B||
```

Where:

- `A` is the new face embedding.
- `B` is a registered face embedding.
- `A · B` is the dot product.
- `||A||` and `||B||` are vector magnitudes.

The system compares the new embedding against the stored embeddings and selects the registered identity with the highest similarity score.

```text
New Face Embedding
        |
        +----> Nikshepa Similarity
        |
        +----> Khushi Similarity
        |
        +----> Ananya Similarity
        |
        v
Highest Similarity
        |
        v
Threshold Decision
```

Cosine similarity is suitable for comparing embedding vectors because it measures the directional similarity between feature representations.

---

## 9. Matching Threshold and Unknown-Person Rejection

The matching threshold is an important part of the recognition system.

Current application threshold:

```text
0.41
```

The evaluation process selected:

```text
0.40
```

The threshold prevents the system from assigning an identity to every detected face.

```text
Similarity Score
       |
       v
Score >= Threshold?
       |
   +---+---+
   |       |
  YES      NO
   |       |
   v       v
 Known   Unknown
```

### If Similarity >= Threshold

The system accepts the highest-scoring registered identity.

### If Similarity < Threshold

The system rejects the match and returns an `Unknown` result.

This is important because the system is designed to recognize registered identities rather than force every face into one of the available classes.

### Similarity Score vs Confidence

The similarity score is a measure of embedding similarity.

It is not automatically a calibrated probability or confidence percentage.

---

## 10. Threshold Calibration

The threshold was evaluated using genuine and impostor comparisons.

### Registered Identities

```text
Nikshepa
Khushi
Ananya
```

### Genuine Comparisons

```text
23
```

Genuine comparisons represent comparisons between embeddings belonging to the same identity.

### Impostor Comparisons

```text
55
```

Impostor comparisons represent comparisons between embeddings belonging to different identities.

### Genuine Similarity

```text
Mean:    0.868
Minimum: 0.781
Maximum: 0.941
```

### Impostor Similarity

```text
Mean:    0.280
Minimum: 0.193
Maximum: 0.398
```

### Selected Threshold

```text
0.40
```

The evaluation shows that the available genuine comparisons have substantially higher similarity scores than the available impostor comparisons.

The production/application threshold is approximately `0.41`.

### Evaluation Limitation

These results are based on the current development/evaluation data. They are not a universal measurement of real-world face-recognition accuracy.

---

## 11. Evaluation Metrics

The evaluation includes:

- Precision
- Recall
- F1-score
- FAR — False Acceptance Rate
- FRR — False Rejection Rate

### Current Results

```text
Precision: 1.000
Recall:    1.000
F1-score:  1.000
FAR:       0.000
FRR:       0.000
```

### Metric Explanation

**Precision**

Measures the proportion of accepted identity predictions that were correct.

**Recall**

Measures the proportion of genuine matching cases that were correctly accepted.

**F1-score**

Combines precision and recall into a single metric.

**False Acceptance Rate (FAR)**

Measures cases where an impostor is incorrectly accepted as a registered identity.

**False Rejection Rate (FRR)**

Measures cases where a genuine registered identity is incorrectly rejected.

### Important Note

The current evaluation results are based on the development dataset and current registered identities. They should not be interpreted as universal production accuracy.

Real-world performance can vary with:

- Lighting
- Pose
- Blur
- Occlusion
- Camera quality
- Face size
- Enrollment quality
- Dataset diversity
- Number of registered identities

---

## 12. Evaluation Methodology

The evaluation is implemented in:

```text
ml/evaluator.py
```

The evaluation process:

```text
Registered Embeddings
        |
        v
Generate Genuine Comparisons
        |
        +----------------+
        |                |
        v                v
 Genuine Pairs      Impostor Pairs
        |                |
        +-------+--------+
                |
                v
       Calculate Similarity
                |
                v
       Test Multiple Thresholds
                |
                v
      Calculate Evaluation Metrics
                |
                v
       Select Evaluation Threshold
```

The evaluator:

1. Loads the registered face embedding database.
2. Generates genuine comparisons.
3. Generates impostor comparisons.
4. Calculates cosine similarity values.
5. Evaluates different thresholds.
6. Calculates precision, recall, F1, FAR, and FRR.
7. Selects an evaluation threshold based on the available comparison results.
8. Saves evaluation results.

Generated files include:

```text
results/threshold_results.csv
results/evaluation_summary.json
```

The evaluation can be executed using:

```bash
python ml/evaluator.py
```

---

## 13. Failure-Case Testing

The system was tested under several practical conditions.

| Test Condition | Expected/System Behavior | Main Limitation |
|---|---|---|
| Normal condition | Detect and identify a registered face | Depends on enrollment quality |
| Low light | Attempt detection and recognition | Detection/embedding quality can decrease |
| Side pose | Process the visible face when detectable | Extreme poses may reduce recognition quality |
| Blur | Attempt recognition from reduced-quality input | Strong blur can reduce matching quality |
| Multiple faces | Detect and process each face independently | Small or overlapping faces can be harder to process |
| Unknown person | Reject when similarity is below threshold | Threshold depends on evaluation data |
| No face | Return a no-face result | No identity can be produced without a detected face |
| Partial occlusion | Attempt recognition using visible facial information | Heavy occlusion can reduce similarity |

The system is designed to fail safely by returning `Unknown` or a no-face result instead of forcing an identity when the available evidence does not satisfy the configured threshold.

---

## 14. Multiple-Face Recognition

FaceSense AI supports multiple faces in the same image.

For every detected face:

```text
Input Image
     |
     v
Detect Faces
     |
     +------> Face 1
     |          |
     |          v
     |      Embedding
     |          |
     |          v
     |      Matching
     |          |
     |          v
     |       Result
     |
     +------> Face 2
                |
                v
            Embedding
                |
                v
             Matching
                |
                v
              Result
```

Each face receives an independent:

- Detection result
- Embedding
- Similarity score
- Identity decision

A face can therefore be identified as a registered person or rejected as `Unknown` independently of other faces in the image.

---

## 15. Live Camera Recognition

FaceSense AI provides browser-based live camera recognition.

The process is:

```text
Browser Camera
      |
      v
Camera Frame
      |
      v
FastAPI Backend
      |
      v
YuNet Detection
      |
      v
SFace Embedding
      |
      v
Similarity Matching
      |
      v
Threshold Decision
      |
      v
Known / Unknown
      |
      v
Frontend Result
```

The frontend captures camera frames and sends them to the backend recognition service.

The backend processes each frame and returns the recognition result.

Live camera recognition is intended for interactive demonstration and testing.

Live camera frames are processed through the recognition endpoint without continuously storing every frame in recognition history.

---

## 16. Person Registration

The registration workflow is:

```text
Enter Person Name
        |
        v
Upload / Capture Face
        |
        v
Face Detection
        |
        v
SFace Embedding
        |
        v
Store Face Embedding
        |
        v
Store User Information
```

The backend validates:

- Person name
- Uploaded file type
- Image content
- Presence of a detectable face

When registration is successful:

1. The face embedding is added to the face database.
2. The person's name is added to the SQLite users table.
3. The identity becomes available for future recognition.

---

## 17. Person Management

The application provides a registration management interface.

Users can:

- View registered people.
- Search registered people.
- Delete registered people.

When a person is deleted:

```text
Delete Registered Person
          |
          v
Remove Face Embedding
          |
          v
Remove SQLite User Record
          |
          v
Refresh Registered People
```

This keeps the face embedding database and SQLite user database synchronized.

---

## 18. Recognition History

FaceSense AI maintains recognition history for image-based recognition requests.

Each history record contains:

| Field | Description |
|---|---|
| Name | Recognized person or `Unknown` |
| Similarity | Matching similarity score |
| Status | Recognition status |
| Timestamp | Time of recognition |

Example statuses include:

```text
Known
Unknown
```

The backend provides a history endpoint that returns recent recognition records.

Live recognition is processed without continuously logging every camera frame, which avoids unnecessary database growth.

---

## 19. Database Design

FaceSense AI uses SQLite for lightweight local data management.

### Users Table

```text
users
├── id
├── name
└── created_at
```

| Field | Description |
|---|---|
| `id` | Unique user identifier |
| `name` | Registered person's name |
| `created_at` | Registration timestamp |

### Recognition History Table

```text
recognition_history
├── id
├── name
├── similarity
├── status
└── timestamp
```

| Field | Description |
|---|---|
| `id` | Unique recognition record |
| `name` | Recognized identity or Unknown |
| `similarity` | Similarity score |
| `status` | Recognition status |
| `timestamp` | Recognition timestamp |

Face embeddings are stored separately in:

```text
data/face_database.pkl
```

SQLite therefore handles user metadata and recognition history, while the face database stores registered face embeddings.

---

## 20. System Architecture

FaceSense AI follows a modular architecture.

```text
+--------------------------------+
|            Frontend            |
|        HTML + CSS + JS         |
+----------------+---------------+
                 |
                 | HTTP / REST API
                 v
+--------------------------------+
|          FastAPI Backend       |
|          backend/main.py       |
+----------------+---------------+
                 |
        +--------+--------+
        |                 |
        v                 v
+---------------+   +-------------+
| Recognition   |   |   SQLite    |
| Service       |   |  Database   |
+-------+-------+   +-------------+
        |
        v
+-----------------------------+
|          ML Pipeline        |
|                             |
| YuNet → Face Detection      |
| SFace → Face Embedding      |
| Cosine Similarity            |
| Threshold → Decision        |
+-------------+---------------+
              |
              v
+-----------------------------+
|   Face Embedding Database   |
|     face_database.pkl       |
+-----------------------------+

        +----------------+
        |   Evaluation   |
        | evaluator.py   |
        +----------------+
```

### Component Responsibilities

**Frontend**

Provides the user interface for registration, recognition, live camera processing, person management, and history.

**FastAPI Backend**

Handles HTTP requests, input validation, recognition requests, user management, and database operations.

**Recognition Service**

Integrates the YuNet and SFace models and performs embedding generation and similarity matching.

**ML Evaluation**

Evaluates genuine and impostor similarity distributions and threshold performance.

**SQLite**

Stores registered identity information and recognition history.

**Face Embedding Database**

Stores the registered face embeddings used for matching.

---

## 21. API Documentation

The application exposes the following FastAPI endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Check application status |
| `GET` | `/health` | Health check |
| `GET` | `/users` | Retrieve registered people |
| `DELETE` | `/users/{name}` | Delete a registered person |
| `GET` | `/history` | Retrieve recognition history |
| `POST` | `/register` | Register a new person |
| `POST` | `/identify` | Identify faces from an uploaded image |
| `POST` | `/identify/image` | Image-based recognition |
| `POST` | `/identify/live` | Process a live camera frame |
| `GET` | `/api/info` | Retrieve system information |
| `GET` | `/evaluation` | Retrieve evaluation results |

FastAPI provides the communication layer between the browser frontend, recognition service, and database.

---

## 22. Project Structure

The current project structure is:

```text
FaceSense-AI/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── recognition_service.py
│   └── __init__.py
│
├── ml/
│   └── evaluator.py
│
├── models/
│   ├── yunet.onnx
│   └── sface.onnx
│
├── data/
│
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Important Files

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI application and API endpoints |
| `backend/database.py` | SQLite database operations |
| `backend/recognition_service.py` | Face detection, embeddings, matching, registration, and deletion |
| `ml/evaluator.py` | Threshold and recognition evaluation |
| `frontend/index.html` | Main web interface |
| `frontend/style.css` | Frontend styling and animations |
| `frontend/script.js` | Frontend logic and API communication |
| `models/yunet.onnx` | YuNet face detector |
| `models/sface.onnx` | SFace face recognition model |

---

## 23. Installation

### 23.1 Clone the Repository

```bash
git clone <your-github-repository-url>
cd FaceSense-AI
```

### 23.2 Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 23.3 Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
fastapi
uvicorn
python-multipart
opencv-python
numpy
pandas
scikit-learn
matplotlib
```

### 23.4 Required Model Files

Place the required ONNX models inside:

```text
models/
├── yunet.onnx
└── sface.onnx
```

### 23.5 Required Directories

The project uses:

```text
data/
models/
results/
```

The application creates/uses local database and embedding files as required.

---

## 24. Running the Application

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Start the FastAPI server:

```bash
uvicorn backend.main:app --reload
```

The application can then be accessed at:

```text
http://127.0.0.1:8000
```

The application provides access to:

- Dashboard
- Person registration
- Image recognition
- Live camera recognition
- Multiple-face recognition
- Person management
- Recognition history
- Evaluation results

---

## 25. Running the Evaluation

The evaluation script is:

```text
ml/evaluator.py
```

Run it using:

```bash
python ml/evaluator.py
```

The evaluator generates:

```text
results/threshold_results.csv
results/evaluation_summary.json
```

### `threshold_results.csv`

Contains threshold-wise evaluation results.

### `evaluation_summary.json`

Contains the summarized evaluation information used by the application.

---

## 26. Results and Outputs

The main evaluation outputs are:

```text
results/
├── threshold_results.csv
└── evaluation_summary.json
```

The evaluation results include information about:

- Genuine similarity
- Impostor similarity
- Tested thresholds
- Precision
- Recall
- F1-score
- FAR
- FRR
- Selected threshold

The face embedding database is stored locally as:

```text
data/face_database.pkl
```

The SQLite database is stored locally under the data directory.

---

## 27. Security and Privacy

Face recognition involves biometric information and should therefore be handled carefully.

FaceSense AI is designed as a local development/project application.

Important considerations include:

- Obtain appropriate consent before registering faces.
- Avoid uploading personal biometric data to public repositories.
- Keep face embeddings private.
- Keep local databases outside GitHub.
- Avoid committing personal test images.
- Use appropriate access control before deploying the system in a real environment.
- Do not use the system for unauthorized surveillance.

The current project does not claim enterprise-grade encryption or production biometric security.

---

## 28. GitHub Data Protection

The `.gitignore` file is configured to prevent generated and sensitive local data from being committed.

Examples include:

```text
data/face_database.pkl
data/*.db
data/*.sqlite
data/*.sqlite3
data/**/*.jpg
data/**/*.jpeg
data/**/*.png
data/**/*.webp
results/*.csv
results/*.json
results/*.pkl
```

These files should remain local because they may contain:

- Face embeddings
- Biometric test data
- Personal images
- Local databases
- Generated evaluation results

Source code, documentation, and dependency files can be committed to GitHub while local biometric data remains excluded.

---

## 29. Limitations

The current implementation has several limitations.

### Dataset Size

The evaluation was performed using the identities and samples available during development. A larger and more diverse dataset is required for broader validation.

### Lighting

Very low-light conditions can reduce face detection and embedding quality.

### Pose

Large changes in head pose can make detection or recognition more difficult.

### Blur

Strong motion or image blur can reduce recognition performance.

### Occlusion

Masks, hands, glasses, or other occlusions can reduce the available facial information.

### Threshold Dependency

The selected threshold depends on the available development data and may require recalibration for a different environment or population.

### No Liveness Detection

The current implementation does not provide dedicated liveness or anti-spoofing detection.

### Local Deployment

The current system is designed for local execution and is not presented as a large-scale production deployment.

### Pretrained Models

The project uses pretrained YuNet and SFace models rather than training a face-recognition model from scratch.

### Evaluation Scope

The current evaluation results should not be interpreted as universal real-world accuracy.

---

## 30. Future Improvements

Possible future improvements include:

### 30.1 More Diverse Enrollment

Use multiple samples per person with different:

- Lighting conditions
- Facial poses
- Expressions
- Camera distances

### 30.2 Larger Evaluation Dataset

Evaluate the system using a larger and more diverse dataset.

### 30.3 Improved Threshold Calibration

Perform threshold calibration on a broader validation dataset and evaluate the trade-off between false acceptance and false rejection.

### 30.4 Liveness / Anti-Spoofing

Add liveness detection to help distinguish real faces from photographs or replay attacks.

### 30.5 Better Pose Handling

Improve recognition under large head rotations and difficult viewing angles.

### 30.6 Low-Light Enhancement

Add image preprocessing or enhancement techniques for challenging lighting conditions.

### 30.7 Face Tracking

Use face tracking to reduce repeated detection and improve live-camera performance.

### 30.8 Secure Biometric Storage

Future versions could use stronger protection mechanisms for stored biometric representations.

### 30.9 Encryption and Access Control

Add encryption and authentication/authorization for sensitive biometric operations.

### 30.10 Performance Optimization

Optimize model inference and frame processing for faster real-time recognition.

### 30.11 Fairness Evaluation

Evaluate recognition behavior across more diverse datasets and demographic groups.

### 30.12 Deployment

The application could be extended for controlled cloud or enterprise deployment after appropriate security, privacy, scalability, and compliance work.

---

## 31. Design Decisions

### Why YuNet?

YuNet is used for the face-detection stage because the project needs a dedicated detector to locate faces before generating embeddings.

### Why SFace?

SFace is used to generate face embeddings that can represent facial characteristics numerically for similarity-based recognition.

### Why Cosine Similarity?

Cosine similarity provides a direct way to compare embedding vectors and determine how similar two facial representations are.

### Why Threshold-Based Rejection?

Without a threshold, the system could assign the closest registered identity even when the similarity is too low.

Threshold-based rejection allows the system to return:

```text
Unknown
```

when no registered identity is sufficiently similar.

### Why FastAPI?

FastAPI provides a lightweight Python backend for connecting the web interface with the ML recognition pipeline.

### Why SQLite?

SQLite is sufficient for the project's local user and recognition-history storage without requiring a separate database server.

### Why HTML/CSS/JavaScript?

The frontend uses standard web technologies to provide a lightweight browser interface without requiring a React-based frontend.

### Why Pretrained Models?

Using pretrained YuNet and SFace models allows the project to focus on integrating a practical face-recognition pipeline, embedding storage, similarity matching, threshold calibration, unknown rejection, and evaluation without requiring large-scale model training.

---

## 32. Why This Is an AI/ML Project

FaceSense AI contains multiple AI/ML components:

```text
Deep Learning Face Detection
          +
Deep Learning Face Representation
          +
Face Embedding Generation
          +
Similarity-Based Recognition
          +
Threshold Calibration
          +
Evaluation Metrics
          +
Unknown-Person Rejection
```

The project uses pretrained deep-learning models for:

- Face detection
- Face representation

It then builds an application-level recognition system around those models using:

- Face embeddings
- Cosine similarity
- Similarity thresholds
- Unknown rejection
- Evaluation metrics

The project therefore demonstrates practical AI/ML model integration rather than simply implementing a conventional CRUD application.

---

## 33. Challenges and Solutions

### Challenge 1: Unknown-Person Recognition

A system that always selects the closest registered identity can incorrectly identify unknown people.

**Solution:**

A similarity threshold was introduced so that low-similarity matches are rejected as `Unknown`.

---

### Challenge 2: Threshold Selection

Choosing an arbitrary threshold can lead to excessive false acceptance or false rejection.

**Solution:**

Genuine and impostor similarity comparisons were evaluated across different thresholds.

---

### Challenge 3: Multiple Faces

An image may contain several faces that need separate recognition decisions.

**Solution:**

Each detected face is processed independently through embedding generation and similarity matching.

---

### Challenge 4: Image Quality

Lighting, blur, pose, and occlusion can affect recognition.

**Solution:**

The system was tested using multiple failure conditions to understand its behavior and limitations.

---

### Challenge 5: Database Synchronization

Deleting a person requires removing both the stored embedding and user record.

**Solution:**

The person-management workflow updates the face embedding database and SQLite users table.

---

### Challenge 6: Frontend and Backend Communication

The browser interface needs to communicate with the Python ML backend.

**Solution:**

FastAPI REST endpoints connect the JavaScript frontend with the recognition and database services.

---

## 34. Example Workflow

### Registration

```text
Person
   ↓
Face Detection
   ↓
SFace Embedding
   ↓
Store Embedding
   ↓
Register Identity
```

### Recognition

```text
New Face
   ↓
Face Detection
   ↓
SFace Embedding
   ↓
Cosine Similarity
   ↓
Best Match
   ↓
Threshold
   ↓
Identity / Unknown
```

### Unknown Rejection

```text
New Face
   ↓
Embedding
   ↓
Similarity = 0.25
   ↓
Threshold = 0.41
   ↓
0.25 < 0.41
   ↓
Unknown
```

---

## 35. Performance and Practical Considerations

FaceSense AI is designed as a lightweight local application using pretrained ONNX models.

Practical considerations include:

- Model inference can be performed locally.
- No external paid AI API is required.
- The application uses a local SQLite database.
- Face embeddings are stored locally.
- The frontend communicates with the backend through HTTP requests.
- Model and data storage requirements depend on the selected ONNX models and registered identities.

No universal FPS or latency benchmark is claimed because hardware-specific performance measurements are not part of the current evaluation.

---

## 36. Ethical and Responsible Use

Face recognition involves sensitive biometric information.

The system should be used responsibly.

Recommended principles include:

- Obtain appropriate consent before registering individuals.
- Protect biometric information.
- Avoid unauthorized identification or surveillance.
- Restrict access to stored biometric data.
- Do not publicly expose personal face images or embeddings.
- Conduct broader validation before real-world deployment.
- Consider privacy, security, and fairness requirements for production use.

FaceSense AI is intended as an AI/ML internship and portfolio project demonstrating face-recognition system design and implementation.

---

## 37. Conclusion

FaceSense AI demonstrates an end-to-end face identification and unknown-person rejection pipeline using pretrained deep-learning models.

The system combines:

```text
YuNet
  ↓
Face Detection
  ↓
SFace
  ↓
Face Embeddings
  ↓
Cosine Similarity
  ↓
Threshold Calibration
  ↓
Known / Unknown Decision
```

The project extends the ML models with practical application features including:

- Person registration
- Image-based recognition
- Live camera recognition
- Multiple-face recognition
- Unknown-person rejection
- Person deletion
- Recognition history
- SQLite storage
- FastAPI backend
- Threshold evaluation
- Failure-case testing

The evaluation performed during development showed separation between genuine and impostor similarity scores for the available registered identities. The system therefore demonstrates how pretrained computer-vision models can be integrated into a complete AI/ML application with similarity-based recognition and unknown rejection.

The project also identifies important limitations and future improvements required before considering a system for broader real-world deployment.

---

## 38. Author

**Author:** Nikshepa A. S.

**Project:** FaceSense AI

**Purpose:** AI/ML Internship Assignment

---

## 39. Project Summary

```text
Project Name:
FaceSense AI

Domain:
Artificial Intelligence / Machine Learning / Computer Vision

Core Models:
YuNet + SFace

Face Detection:
YuNet

Face Embedding:
SFace

Matching:
Cosine Similarity

Application Threshold:
Approximately 0.41

Evaluation Threshold:
0.40

Backend:
FastAPI

Frontend:
HTML + CSS + JavaScript

Database:
SQLite

Embedding Storage:
Local face embedding database

Unknown Rejection:
Threshold-based

Recognition Modes:
Image + Live Camera

Evaluation:
Precision, Recall, F1, FAR, FRR

Development Evaluation:
23 Genuine Comparisons
55 Impostor Comparisons
```

---

## Disclaimer

FaceSense AI is a development and internship project. The reported evaluation results are specific to the available development data and should not be interpreted as a guarantee of real-world face-recognition performance.

Biometric applications should undergo appropriate privacy, security, fairness, robustness, and compliance evaluation before deployment in real-world environments.
