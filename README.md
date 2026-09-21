# FaceSense AI

## Intelligent Face Identification, Registration & Unknown-Person Rejection System

FaceSense AI is a computer-vision based face identification system that can detect faces, generate face embeddings, compare them with registered identities, and reject faces that do not match the registered database.

The project was developed as an end-to-end AI/ML application with a Python backend and a simple web-based frontend. The system focuses on practical face recognition rather than training a deep learning model from scratch.

---

## 1. Project Overview

Face recognition systems need to perform more than simply detecting a face in an image. A practical system should be able to:

- Detect a face from an image or camera frame
- Extract meaningful facial features
- Represent the face as a numerical embedding
- Compare the embedding with registered identities
- Identify a known person
- Reject an unregistered person as unknown
- Handle multiple faces in an image
- Maintain a record of recognition events
- Evaluate similarity scores and select a suitable matching threshold

FaceSense AI implements these steps using pretrained computer vision models combined with a similarity-based matching system.

---

## 2. Objectives

The main objectives of FaceSense AI are:

1. Detect faces from images.
2. Generate facial embeddings for detected faces.
3. Register people using their facial embeddings.
4. Compare new face embeddings with registered embeddings.
5. Identify known individuals using cosine similarity.
6. Reject unknown individuals using a configurable threshold.
7. Support image upload and live camera recognition.
8. Store registered users and recognition history.
9. Evaluate genuine and impostor similarity scores.
10. Analyze different threshold values to determine a suitable operating threshold.

---

## 3. Main Features

### Face Detection

The system uses the YuNet face detection model to locate faces in an image.

### Face Embeddings

The SFace model is used to convert a detected face into a numerical feature representation called a face embedding.

### Person Registration

A user can enter a person's name and provide a face image. The system detects the face, generates its embedding, and stores it in the face database.

### Face Identification

For a new image, the system:

1. Detects the face.
2. Generates its embedding.
3. Compares it with registered embeddings.
4. Finds the highest similarity score.
5. Applies the configured threshold.
6. Returns the identity or marks the face as unknown.

### Unknown-Person Rejection

If the highest similarity score is below the configured threshold, the system does not force an identity. Instead, it returns the result as:

`Unknown`

This is important because a face recognition system should not assign a registered identity to every detected face.

### Multiple-Face Recognition

The system can process multiple detected faces in the same image and return recognition results for the detected faces.

### Live Recognition

The application can access the user's camera and perform face recognition on captured frames.

### Recognition History

Recognition results can be stored in SQLite, including:

- Person name
- Similarity score
- Recognition status
- Timestamp

### Person Management

The system provides registration management functionality that allows users to:

- View registered people
- Search registered people
- Delete a registered identity

When a person is deleted, their stored face embedding is removed from the face database and their registered identity is removed from the SQLite users table.

### Evaluation

The project includes an evaluation script that compares:

- Genuine face comparisons
- Impostor face comparisons

It also calculates:

- Precision
- Recall
- F1-score
- FAR
- FRR
- TP
- TN
- FP
- FN

The evaluation script tests multiple similarity thresholds and selects a suitable operating threshold based on the evaluation results.

---

## 4. Technology Stack

### Programming Language

- Python

### Computer Vision / Machine Learning

- OpenCV
- YuNet
- SFace
- NumPy
- Pandas
- Scikit-learn

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Database

- SQLite

### Data Storage

- Pickle-based face embedding database

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

## 5. Machine Learning Models

FaceSense AI uses two pretrained deep-learning models.

### YuNet — Face Detection

YuNet is used for face detection.

Its responsibility is to determine:

- Whether a face is present
- Where the face is located
- The bounding box of the detected face

The output of the detector is then used as input for the face recognition stage.

### SFace — Face Recognition and Embedding Generation

SFace is used for face recognition and embedding generation.

Instead of directly predicting a person's name, SFace converts the detected face into a numerical representation called a face embedding.

Conceptually:

```text
Face Image
    |
    v
SFace
    |
    v
Face Embedding
    |
    v
Numerical Feature Vector