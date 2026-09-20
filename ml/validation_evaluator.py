import pickle
import os
import numpy as np
import pandas as pd


DATABASE_FILE = "data/face_database.pkl"
VALIDATION_FILE = "results/validation_data.pkl"


# ---------------------------------------
# Load files
# ---------------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Registration database not found.")
    exit()

if not os.path.exists(VALIDATION_FILE):
    print("❌ Validation data not found.")
    exit()


with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


with open(VALIDATION_FILE, "rb") as file:
    validation_data = pickle.load(file)


names = list(database.keys())


print("\n======================================")
print("FACE SENSE AI - VALIDATION EVALUATION")
print("======================================")

print("\nRegistered identities:")

for name in names:
    print(f"  - {name}")


# ---------------------------------------
# Cosine similarity
# ---------------------------------------

def cosine_similarity(a, b):

    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ---------------------------------------
# Genuine scores
# ---------------------------------------

genuine_scores = []

# ---------------------------------------
# Impostor scores
# ---------------------------------------

impostor_scores = []


# ---------------------------------------
# Evaluate every validation sample
# ---------------------------------------

for true_name, validation_embeddings in validation_data.items():

    print(
        f"\nEvaluating validation samples for: "
        f"{true_name}"
    )

    for validation_embedding in validation_embeddings:

        # --------------------------------
        # Compare against TRUE identity
        # --------------------------------

        genuine_identity_scores = []

        for stored_embedding in database[true_name]:

            score = cosine_similarity(
                validation_embedding,
                stored_embedding
            )

            genuine_identity_scores.append(score)


        # Best match against correct identity
        genuine_score = max(
            genuine_identity_scores
        )

        genuine_scores.append(
            genuine_score
        )


        # --------------------------------
        # Compare against OTHER identities
        # --------------------------------

        other_identity_scores = []


        for other_name in names:

            if other_name == true_name:
                continue


            for stored_embedding in database[other_name]:

                score = cosine_similarity(
                    validation_embedding,
                    stored_embedding
                )

                other_identity_scores.append(score)


        if other_identity_scores:

            # Strongest impostor similarity
            impostor_score = max(
                other_identity_scores
            )

            impostor_scores.append(
                impostor_score
            )


# ---------------------------------------
# Print similarity statistics
# ---------------------------------------

print("\n======================================")
print("VALIDATION SIMILARITY RESULTS")
print("======================================")

print(
    f"Genuine samples  : "
    f"{len(genuine_scores)}"
)

print(
    f"Impostor samples : "
    f"{len(impostor_scores)}"
)


print("\nGenuine scores")

print(
    f"Minimum : "
    f"{min(genuine_scores):.3f}"
)

print(
    f"Maximum : "
    f"{max(genuine_scores):.3f}"
)

print(
    f"Average : "
    f"{np.mean(genuine_scores):.3f}"
)


print("\nImpostor scores")

print(
    f"Minimum : "
    f"{min(impostor_scores):.3f}"
)

print(
    f"Maximum : "
    f"{max(impostor_scores):.3f}"
)

print(
    f"Average : "
    f"{np.mean(impostor_scores):.3f}"
)


# ---------------------------------------
# Threshold evaluation
# ---------------------------------------

thresholds = np.arange(
    0.20,
    0.91,
    0.01
)


results = []


for threshold in thresholds:

    # Genuine accepted
    TP = sum(
        score >= threshold
        for score in genuine_scores
    )

    # Genuine rejected
    FN = sum(
        score < threshold
        for score in genuine_scores
    )

    # Impostor accepted
    FP = sum(
        score >= threshold
        for score in impostor_scores
    )

    # Impostor rejected
    TN = sum(
        score < threshold
        for score in impostor_scores
    )


    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0
    )


    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0
    )


    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )


    FAR = (
        FP / (FP + TN)
        if FP + TN > 0
        else 0
    )


    FRR = (
        FN / (FN + TP)
        if FN + TP > 0
        else 0
    )


    results.append({

        "threshold": round(
            threshold,
            2
        ),

        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "FAR": FAR,
        "FRR": FRR

    })


# ---------------------------------------
# Find best validation threshold
# ---------------------------------------

results_df = pd.DataFrame(results)


best_index = results_df[
    "f1"
].idxmax()


best = results_df.loc[
    best_index
]


# ---------------------------------------
# Print final validation result
# ---------------------------------------

print("\n======================================")
print("THRESHOLD VALIDATION")
print("======================================")

print(
    f"Selected threshold : "
    f"{best['threshold']:.2f}"
)

print(
    f"TP : {int(best['TP'])}"
)

print(
    f"TN : {int(best['TN'])}"
)

print(
    f"FP : {int(best['FP'])}"
)

print(
    f"FN : {int(best['FN'])}"
)

print(
    f"Precision : "
    f"{best['precision']:.3f}"
)

print(
    f"Recall    : "
    f"{best['recall']:.3f}"
)

print(
    f"F1-score  : "
    f"{best['f1']:.3f}"
)

print(
    f"FAR       : "
    f"{best['FAR']:.3f}"
)

print(
    f"FRR       : "
    f"{best['FRR']:.3f}"
)


# ---------------------------------------
# Save results
# ---------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)


results_df.to_csv(
    "results/validation_threshold_results.csv",
    index=False
)


print()
print(
    "✅ Saved:"
)

print(
    "results/validation_threshold_results.csv"
)