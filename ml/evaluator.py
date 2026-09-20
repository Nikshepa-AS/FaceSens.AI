import pickle
import os
import numpy as np
import pandas as pd


DATABASE_FILE = "data/face_database.pkl"


# ----------------------------------------
# Load database
# ----------------------------------------

if not os.path.exists(DATABASE_FILE):
    print("❌ Face database not found.")
    exit()

with open(DATABASE_FILE, "rb") as file:
    database = pickle.load(file)


names = list(database.keys())

if len(names) < 2:
    print("❌ At least 2 registered people are required.")
    exit()


print("Registered identities:")
for name in names:
    print(f"  - {name}")


# ----------------------------------------
# Cosine similarity
# ----------------------------------------

def cosine_similarity(a, b):

    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


# ----------------------------------------
# Create genuine and impostor scores
# ----------------------------------------

genuine_scores = []
impostor_scores = []


for name in names:

    embeddings = database[name]

    # Genuine comparisons
    for i in range(len(embeddings)):

        for j in range(i + 1, len(embeddings)):

            score = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            genuine_scores.append(score)


# Compare people from different identities

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        person_a = names[i]
        person_b = names[j]

        for emb_a in database[person_a]:

            for emb_b in database[person_b]:

                score = cosine_similarity(
                    emb_a,
                    emb_b
                )

                impostor_scores.append(score)


print()
print("======================================")
print("SIMILARITY DATASET")
print("======================================")

print(f"Genuine comparisons  : {len(genuine_scores)}")
print(f"Impostor comparisons : {len(impostor_scores)}")

print()
print(
    f"Genuine mean  : "
    f"{np.mean(genuine_scores):.3f}"
)

print(
    f"Genuine min   : "
    f"{np.min(genuine_scores):.3f}"
)

print(
    f"Genuine max   : "
    f"{np.max(genuine_scores):.3f}"
)

print()
print(
    f"Impostor mean : "
    f"{np.mean(impostor_scores):.3f}"
)

print(
    f"Impostor min  : "
    f"{np.min(impostor_scores):.3f}"
)

print(
    f"Impostor max  : "
    f"{np.max(impostor_scores):.3f}"
)


# ----------------------------------------
# Evaluate thresholds
# ----------------------------------------

thresholds = np.arange(
    0.20,
    0.91,
    0.01
)


results = []


for threshold in thresholds:

    # Genuine correctly accepted
    true_positive = sum(
        score >= threshold
        for score in genuine_scores
    )

    # Genuine incorrectly rejected
    false_negative = sum(
        score < threshold
        for score in genuine_scores
    )

    # Impostor incorrectly accepted
    false_positive = sum(
        score >= threshold
        for score in impostor_scores
    )

    # Impostor correctly rejected
    true_negative = sum(
        score < threshold
        for score in impostor_scores
    )


    precision = (
        true_positive /
        (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0
    )


    recall = (
        true_positive /
        (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0
    )


    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall)
        else 0
    )


    far = (
        false_positive /
        (false_positive + true_negative)
        if (false_positive + true_negative)
        else 0
    )


    frr = (
        false_negative /
        (false_negative + true_positive)
        if (false_negative + true_positive)
        else 0
    )


    results.append({
        "threshold": round(threshold, 2),
        "TP": true_positive,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "FAR": far,
        "FRR": frr
    })


# ----------------------------------------
# Find threshold using F1
# ----------------------------------------

results_df = pd.DataFrame(results)

best_index = results_df["f1"].idxmax()

best_result = results_df.loc[best_index]


print()
print("======================================")
print("THRESHOLD EVALUATION")
print("======================================")

print(
    f"Selected threshold: "
    f"{best_result['threshold']:.2f}"
)

print(
    f"Precision : "
    f"{best_result['precision']:.3f}"
)

print(
    f"Recall    : "
    f"{best_result['recall']:.3f}"
)

print(
    f"F1-score  : "
    f"{best_result['f1']:.3f}"
)

print(
    f"FAR       : "
    f"{best_result['FAR']:.3f}"
)

print(
    f"FRR       : "
    f"{best_result['FRR']:.3f}"
)


# ----------------------------------------
# Save results
# ----------------------------------------

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/threshold_results.csv",
    index=False
)

print()
print("✅ Results saved to:")
print("results/threshold_results.csv")
# ----------------------------------------
# Save evaluation summary
# ----------------------------------------

import json

summary = {
    "registered_identities": names,
    "genuine_comparisons": len(genuine_scores),
    "impostor_comparisons": len(impostor_scores),

    "genuine_mean": float(np.mean(genuine_scores)),
    "genuine_min": float(np.min(genuine_scores)),
    "genuine_max": float(np.max(genuine_scores)),

    "impostor_mean": float(np.mean(impostor_scores)),
    "impostor_min": float(np.min(impostor_scores)),
    "impostor_max": float(np.max(impostor_scores)),

    "selected_threshold": float(
        best_result["threshold"]
    ),

    "precision": float(
        best_result["precision"]
    ),

    "recall": float(
        best_result["recall"]
    ),

    "f1_score": float(
        best_result["f1"]
    ),

    "far": float(
        best_result["FAR"]
    ),

    "frr": float(
        best_result["FRR"]
    ),

    "true_positive": int(
        best_result["TP"]
    ),

    "true_negative": int(
        best_result["TN"]
    ),

    "false_positive": int(
        best_result["FP"]
    ),

    "false_negative": int(
        best_result["FN"]
    )
}

with open(
    "results/evaluation_summary.json",
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print(
    "results/evaluation_summary.json"
)