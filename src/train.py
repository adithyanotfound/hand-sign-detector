"""Evaluate on people the model has never seen, then train the final sign classifier.

A random train/test split would be meaningless here: the dataset is burst photos of a
few people, so near-identical frames would land on both sides. Every number reported
comes from a person (and, for "other", hand shapes) held out of training.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

from features import featurize
from model import SignClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIGNS = ["hello", "i_love_you", "peace", "thank_you", "yes"]
OTHER = "other"
CLASSES = SIGNS + [OTHER]

# Hand shapes that are none of the signs. The dataset's `ok` folder is two raised
# fingers rather than a real OK sign, so it serves as a negative example too.
OTHER_LETTERS = list("CDFGHIJKLOPQRUWXYZ")
# Never trained on. V is the same handshape as `peace`, which gives an outside check.
# The fists resemble `yes`, and B resembles `hello`/`thank_you`, so their labels are ambiguous.
EXTERNAL_PEACE = "V"
FIST_LETTERS = list("AEMNST")

# One representative image per outfit cluster, assigned to a person by eye. Merging
# outfits that might be different people can only make the test stricter.
OUTFIT_ANCHORS = [
    ("yes", 13, "A"), ("hello", 11, "A"), ("thank_you", 22, "A"),
    ("peace", 243, "B"), ("peace", 51, "B"), ("peace", 69, "B"),
    ("i_love_you", 44, "C"),
]
NUM_OUTFITS = len(OUTFIT_ANCHORS)
SEEDS = range(5)
# From the table train.py prints. Thresholds rejected few unfamiliar shapes, so this is
# set by what looks trustworthy on screen, not to chase a slightly higher score at 0.43.
LIVE_THRESHOLD = 0.60
LETTER_FOLDS = {"A": OTHER_LETTERS[0::3], "B": OTHER_LETTERS[1::3], "C": OTHER_LETTERS[2::3]}


def assign_people(data):
    mask = data["sources"] == "gestures"
    names, numbers = data["names"][mask], data["numbers"][mask]
    clusters = AgglomerativeClustering(n_clusters=NUM_OUTFITS, linkage="ward").fit_predict(data["torso"][mask])
    mapping = {}
    for name, number, person in OUTFIT_ANCHORS:
        mapping[int(clusters[(names == name) & (numbers == number)][0])] = person
    if len(mapping) != NUM_OUTFITS:
        raise SystemExit("Outfit anchors no longer identify distinct clusters.")
    people = np.full(len(data["names"]), "", dtype="<U1")
    people[mask] = [mapping[int(c)] for c in clusters]
    return people


def augment(points, rng, flip=True):
    """Random mirror (so either hand works), in-plane tilt, and landmark jitter."""
    points = points.copy()
    centred = points - points[:, :1]
    if flip:
        centred[:, :, 0] *= rng.choice([-1.0, 1.0], size=(len(points), 1))
    angle = np.radians(rng.uniform(-20, 20, size=len(points)))
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    x, y = centred[:, :, 0].copy(), centred[:, :, 1].copy()
    centred[:, :, 0], centred[:, :, 1] = cos * x - sin * y, sin * x + cos * y
    span = np.linalg.norm(centred[:, :, :2], axis=2).max(axis=1)[:, None, None]
    return centred + rng.normal(0, 0.008, size=centred.shape) * span


def featurize_all(points, use_z=True):
    features = np.stack([featurize(p) for p in points])
    if not use_z:
        features = features.reshape(-1, 21, 3)[:, :, :2].reshape(len(features), -1)
    return features


def balanced_sample(labels, people, rng, per_class):
    """Draw equally from each class and, inside it, from each person or letter."""
    chosen = []
    for label in range(len(CLASSES)):
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        groups = np.unique(people[idx])
        counts = {g: (people[idx] == g).sum() for g in groups}
        weights = np.array([1.0 / (len(groups) * counts[people[i]]) for i in idx])
        chosen.append(rng.choice(idx, size=per_class, replace=True, p=weights / weights.sum()))
    return np.concatenate(chosen)


def fit(points, labels, groups, make_model, use_z, seed=0, per_class=1500):
    rng = np.random.default_rng(seed)
    idx = balanced_sample(labels, groups, rng, per_class)
    x = featurize_all(augment(points[idx], rng), use_z)
    mean, std = x.mean(0), x.std(0) + 1e-6
    model = make_model(seed)
    model.fit((x - mean) / std, labels[idx])
    return model, mean, std


def logits_of(model, x):
    if isinstance(model, LogisticRegression):
        return model.decision_function(x)
    hidden = x
    for weight, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        hidden = np.maximum(hidden @ weight + bias, 0)
    return hidden @ model.coefs_[-1] + model.intercepts_[-1]


def softmax(z):
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    def nll(t):
        return -np.log(softmax(logits / t)[np.arange(len(labels)), labels] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(0.3, 20), method="bounded").x)


def expected_calibration_error(probabilities, labels, bins=10):
    confidence, correct = probabilities.max(1), probabilities.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            error += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(error)


def decide(probabilities, threshold):
    """Final decision: a sign only if it beats `other` and clears the threshold."""
    predicted = probabilities.argmax(1)
    predicted = np.where(probabilities.max(1) < threshold, CLASSES.index(OTHER), predicted)
    return predicted


CANDIDATES = {
    "logistic": (lambda s: LogisticRegression(max_iter=600, C=1.0, random_state=s), True),
    "mlp32": (lambda s: MLPClassifier((32,), alpha=1e-3, max_iter=60, random_state=s), True),
    "mlp64-32": (lambda s: MLPClassifier((64, 32), alpha=1e-3, max_iter=60, random_state=s), True),
    "mlp64-32-noz": (lambda s: MLPClassifier((64, 32), alpha=1e-3, max_iter=60, random_state=s), False),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(ROOT, "data/landmarks.npz"))
    parser.add_argument("--out", default=os.path.join(ROOT, "models/sign_classifier.npz"))
    parser.add_argument("--report", default=os.path.join(ROOT, "models/report.json"))
    args = parser.parse_args()

    started = time.time()
    data = dict(np.load(args.data))
    people = assign_people(data)
    sources, names, detected = data["sources"], data["names"], data["detected"]
    is_gesture, is_alpha = sources == "gestures", sources == "alphabet"

    is_other_letter = is_alpha & np.isin(names, OTHER_LETTERS)
    is_ok = is_gesture & (names == "ok")
    is_sign = is_gesture & np.isin(names, SIGNS)
    usable = detected & (is_sign | is_ok | is_other_letter)

    labels = np.full(len(names), -1)
    for i, sign in enumerate(SIGNS):
        labels[is_sign & (names == sign)] = i
    labels[is_ok | is_other_letter] = CLASSES.index(OTHER)

    # Each group identifies a person, or a letter for the alphabet images.
    groups = np.where(is_alpha, names, people)
    letter_fold = {letter: fold for fold, letters in LETTER_FOLDS.items() for letter in letters}
    fold_of = np.array([
        letter_fold.get(n, "") if a else p for n, a, p in zip(names, is_alpha, people)
    ])

    print("images per person (columns: " + ", ".join(SIGNS) + ", ok)")
    for person in "ABC":
        counts = [int(((people == person) & usable & (names == n)).sum()) for n in SIGNS + ["ok"]]
        print(f"  person {person}: {counts}")

    points = data["points"]
    results = {}
    held_out = {}
    sign_rows = usable & is_sign
    for name, (make_model, use_z) in CANDIDATES.items():
        runs, macros, person_scores = [], [], {p: [] for p in "ABC"}
        for seed in SEEDS:
            all_logits = np.zeros((len(names), len(CLASSES)))
            for fold in "ABC":
                train = usable & (fold_of != fold)
                test = usable & (fold_of == fold)
                model, mean, std = fit(points[train], labels[train], groups[train], make_model, use_z, seed)
                x = featurize_all(points[test], use_z)
                all_logits[test] = logits_of(model, (x - mean) / std)
            runs.append(all_logits[usable])
            predicted = all_logits[sign_rows].argmax(1)
            macros.append(f1_score(labels[sign_rows], predicted, labels=range(len(SIGNS)), average="macro", zero_division=0))
            for p in "ABC":
                rows = people[sign_rows] == p
                person_scores[p].append((predicted[rows] == labels[sign_rows][rows]).mean())
        held_out[name] = runs
        results[name] = {
            "macro_f1": float(np.mean(macros)), "macro_f1_std": float(np.std(macros)),
            "per_person": {p: float(np.mean(v)) for p, v in person_scores.items()},
        }
        print(f"{name:14s} macro-F1 {np.mean(macros):.3f} +/- {np.std(macros):.3f}  per-person "
              + " ".join(f"{p}={np.mean(v):.3f}" for p, v in person_scores.items()))

    best = max(results, key=lambda n: results[n]["macro_f1"])
    make_model, use_z = CANDIDATES[best]
    print("selected:", best)

    y = labels[usable]
    pooled = np.concatenate(held_out[best])
    pooled_y = np.tile(y, len(SEEDS))
    nll_temperature = fit_temperature(pooled, pooled_y)
    print("\ncalibration error by temperature:")
    for t in (1.0, 1.5, 2.0, nll_temperature):
        print(f"   T={t:.2f}: {expected_calibration_error(softmax(pooled / t), pooled_y):.3f}")
    temperature = min(
        np.arange(1.0, 4.01, 0.1),
        key=lambda t: expected_calibration_error(softmax(pooled / t), pooled_y),
    )
    temperature = float(temperature)
    print(f"using T={temperature:.2f} (lowest calibration error on held-out people)")

    sign_mask = np.isin(y, range(len(SIGNS)))
    letter_mask = is_other_letter[usable]
    other_id = CLASSES.index(OTHER)

    def operating_point(threshold):
        accuracy, false_accept = [], []
        for run in held_out[best]:
            d = decide(softmax(run / temperature), threshold)
            accuracy.append((d[sign_mask] == y[sign_mask]).mean())
            false_accept.append((d[letter_mask] != other_id).mean())
        return np.mean(accuracy), np.std(accuracy), np.mean(false_accept), np.std(false_accept)

    print("\nthreshold  signs named correctly  unseen shapes wrongly named")
    for threshold in np.arange(0.40, 0.96, 0.05):
        a, a_sd, f, f_sd = operating_point(threshold)
        print(f"   {threshold:.2f}        {a:.3f} +/- {a_sd:.3f}        {f:.3f} +/- {f_sd:.3f}")

    best_threshold = LIVE_THRESHOLD
    decisions = decide(softmax(held_out[best][0] / temperature), best_threshold)

    print(f"\nHeld-out performance at threshold {best_threshold:.2f}:")
    matrix = confusion_matrix(y, decisions, labels=range(len(CLASSES)))
    width = max(len(c) for c in CLASSES)
    print(" " * (width + 1) + " ".join(f"{c[:6]:>6}" for c in CLASSES))
    for name, row in zip(CLASSES, matrix):
        print(f"{name:<{width}} " + " ".join(f"{v:6d}" for v in row))
    recall = {c: float(matrix[i, i] / max(matrix[i].sum(), 1)) for i, c in enumerate(CLASSES)}
    sign_accuracy, sign_sd, false_accept, false_sd = (float(v) for v in operating_point(best_threshold))
    print(f"signs correctly named: {sign_accuracy:.3f} +/- {sign_sd:.3f} (over {len(SEEDS)} seeds)")
    print(f"unseen non-sign hand shapes wrongly named as a sign: {false_accept:.3f} +/- {false_sd:.3f}")
    for letter in OTHER_LETTERS:
        mask = names[usable] == letter
        rate = (decisions[mask & letter_mask] != CLASSES.index(OTHER)).mean()
        top = CLASSES[int(np.bincount(decisions[mask & letter_mask], minlength=len(CLASSES)).argmax())]
        print(f"   {letter}: false-accept {rate:.2f}  most often: {top}")

    # Final model: all usable data.
    final, mean, std = fit(points[usable], labels[usable], groups[usable], make_model, use_z)
    if isinstance(final, LogisticRegression):
        raise SystemExit(f"{best} was selected, but export only supports MLPs.")
    arrays = {f"w{i}": w.astype(np.float32) for i, w in enumerate(final.coefs_)}
    arrays.update({f"b{i}": b.astype(np.float32) for i, b in enumerate(final.intercepts_)})
    np.savez(
        args.out,
        classes=np.array(CLASSES),
        temperature=temperature,
        threshold=best_threshold,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        num_layers=len(final.coefs_),
        use_z=use_z,
        **arrays,
    )

    # Outside check with the exported file: different people and setting, never trained on.
    classifier = SignClassifier(args.out)
    def external(letters):
        mask = is_alpha & detected & np.isin(names, letters)
        probabilities = classifier.probabilities(featurize_all(points[mask], use_z))
        return decide(probabilities, best_threshold), mask.sum()
    v_decisions, v_count = external([EXTERNAL_PEACE])
    v_rate = float((v_decisions == CLASSES.index("peace")).mean())
    print(f"\nOutside check: letter V (same shape as peace) named `peace`: {v_rate:.3f} of {v_count}")
    fist_decisions, fist_count = external(FIST_LETTERS)
    fist = {c: float((fist_decisions == i).mean()) for i, c in enumerate(CLASSES)}
    print(f"Fist letters A,E,M,N,S,T ({fist_count}), final model: " + ", ".join(f"{c} {v:.2f}" for c, v in fist.items()))

    # Latency and size of the classifier alone.
    sample = featurize_all(points[usable][:1], use_z)
    tick = time.perf_counter()
    for _ in range(2000):
        classifier.probabilities(sample)
    micros = (time.perf_counter() - tick) / 2000 * 1e6
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nclassifier: {size_kb:.1f} KB, {micros:.0f} microseconds per prediction")
    print(f"total time {time.time() - started:.0f}s")

    with open(args.report, "w") as f:
        json.dump({
            "candidates": results, "selected": best, "temperature": temperature,
            "threshold": best_threshold, "sign_accuracy": sign_accuracy, "recall": recall,
            "false_accept_unseen_shapes": false_accept, "external_v_as_peace": v_rate,
            "fist_letters": fist, "confusion_matrix": matrix.tolist(), "classes": CLASSES,
            "classifier_kb": size_kb, "classifier_microseconds": micros,
        }, f, indent=2)


if __name__ == "__main__":
    main()
