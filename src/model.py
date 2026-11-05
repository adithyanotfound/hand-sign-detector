"""The sign classifier at inference time: a small MLP in plain NumPy."""
import numpy as np

from features import featurize


class SignClassifier:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.classes = [str(c) for c in data["classes"]]
        self.temperature = float(data["temperature"])
        self.threshold = float(data["threshold"])
        self.use_z = bool(data["use_z"])
        self.mean, self.std = data["mean"], data["std"]
        count = int(data["num_layers"])
        self.weights = [data[f"w{i}"] for i in range(count)]
        self.biases = [data[f"b{i}"] for i in range(count)]

    def logits(self, features):
        x = (features - self.mean) / self.std
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ weight + bias, 0.0)
        return x @ self.weights[-1] + self.biases[-1]

    def probabilities(self, features):
        z = self.logits(features) / self.temperature
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        return z / z.sum(axis=-1, keepdims=True)

    def predict_points(self, points):
        """Class probabilities for one hand's 21x3 isotropic landmarks."""
        features = featurize(points)
        if not self.use_z:
            features = features.reshape(-1, 3)[:, :2].ravel()
        return self.probabilities(features)
