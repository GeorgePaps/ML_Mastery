import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=1000):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the perceptron.
        X: ndarray of shape (n_samples, n_features)
        y: array of labels (-1 or +1)
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.max_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * target
                if target * (np.dot(xi, self.w) + self.b) <= 0:
                    self.w += update * xi
                    self.b += update
                    errors += 1
            if errors == 0:
                print(f"Converged after {epoch+1} epochs.")
                break

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Returns: array of +1 or -1
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)


# Example usage: OR logic gate
if __name__ == "__main__":
    # Training data for OR gate
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([-1, 1, 1, 1])  # -1 for 0, 1 for 1

    perceptron = Perceptron(learning_rate=0.1, max_epochs=10)
    perceptron.fit(X, y)

    print("Weights:", perceptron.w)
    print("Bias:", perceptron.b)

    # Predictions
    preds = perceptron.predict(X)
    print("Predictions:", preds)
