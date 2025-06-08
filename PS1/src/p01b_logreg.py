import numpy as np
from scipy.sparse import diags
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression(eps=1e-5)

    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, f'output/p01b_{pred_path[-3]}.png')

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    y_eval = model.predict(x_eval)

    np.savetxt(pred_path, y_eval > 0.5, fmt="%d")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n, dtype=np.float64)

        while True:
            theta_old = np.copy(self.theta)

            h_theta = 1 / (1 + np.exp(-x @ self.theta))

            # H = (x.T * h_theta * (1 - h_theta)).dot(x) / m
            S = np.diag(h_theta*(1 - h_theta))
            H = x.T @ S @ x / m     # @ -> np.dot()

            grad_J = x.T @ (h_theta - y) / m

            self.theta -= np.linalg.inv(H) @ grad_J

            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        return sigmoid(x @ self.theta)
        # *** END CODE HERE ***
