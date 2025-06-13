import matplotlib.pyplot as plt
import numpy as np
import util
import os

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    pred_y = model.predict(x_eval)
    
    MSE = np.mean(np.square(pred_y - y_eval))
    print(f"MSE: {MSE}")

    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    os.makedirs('output', exist_ok=True)
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_eval, pred_y, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b_.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        pred_y = np.zeros(m, dtype=np.float64)
        
        for i in range(m):
            diff = self.x - x[i]
            square = -np.sum(diff ** 2, axis=1)
            weights = np.exp(square / (2 * self.tau ** 2))
            W = np.diag(weights)
            
            pred_y[i] = (
                x[i] @
                np.linalg.pinv(self.x.T @ W @ self.x)
                @ self.x.T 
                @ W 
                @ self.y 
                )

        return pred_y
        # *** END CODE HERE ***
