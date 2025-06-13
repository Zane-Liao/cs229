import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)
    pred_y = model.predict(x_valid)
    mse_lists = []

    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        model.tau = tau

        MSE = np.mean((pred_y - y_valid)**2)
        mse_lists.append(MSE)
        print(f"tau: {tau}, MSE: {MSE}")
        # Plot data...

    # Solution
    tau_output = tau_values[np.argmin(mse_lists)]
    print(f'valid set: lowest MSE={min(mse_lists)}, tau={tau_output}')
    # Fit a LWR model with the best tau value
    model.tau = tau_output

    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    y_pred = model.predict(x_test)
    np.savetxt(pred_path, y_pred)

    MSE = np.mean((y_pred - y_test)**2)
    print(f'test set: tau={tau_output}, MSE={MSE}')

    # *** END CODE HERE ***
