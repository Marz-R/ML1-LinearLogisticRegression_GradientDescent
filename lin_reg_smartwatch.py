import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def pearson_coeff(x, y):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional)
    :return: Pearson coefficient of correlation
    """
    # Implement it yourself, you are allowed to use np.mean, np.sqrt, np.sum.
    r = ((100 * np.sum(x*y)) - (np.sum(x) * np.sum(y))) / np.sqrt((100 * np.sum(np.square(x)) - np.square(np.sum(x))) * (100 * np.sum(np.square(y)) - np.square(np.sum(y))))
    return r

def design_matrix(x, degree): # Simple, and polynomial
    """
    :param x: Feature vector (one-dimensional)
    :param degree: Degree in the polynomial expansion (in the simplest case, degree 1)
    :return: Design matrix of shape (n_samples,  degree + 1). e.g., for degree 1, shape is (n_samples, 2)
    """
    # Hint: use np.power and np.arange
    ones = np.ones(100)
    combine = np.vstack((ones, x))

    for i in range(degree-1):
        combine = np.vstack((combine, x))

    transpose = combine.T

    exp = np.arange(degree+1)
    X = np.power(transpose, exp)

    print(f'Degree: {degree}')
    return X

def design_matrix_multilinear(x): # Multilinear
    """
    :param x: Features (MATRIX), shape (n_samples, n_features)
    :return: Design matrix of shape (n_samples, n_features)
    """
    # Hint: Use np.concatenate or np.stack
    ones = np.ones(100)
    combine = np.vstack((ones, x))

    X = combine.T
    return X    

def scatterplot_and_line(x, y, theta):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional), dependent variable
    :param theta: Coefficients of line that fits the data
    :return:
    """
    # Theta will be an array with two coefficients, representing slope and intercept.
    # In which format is it stored in the theta array? Take care of that when plotting the line.
    # TODO

    plt.scatter(x, y)
    plt.plot(x, theta[1]*x + theta[0])
    plt.show()

    pass

def scatterplot_and_curve(x, y, theta):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional), dependent variable
    :param theta: Coefficients of line that fits the data
    :return:
    """
    # Theta will be an array with coefficients.
    # In which format is it stored in the theta array? Take care of that when plotting.
    # Hint: use np.polyval
    # TODO
    plt.scatter(x, y)
    plt.plot(x, np.polyval(theta, x)) # theta here is z in fit_predict_mse
    plt.show()

    pass

def fit_predict_mse(x, y, degree):
    """
    Use this function for solving the tasks Meaningful relations, No linear relations, and Polynomial regression!!!

    :param x: Variable 1 (Feature vector (one-dimensional))
    :param y: Variable_2 (one-dimensional), dependent variable
    :param degree: Degree in the polynomial expansion (in the simplest case, degree 1)
    :return: Theta - optimal parameters found; mse - Mean Squared Error
    """
    y1 = y[np.newaxis]
    X = design_matrix(x, degree) # TODO create a design matrix (use design_matrix function)
    theta = np.dot(pinv(X), y1.T) # TODO calculate theta using pinv from numpy.linalg (already imported)

    if degree == 1:
        y_pred = theta[1]*x + theta[0]
        scatterplot_and_line(x, y, theta)
    else:
        z = theta[::-1]
        y_pred = np.polyval(z, x)
        scatterplot_and_curve(x, y, z)
    
    mse = np.square(np.subtract(y, y_pred)).mean() # TODO calculate MSE

    return theta, mse

def multilinear_fit_predict_mse(a, b, y):
    """
    Use this function for solving the task Multilinear regression!!!

    :param x: Features (MATRIX), shape (n_samples, n_features)
    :param y: Dependent variable (one-dimensional)
    :return: Theta - optimal parameters found; mse - Mean Squared Error
    """

    x = np.concatenate((a[np.newaxis], b[np.newaxis]))
    y1 = y[np.newaxis]
    X = design_matrix_multilinear(x) # TODO create a design matrix (use design_matrix_multilinear function)
    theta = np.dot(pinv(X), y1.T) # TODO calculate theta using pinv from numpy.linalg (already imported)

    y_pred = theta[2]*b + theta[1]*a + theta[0] # TODO  Predict the value of y
    mse = np.square(np.subtract(y, y_pred)).mean() # TODO calculate MSE
    return theta, mse
