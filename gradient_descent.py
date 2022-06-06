import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_ackley_function(f):
    """
    Plotting the 3D surface for a given cost function f.
    :param f: The function to optimize
    :return:
    """
    n = 200
    bounds = [-2, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = np.zeros(XX.shape)
    ZZ = f(XX, YY)

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()


def gradient_descent(f, df, x0, y0, learning_rate, lr_decay, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the parameter x by the gradient_x times the learning_rate,
    and y by the gradient_y times the learning_rate
    The function should return the point (x, y) and the list of errors in each iteration in a numpy array.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate:
    :param lr_decay: A number to multiply learning_rate with, in each iteration (choose a value between 0.75 to 1.0)
    :param max_iter: maximum number of iterations
    :return: x, y (solution), E_list (array of errors over iterations)
    """

    # Implement a gradient descent algorithm, with a decaying learning rate
    E_list = np.zeros(max_iter)
    x, y = x0, y0
    iters = 0

    while iters < max_iter:
        E_list[iters] = f(x, y)
        x_gd, y_gd = df(x, y) # receive gradient_ackey values from previous x and y
        x = x - lr_decay*learning_rate*x_gd # update x and y
        y = y - lr_decay*learning_rate*y_gd
        iters = iters + 1

    return x, y, E_list

def ackley(x, y):
    # Implement the cost function specified in the HW1 sheet
    z = -20.0 * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5 * (np.cos(np.pi * 2.0 * x) + np.cos(np.pi * 2.0 * y))) + np.exp(1) + 20
    return z


def gradient_ackey(x, y):
    # Implement gradients of Ackley function w.r.t. x and y
    grad_x = np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) * (1/np.sqrt(x**2 + y**2)) * x  +  np.exp(0.5*(np.cos(np.pi*2.0*x)+np.cos(np.pi*2.0*y))) * np.pi * np.sin(np.pi*2.0*x)
    grad_y = np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) * (1/np.sqrt(x**2 + y**2)) * y  +  np.exp(0.5*(np.cos(np.pi*2.0*x)+np.cos(np.pi*2.0*y))) * np.pi * np.sin(np.pi*2.0*y)
    return grad_x, grad_y