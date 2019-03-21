import numpy as np
import matplotlib.pyplot as plt


def gradient_limit_amplitude(grad, threshold):
    return threshold*(1 - 2/(1 + np.exp(2*grad/threshold)))


def gradient_one_direction(f, point, direction):
    """
    f: input function, input dimension is m, output dimension is 1,
    point: numpy.array, point to calculate gradient, dimension of point is m,
    direction: a integer ranges from 0 to m-1, represents direction to calculate gradient
    return one dimension gradient
    """
    epsilon = 1e-3
    m = point.shape[0]
    small_vector = np.zeros(m)
    small_vector[direction] = epsilon
    f_x = f(point + small_vector)
    delta_f = f_x - f(point)
    grad = delta_f/epsilon
    return grad


def gradient(f, point):
    """
    f: input function, input dimension is m, output dimension is 1,
    point: numpy.array, point to calculate gradient, dimension of point is m
    return gradient vector
    """
    m = point.shape[0]
    grad = [gradient_one_direction(f, point, i) for i in range(m)]
    grad = np.array(grad)
    return grad


def gradient_descent(obj_function, initial_point, step, error):
    grad = gradient(obj_function, initial_point)
    x = initial_point.astype(np.float64)
    i = 0
    max_grad = np.abs(initial_point).max()
    error = abs(error)
    while np.abs(grad).max() > error:
        grad = gradient(obj_function, x)
        grad = gradient_limit_amplitude(grad, max_grad)
        x -= grad*step
        i += 1
    return x, i


if __name__ == "__main__":
    p0 = np.array([20.0])
    f = lambda x: np.power(x[0] - 2, 2)
    x, i = gradient_descent(f, initial_point=p0, step=.5, error=1e-3)
    loss = f(x)
    print("x: ", x,
          "loss: ", loss,
          "i: ", i)
