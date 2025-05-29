from loguru import logger
import numpy as np

def spline_function_factory(x, a, b, c, d):
    def spline(x_new):
        i = np.searchsorted(x, x_new) - 1
        i = np.clip(i, 0, len(x)-2)
        dx = x_new - x[i]
        return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return spline

def inverse_spline_function(spline, x_range):
    def find_x(y_target):
        a, b = x_range
        tol = 1e-6
        def f(x): return spline(x) - y_target
        while abs(b - a) > tol:
            c = (a + b) / 2
            if f(c) == 0:
                return c
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return (a + b) / 2
    return find_x


def monotone_cubic_spline(x, y):
    n = len(x)
    logger.info(f"monotone_cubic_spline : X shape: {n}")
    
    # Calculate slopes between points
    h = np.diff(x)
    delta = np.diff(y) / h
    
    # Initialize tangents
    m = np.zeros(n)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2
    
    # Adjust tangents to maintain monotonicity
    for i in range(1, n-1):
        if delta[i-1] == 0 or delta[i] == 0:
            m[i] = 0
        else:
            alpha = m[i] / delta[i-1]
            beta = m[i] / delta[i]
            if alpha < 0 or beta < 0:
                m[i] = 0
            elif alpha + beta > 3:
                tau = 3 / (alpha + beta)
                m[i] = tau * alpha * delta[i-1]

    m[0] = delta[0] if delta[0] == 0 else (3 * delta[0] - m[1]) / 2
    m[-1] = delta[-1] if delta[-1] == 0 else (3 * delta[-1] - m[-2]) / 2
    
    # Cubic spline coefficients
    a = y[:-1]
    b = m[:-1]
    c = (3 * delta - 2 * m[:-1] - m[1:]) / h
    d = (m[:-1] + m[1:] - 2 * delta) / (h**2)

    model_size = a.nbytes + b.nbytes + c.nbytes + d.nbytes + x.nbytes

    spline = spline_function_factory(x, a, b, c, d)
    inverse_spline = inverse_spline_function(spline, (min(x), max(x)))
    
    return spline, inverse_spline, model_size


def test_monotone_cubic_spline():
    # Example usage
    x = np.array([0, 1, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 1,  1.5, 1.5, 2, 3])

    spline, inv_spline = monotone_cubic_spline(x, y)

    # Evaluate the spline at some new points
    x_new = np.linspace(0, 5, 10)
    y_new = spline(x_new)

    for x, y in zip(x_new, y_new):
        print(f"x: {x} | y: {y}")

    print("\nInverse Spline")
    for y in [0.5, 1.25, 1.75, 2, 2.5]:
        x = inv_spline(y)
        print(f"y: {y} | x: {x}")

    # # Plotting the results
    # import matplotlib.pyplot as plt

    # plt.plot(x, y, 'o', label='Data points')
    # plt.plot(x_new, y_new, '-', label='Monotone cubic spline')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    test_monotone_cubic_spline()