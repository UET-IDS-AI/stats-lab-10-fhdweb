import numpy as np

# Question 1: Joint Gaussian PDF and Marginals

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).
    """

    coefficient = 1 / (
        2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)
    )

    q = (
        ((x - mu_x) ** 2) / (sigma_x ** 2)
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y) ** 2) / (sigma_y ** 2)
    )

    exponent = np.exp(-q / (2 * (1 - rho**2)))

    return coefficient * exponent


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """

    return (
        1 /
        (np.sqrt(2 * np.pi) * sigma_x)
    ) * np.exp(
        -((x - mu_x) ** 2) / (2 * sigma_x**2)
    )


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """

    return (
        1 /
        (np.sqrt(2 * np.pi) * sigma_y)
    ) * np.exp(
        -((y - mu_y) ** 2) / (2 * sigma_y**2)
    )


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix.
    """

    return np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    """
    Numerically approximate integral of joint Gaussian PDF.
    """

    x_values = np.linspace(
        mu_x - 4 * sigma_x,
        mu_x + 4 * sigma_x,
        n
    )

    y_values = np.linspace(
        mu_y - 4 * sigma_y,
        mu_y + 4 * sigma_y,
        n
    )

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]

    total = 0

    for x in x_values:
        for y in y_values:
            total += joint_gaussian_pdf(
                x,
                y,
                mu_x,
                mu_y,
                sigma_x,
                sigma_y,
                rho
            )

    return total * dx * dy

# Question 2: Simulation and Independence

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate samples from jointly Gaussian distribution.
    """

    np.random.seed(seed)

    mean = [mu_x, mu_y]

    covariance = [
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ]

    samples = np.random.multivariate_normal(
        mean,
        covariance,
        size=n
    )

    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    return x_samples, y_samples


def sample_means(x_samples, y_samples):
    """
    Return sample means.
    """

    mean_x = np.mean(x_samples)
    mean_y = np.mean(y_samples)

    return mean_x, mean_y


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return sample covariance matrix.
    """

    return np.cov(x_samples, y_samples)


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """

    correlation_matrix = np.corrcoef(x_samples, y_samples)

    return correlation_matrix[0, 1]


def gaussian_independence_check(rho):
    """
    Check independence for jointly Gaussian variables.
    """

    return bool(rho == 0)


def zero_rho_covariance_check(n=100000):
    """
    Check covariance when rho = 0.
    """

    x, y = generate_joint_gaussian_samples(
        n=n,
        rho=0,
        seed=42
    )

    covariance = np.cov(x, y)[0, 1]

    return bool(abs(covariance) < 0.1)


def nonzero_rho_covariance_check(n=100000):
    """
    Check covariance when rho = 0.6.
    """

    x, y = generate_joint_gaussian_samples(
        n=n,
        rho=0.6,
        seed=42
    )

    covariance = np.cov(x, y)[0, 1]

    expected = 0.6 * 2 * 3

    return bool(abs(covariance - expected) < 0.15)
