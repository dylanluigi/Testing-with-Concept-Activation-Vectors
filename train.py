"""Module containing the attribution functions used in the AIXI method.

Written by: Miquel Miró Nicolau (UIB), 2022
"""

import autograd.numpy as np


def ssin_term(weight, x):
    """Individual term for the ssin function

    Args:
        weight: Float, between 0 and 1 for the weight of the term
        x: Float, input data

    Returns:
        Value of the term: weight * sin(pi * (x / 2))
    """
    return weight * np.sin(np.pi * (x / 2))


def ssin(x1=0, x2=0, x3=0):
    """Computes the SSIM of the label.

    Math form:
        1/2*sin(x1*pi/2) + 1/4*sin(x2*pi/2) + 1/6*sin(x3*pi/2)

    Args:
        x1:
        x2:
        x3:

    Returns:

    """
    y = ssin_term(1 / 2, x1) + ssin_term(1 / 4, x2) + ssin_term(1 / 6, x3)
    y = np.array([y]).astype(np.float32)

    return y


def ssum_term(weight, x):
    """Individual term for the ssum function

    Args:
        weight: Float, between 0 and 1 for the weight of the term
        x: Float, input data

    Returns:
        Value of the term: weight * x
    """
    return weight * x


def ssum(x1=0, x2=0, x3=0):
    """Computes the SSUM of the label.

    Math form:
        1/2*x1 + 1/4*x2 + 1/6*x3

    Args:
        x1:
        x2:
        x3:

    Returns:

    """
    y = ssum_term(1 / 2, x1) + ssum_term(1 / 4, x2) + ssum_term(1 / 6, x3)
    y = np.array([y]).astype(np.float32)

    return y


def discrete(x1=0, x2=0, x3=0):
    """Computes the discrete version of the label.

    Args:
        label:

    Returns:

    """
    y = ((1 * x1) - (0.5 * x2) + (0.25 * x3)) >= 0
    y = np.array([y]).astype(np.float32)

    return y


def seneca(label):
    return float(label > 0)


def circle_only(x1=0, x2=0, x3=0):
    """Binary circle detector: 1 if at least one circle present, 0 otherwise.

    Args:
        x1: Number of circles
        x2: Number of squares (unused)
        x3: Number of crosses (unused)

    Returns:
        numpy array [1.0] if x1 > 0, else [0.0]
    """
    y = np.array([float(x1 > 0)]).astype(np.float32)
    return y


attr_functions = {
    "ssin": ssin,
    "ssum": ssum,
    "discrete": discrete,
    "circle_only": circle_only,
}
