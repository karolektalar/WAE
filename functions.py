import random
import numpy as np
import math
import sys
import main


def sum_function(point, add_uncertainty, value_uncertainty_value):
    result = np.sum(point)
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


def square_two_dim_function(point, add_uncertainty, value_uncertainty_value):
    result = -pow(point[0], 2) - pow(point[1], 2) + point[0] + point[1] + 10
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


def random_high_dimensional_function(point, add_uncertainty, value_uncertainty_value):
    result = -pow(point[0], 5) + 1 / 100 * pow(point[1], 5) + pow(point[2], 4) + pow(point[3], 4) - 10 * pow(point[5],
                                                                                                             2) * \
             point[6] * point[7] * point[8] * math.sin(point[9])
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


# https://www.researchgate.net/publication/27382766_On_benchmarking_functions_for_genetic_algorithm
# next functions are from this paper


def f1(point, add_uncertainty, value_uncertainty_value):
    result = pow(point[0], 2) + pow(point[1], 2)
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


def f2(point, add_uncertainty, value_uncertainty_value):
    result = (100 * pow(pow(point[0], 2) - point[1], 2)) + pow(1 - point[0], 2)
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


def f4(point, add_uncertainty, value_uncertainty_value):
    result = 0
    for i in range(len(point)):
        result += i * pow(point[i], 4) + random.gauss(0, 1)
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


def f6(point, add_uncertainty, value_uncertainty_value):
    result = 41898.29101
    for i in range(len(point)):
        result += -point[i] * math.sin(math.sqrt(math.fabs(point[i])))
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


# funcitons from paper https://link.springer.com/chapter/10.1007/978-3-319-07173-2_32

def f7(point, add_uncertainty, value_uncertainty_value):
    result = np.sum(np.abs(point)) + np.prod(np.abs(point))
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


def f8(point, add_uncertainty, value_uncertainty_value):
    result = 0
    for i in range(len(point)):
        result += (point[i] + .5) ** 2
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result


def f9(point, add_uncertainty, value_uncertainty_value):
    result = 0
    for i in range(len(point)):
        result += -point[i] * np.sin(np.sqrt(np.abs(point[i])))
    if add_uncertainty:
        return main.uncertainty_on_value(result, value_uncertainty_value)
    else:
        return result
