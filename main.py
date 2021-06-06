import random
import numpy as np
import sys
import functions

GAUSSIAN_SIGMA = 5
SELECTION_SIZE = 10
POINTS_MIN = -100
POINTS_MAX = 100


def evolutionary_algorithm(sample, number_of_epochs, evaluation_function, uncertainty_on_values,
                           uncertainty_on_arguments):
    best_result = sys.float_info.max
    while number_of_epochs > 0:
        sample = tournament_selection_and_mutation(sample, evaluation_function, SELECTION_SIZE, uncertainty_on_values,
                                                   uncertainty_on_arguments)
        number_of_epochs -= 1
        for i in range(len(sample)):
            value = evaluation_function(sample[i], False)
            if value < best_result:
                best_result = value
    return best_result


def create_sample(sample_size, function_dimension_size, min, max):
    points = np.zeros(shape=(sample_size, function_dimension_size))
    for i in range(sample_size):
        point = points[i]
        for j in range(function_dimension_size):
            point[j] = random.uniform(min, max)
        points[i] = point
    return points


def tournament_selection_and_mutation(sample, evaluation_function, selection_size, uncertainty_on_values,
                                      uncertainty_on_arguments):
    new_sample = np.zeros(shape=(len(sample), sample.shape[1]))
    for i in range(len(sample)):
        tmp_points = np.zeros(shape=(selection_size, sample.shape[1]))
        tmp_values = np.zeros(shape=selection_size)
        for j in range(selection_size):
            tmp_points[j] = sample[random.randint(0, len(sample) - 1)]
            if uncertainty_on_arguments:
                tmp_values[j] = evaluation_function(
                    gaussian_uncertainty(tmp_points[j], GAUSSIAN_UNCERTAINTY_ON_ARGUMENTS_SIGMA), uncertainty_on_values)
            else:
                tmp_values[j] = evaluation_function(tmp_points[j], uncertainty_on_values)
        new_sample[i] = mutation(tmp_points[np.argmin(tmp_values)])
    return new_sample


def mutation(point):
    for i in range(len(point)):
        point[i] = random.gauss(point[i], GAUSSIAN_SIGMA)
    return (point)


def gaussian_uncertainty(point, sigma):
    for i in range(len(point)):
        point[i] = random.gauss(point[i], sigma)
    return (point)


def uncertainty_on_value(value):
    return random.gauss(value, GAUSSIAN_UNCERTAINTY_ON_VALUES_SIGMA)


def run_functions(uncertainty_on_values, uncertainty_on_arguments):
    if uncertainty_on_values:
        print("Wyniki z niepewnością na wartościach funkcji")
    if uncertainty_on_arguments:
        print("Wyniki z niepewnością na argumentach funkcji")
    if not uncertainty_on_values and not uncertainty_on_arguments:
        print("Wyniki bez niepewności")
    sample = create_sample(100, 10, -10, 10)
    print(evolutionary_algorithm(sample, 100, functions.sum_function, uncertainty_on_values, uncertainty_on_arguments))

    sample = create_sample(100, 2, -10, 10)
    print(evolutionary_algorithm(sample, 100, functions.square_two_dim_function, uncertainty_on_values, uncertainty_on_arguments))

    sample = create_sample(100, 10, -1, 1)
    print(evolutionary_algorithm(sample, 100, functions.random_high_dimensional_function, uncertainty_on_values,
                                 uncertainty_on_arguments))
    sample = create_sample(100, 2, -5.12, 5.12)
    print(evolutionary_algorithm(sample, 100, functions.f1, uncertainty_on_values,
                                 uncertainty_on_arguments))
    sample = create_sample(100, 2, -2.048, 2.048)
    print(evolutionary_algorithm(sample, 100, functions.f2, uncertainty_on_values,
                                 uncertainty_on_arguments))
    sample = create_sample(100, 30, -1.28, 1.28)
    print(evolutionary_algorithm(sample, 100, functions.f4, uncertainty_on_values,
                                 uncertainty_on_arguments))
    sample = create_sample(100, 10, -500, 500)
    print(evolutionary_algorithm(sample, 100, functions.f6, uncertainty_on_values,
                                 uncertainty_on_arguments))


if __name__ == '__main__':
    random.seed(10)
    Values_Array = [0.1, 1, 10, 200]
    Arguments_Array = [0.1, 1, 10, 200]

    for value in Values_Array:
        GAUSSIAN_UNCERTAINTY_ON_VALUES_SIGMA = value
        for argument in Arguments_Array:
            GAUSSIAN_UNCERTAINTY_ON_ARGUMENTS_SIGMA = argument
            print("Sigma niepewności na argumentach : " + str(argument) + " niepewności na wartościach: " + str(value))
            run_functions(False, False)
            run_functions(True, False)
            run_functions(False, True)
            run_functions(True, True)
