import random
import numpy as np

GAUSSIAN_SIGMA = 5
GAUSSIAN_UNCERTAINTY_SIGMA = 5
SELECTION_SIZE = 5
POINTS_MIN = -100
POINTS_MAX = 100
RUN_WITH_UNCERTAINTY_ON_ARGUMENTS = True
RUN_WITH_UNCERTAINTY_ON_VALUES = True


def evolutionary_algorithm(sample, number_of_epochs, evaluation_function):
    best_result = 0
    while number_of_epochs > 0:
        sample = tournament_selection_and_mutation(sample, evaluation_function, SELECTION_SIZE)
        number_of_epochs -= 1
        for i in range(len(sample)):
            value = evaluation_function(sample[i], False)
            if value > best_result:
                best_result = value
    return best_result


def create_sample(sample_size, function_dimension_size):
    points = np.zeros(shape=(sample_size, function_dimension_size))
    for i in range(sample_size):
        point = points[i]
        for j in range(function_dimension_size):
            point[j] = random.uniform(POINTS_MIN, POINTS_MAX)
        points[i] = point
    return points


def tournament_selection_and_mutation(sample, evaluation_function, selection_size):
    new_sample = np.zeros(shape=(len(sample), sample.shape[1]))
    for i in range(len(sample)):
        tmp_points = np.zeros(shape=(selection_size, sample.shape[1]))
        tmp_values = np.zeros(shape=selection_size)
        for j in range(selection_size):
            tmp_points[j] = sample[random.randint(0, len(sample) - 1)]
            if RUN_WITH_UNCERTAINTY_ON_ARGUMENTS:
                tmp_values[j] = evaluation_function(gaussian_uncertainty(tmp_points[j]), RUN_WITH_UNCERTAINTY_ON_VALUES)
            else:
                tmp_values[j] = evaluation_function(tmp_points[j], RUN_WITH_UNCERTAINTY_ON_VALUES)
        new_sample[i] = mutation(tmp_points[np.argmax(tmp_values)])
    return new_sample


def mutation(point):
    for i in range(len(point)):
        point[i] = random.gauss(point[i], GAUSSIAN_SIGMA)
    return (point)


def gaussian_uncertainty(point):
    for i in range(len(point)):
        point[i] = random.gauss(point[i], GAUSSIAN_UNCERTAINTY_SIGMA)
    return (point)


def uncertainty_on_value(value):
    return random.gauss(value, GAUSSIAN_UNCERTAINTY_SIGMA)


def sum_function(point, add_uncertainty):
    result = np.sum(point)
    if add_uncertainty:
        return uncertainty_on_value(result)
    else:
        return result


def square_two_dim_function(point, add_uncertainty):
    result = -pow(point[0], 2) - pow(point[1], 2) + point[0] + point[1] + 10
    if add_uncertainty:
        return uncertainty_on_value(result)
    else:
        return result


if __name__ == '__main__':
    random.seed(10)
    sample = create_sample(100, 2)
    print(evolutionary_algorithm(sample, 100, square_two_dim_function))

    # sample = create_sample(100, 10)
    # print(evolutionary_algorithm(sample, 100, sum_function))
