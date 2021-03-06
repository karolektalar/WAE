import random
import numpy as np
import sys
import functions
from typing import Callable
import csv

GAUSSIAN_SIGMA = 5
SELECTION_SIZE = 10
POINTS_MIN = -100
POINTS_MAX = 100


def evolutionary_algorithm(sample, number_of_epochs, evaluation_function, uncertainty_on_values,
                           uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value):
    best_result = sys.float_info.max
    while number_of_epochs > 0:
        sample = tournament_selection_and_mutation(sample, evaluation_function, SELECTION_SIZE, uncertainty_on_values,
                                                   uncertainty_on_arguments, value_uncertainty_value,
                                                   argument_uncertainty_value)
        number_of_epochs -= 1
        for i in range(len(sample)):
            value = evaluation_function(sample[i], False, 0)
            if value < best_result:
                best_result = value
    return best_result


def differential_evolution_algorithm(population: np.ndarray, epochs: int, fitness_function: Callable,
                                     recombination_prob: float,
                                     uncertainty_on_values: bool, uncertainty_on_arguments: bool, value_uncertainty_value: bool,
                                     argument_uncertainty_value: bool
                                     ) -> float:
    best_result = sys.float_info.max
    fitness_values = calculate_fitness_values(fitness_function, population, uncertainty_on_values, uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value)
    while epochs > 0:
        mutant_vectors = create_mutant_vectors(population)
        trial_vectors = create_trial_vectors(population, mutant_vectors, recombination_prob)
        fitness_value_for_trial_vector = calculate_fitness_values(fitness_function, trial_vectors, uncertainty_on_values, uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value)
        selected_winning_vectors = select_winning_vectors(trial_vectors, population, fitness_values,
                                                          fitness_value_for_trial_vector)
        population = selected_winning_vectors
        epochs -= 1

        for member in population:
            val = fitness_function(member, False, 0)
            if val < best_result:
                best_result = val

    return best_result


def calculate_fitness_values(fitness_function: Callable, population: np.ndarray,
                             uncertainty_on_values: bool, uncertainty_on_arguments: bool, value_uncertainty_value: bool,
                             argument_uncertainty_value: bool
                             ) -> np.ndarray:
    values = np.zeros(shape=(population.shape[0], 1))
    for i in range(0, population.shape[0]):
        if uncertainty_on_arguments:
            values[i] = fitness_function(gaussian_uncertainty(population[i], argument_uncertainty_value), uncertainty_on_values, value_uncertainty_value)
        else:
            values[i] = fitness_function(population[i], uncertainty_on_values, value_uncertainty_value)
    return values


def select_winning_vectors(trial_vectors: np.ndarray, population: np.ndarray, population_fitness_values,
                           trial_fitness_values) -> np.ndarray:
    new_population = np.zeros(shape=population.shape)
    for i in range(0, population.shape[0]):
        new_population[i] = trial_vectors[i] if trial_fitness_values[i] < population_fitness_values[i] else population[
            i]
    return new_population


def create_trial_vectors(population: np.ndarray, mutant_vectors: [], CR: float) -> np.ndarray:
    trial_vectors = np.zeros(shape=population.shape)
    j_rand = random.randint(0, population.shape[0])
    for i in range(0, len(mutant_vectors)):
        member = population[i]
        mutant = mutant_vectors[i]
        if random.uniform(0, 1) < CR or i == j_rand:
            trial_vectors[i] = mutant
        else:
            trial_vectors[i] = member
    return trial_vectors


def create_mutant_vectors(population: np.ndarray) -> []:
    mutant_vectors = []

    for member in population:
        # Scaling factor
        f = random.uniform(0, 2)
        r1 = population[random.randint(0, population.shape[0] - 1)]
        r2 = population[random.randint(0, population.shape[0] - 1)]
        r3 = population[random.randint(0, population.shape[0] - 1)]
        mutant_vector = r1 + f * (r2 - r3)
        mutant_vectors.append(mutant_vector)
    return mutant_vectors


def create_sample(sample_size, function_dimension_size, min, max):
    points = np.zeros(shape=(sample_size, function_dimension_size))
    for i in range(sample_size):
        point = points[i]
        for j in range(function_dimension_size):
            point[j] = random.uniform(min, max)
        points[i] = point
    return points


def tournament_selection_and_mutation(sample, evaluation_function, selection_size, uncertainty_on_values,
                                      uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value):
    new_sample = np.zeros(shape=(len(sample), sample.shape[1]))
    for i in range(len(sample)):
        tmp_points = np.zeros(shape=(selection_size, sample.shape[1]))
        tmp_values = np.zeros(shape=selection_size)
        for j in range(selection_size):
            tmp_points[j] = sample[random.randint(0, len(sample) - 1)]
            if uncertainty_on_arguments:
                tmp_values[j] = evaluation_function(
                    gaussian_uncertainty(tmp_points[j], argument_uncertainty_value), uncertainty_on_values,
                    value_uncertainty_value)
            else:
                tmp_values[j] = evaluation_function(tmp_points[j], uncertainty_on_values, value_uncertainty_value)
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


def uncertainty_on_value(value, value_uncertainty_value):
    return random.gauss(value, value_uncertainty_value)


def run_functions(writer,uncertainty_on_values, uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value):
    if uncertainty_on_values:
        print("Wyniki z niepewno??ci?? na warto??ciach funkcji")
    if uncertainty_on_arguments:
        print("Wyniki z niepewno??ci?? na argumentach funkcji")
    if not uncertainty_on_values and not uncertainty_on_arguments:
        print("Wyniki bez niepewno??ci")

    samples = [create_sample(100, 10, -10, 10), create_sample(100, 2, -10, 10), create_sample(100, 10, -1, 1), create_sample(100, 2, -5.12, 5.12), create_sample(100, 2, -2.048, 2.048),
    create_sample(100, 30, -1.28, 1.28), create_sample(100, 10, -500, 500), create_sample(100, 10, -500, 500), create_sample(100, 10, -500, 500), create_sample(100, 10, -500, 500)]
    eval_functions = [functions.sum_function, functions.square_two_dim_function, functions.random_high_dimensional_function, functions.f1, functions.f2,
    functions.f4, functions.f6, functions.f7, functions.f8, functions.f9]
    eval_functions_str = ['sum', 'square_two', 'random_high_dim', 'f1', 'f2','f4', 'f6', 'f7', 'f8', 'f9']

    for i in range(len(samples)):
        res = evolutionary_algorithm(samples[i], 100, eval_functions[i], uncertainty_on_values, uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value)
        data_row = ['EA', samples[i].shape, '100', eval_functions_str[i]]
        if uncertainty_on_values:
            data_row.append(value_uncertainty_value)
        else:
            data_row.append(-1)

        if uncertainty_on_arguments:
            data_row.append(argument_uncertainty_value)
        else:
            data_row.append(-1)
        data_row.append(res)
        print(data_row)
        writer.writerow(data_row)

        data_row = []
        res = differential_evolution_algorithm(samples[i], 100, eval_functions[i], .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value)

        data_row = ['DE', samples[i].shape, '100', eval_functions_str[i]]
        if uncertainty_on_values:
            data_row.append(value_uncertainty_value)
        else:
            data_row.append(-1)

        if uncertainty_on_arguments:
            data_row.append(argument_uncertainty_value)
        else:
            data_row.append(-1)
        data_row.append(res)
        writer.writerow(data_row)
        print(data_row)


if __name__ == '__main__':
    random.seed(10)
    Values_Array = [0, 0.1, 1, 10, 200]
    output_file = open('./results','w')
    writer = csv.writer(output_file)
    headers = ['algorithm', 'sample shape', 'epochs', 'evaluation function', 'uncertainty on value', 'uncertainty on argument','result']
    writer.writerow(headers)

    for value in Values_Array:
        value_uncertainty_value = value
        for argument in Values_Array:
            argument_uncertainty_value = argument
            print("Sigma niepewno??ci na argumentach : " + str(argument) + " niepewno??ci na warto??ciach: " + str(value))
            run_functions(writer, True, True, value_uncertainty_value, argument_uncertainty_value)
    
    output_file.close()