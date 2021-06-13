import random
import numpy as np
import sys
import functions
from typing import Callable

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
    # TODO Calculate fitness function
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


def run_functions(uncertainty_on_values, uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value):
    if uncertainty_on_values:
        print("Wyniki z niepewnością na wartościach funkcji")
    if uncertainty_on_arguments:
        print("Wyniki z niepewnością na argumentach funkcji")
    if not uncertainty_on_values and not uncertainty_on_arguments:
        print("Wyniki bez niepewności")
    sample = create_sample(100, 10, -10, 10)
    print(evolutionary_algorithm(sample, 100, functions.sum_function, uncertainty_on_values, uncertainty_on_arguments,
                                 value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample,100,functions.sum_function, .85,uncertainty_on_values, uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))

    sample = create_sample(100, 2, -10, 10)
    print(evolutionary_algorithm(sample, 100, functions.square_two_dim_function, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample,100,functions.square_two_dim_function, .85,uncertainty_on_values, uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))

    sample = create_sample(100, 10, -1, 1)
    print(evolutionary_algorithm(sample, 100, functions.random_high_dimensional_function, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.random_high_dimensional_function, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))

    sample = create_sample(100, 2, -5.12, 5.12)
    print(evolutionary_algorithm(sample, 100, functions.f1, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.f1, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))

    sample = create_sample(100, 2, -2.048, 2.048)
    print(evolutionary_algorithm(sample, 100, functions.f2, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.f2, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))

    sample = create_sample(100, 30, -1.28, 1.28)
    print(evolutionary_algorithm(sample, 100, functions.f4, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.f4, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))

    sample = create_sample(100, 10, -500, 500)
    print(evolutionary_algorithm(sample, 100, functions.f6, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.f6, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))
    sample = create_sample(100, 10, -500, 500)
    print(evolutionary_algorithm(sample, 100, functions.f7, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.f7, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))
    sample = create_sample(100, 10, -500, 500)
    print(evolutionary_algorithm(sample, 100, functions.f8, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.f8, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))
    sample = create_sample(100, 10, -500, 500)
    print(evolutionary_algorithm(sample, 100, functions.f9, uncertainty_on_values,
                                 uncertainty_on_arguments, value_uncertainty_value, argument_uncertainty_value))
    print(differential_evolution_algorithm(sample, 100, functions.f9, .85, uncertainty_on_values,
                                           uncertainty_on_arguments,
                                           value_uncertainty_value, argument_uncertainty_value))


if __name__ == '__main__':
    random.seed(10)
    Values_Array = [0.1, 1, 10, 200]
    Arguments_Array = [0.1, 1, 10, 200]

    for value in Values_Array:
        value_uncertainty_value = value
        for argument in Arguments_Array:
            argument_uncertainty_value = argument
            print("Sigma niepewności na argumentach : " + str(argument) + " niepewności na wartościach: " + str(value))
            run_functions(False, False, value_uncertainty_value, argument_uncertainty_value)
            run_functions(True, False, value_uncertainty_value, argument_uncertainty_value)
            run_functions(False, True, value_uncertainty_value, argument_uncertainty_value)
            run_functions(True, True, value_uncertainty_value, argument_uncertainty_value)
