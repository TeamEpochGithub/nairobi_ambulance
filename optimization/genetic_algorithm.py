import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import random


def find_distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class GeneticAlgorithm:

    def __init__(self, crashes, max_num_iterations):

        self.car_crashes = crashes

        #car_crashes = [(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)) for i in range(10)]
        #self.car_crashes = np.array(car_crashes)

        self.max_num_iterations = max_num_iterations

    def calculate_fitness(self, solution):
        """
        Calculates the value of the fitness function for a given solution (positions of 6 ambulances).
        :param solution: list of lists of two floats each with x and y coordinates of the ambulances
        :return:
        """
        solution = np.array(solution)
        solution = solution.reshape(6, 2)

        total_fitness = 0

        for car_crash in self.car_crashes:
            closest_ambulance = self.find_closest_ambulance(car_crash, solution)
            total_fitness += find_distance(car_crash, closest_ambulance)

        return total_fitness

    @staticmethod
    def find_closest_ambulance(car_crash, solution):
        shortest_distance = find_distance(car_crash, solution[0])
        index_closest_ambulance = 0

        for i, ambulance in enumerate(solution):
            curr_distance = find_distance(car_crash, ambulance)

            if curr_distance < shortest_distance:
                shortest_distance = curr_distance
                index_closest_ambulance = i

        return solution[index_closest_ambulance]

    def run(self):
        varbound = np.array([[36.2, 38], [-3.2, -0.5]] * 6)

        algorithm_param = {'max_num_iteration': self.max_num_iterations,
                           'population_size': 100,
                           'mutation_probability': 0.1,
                           'elit_ratio': 0.01,
                           'crossover_probability': 0.5,
                           'parents_portion': 0.3,
                           'crossover_type': 'uniform',
                           'max_iteration_without_improv': None}

        model = ga(function=self.calculate_fitness, dimension=12, variable_type='real',
                   variable_boundaries=varbound, algorithm_parameters=algorithm_param)

        model.run()

        convergence = model.report

        #print(convergence)
        return model.output_dict['variable']
