import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import random
import time


def find_distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class GeneticAlgorithm:

    def __init__(self, crashes=None, max_num_iterations=None):

        self.car_crashes = crashes

        car_crashes = [(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)) for i in range(10)]
        self.car_crashes = np.array(car_crashes)

        self.max_num_iterations = max_num_iterations

    def calculate_fitness(self, solution):
        """
        Calculates the value of the fitness function for a given solution (positions of 6 ambulances).
        The geneic algorithm tries to minimize this value.
        :param solution: list of lists of two floats each with x and y coordinates of the ambulances
        :return:
        """
        solution = np.array(solution)
        solution = solution.reshape(6, 2)

        total_fitness = 0

        for car_crash in self.car_crashes:
            distance_to_closest_ambulance = self.distance_to_closest_ambulance(car_crash, solution)
            total_fitness += distance_to_closest_ambulance

        return total_fitness

    @staticmethod
    def distance_to_closest_ambulance(car_crash, solution):
        shortest_distance = find_distance(car_crash, solution[0])

        for i, ambulance in enumerate(solution):
            curr_distance = find_distance(car_crash, ambulance)

            if curr_distance < shortest_distance:
                shortest_distance = curr_distance

        return shortest_distance

    def run(self):
        start_time = time.time()
        varbound = np.array([[36.2, 38], [-3.2, -0.5]] * 6)

        algorithm_param = {'max_num_iteration': self.max_num_iterations,
                           'population_size': 100,
                           'mutation_probability': 0.1,
                           'elit_ratio': 0.01,
                           'crossover_probability': 0.5,
                           'parents_portion': 0.3,
                           'crossover_type': 'uniform',
                           'max_iteration_without_improv': 200}

        model = ga(function=self.calculate_fitness, dimension=12, variable_type='real',
                   variable_boundaries=varbound, algorithm_parameters=algorithm_param)

        model.run()
        print("--- %s seconds for genetic algorithm ---" % (time.time() - start_time))

        convergence = model.report

        #print(convergence)
        return model.output_dict['variable']
