from deap import base
from deap import tools
from deap import algorithms
from deap import creator
import random
import multiprocessing

from optimization.genetic_algorithm import GeneticAlgorithm


def find_distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


if __name__ == "__main__":
    toolbox = base.Toolbox()

    car_crashes = [(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)) for i in range(6)]
    print('Car crashes:')
    print(car_crashes)

    def evaluate(individual, representatives):
        # Compute the collaboration fitness
        total_distance = 0
        distance_to_representatives = 0

        for r in representatives:
            distance_to_representatives += find_distance(individual, r)

        for car_crash in car_crashes:
            total_distance += find_distance(car_crash, individual)
        return [total_distance - distance_to_representatives]

    def generate_random_float():
        return random.uniform(-10.0, 10.0)

    SPECIES_SIZE = 50
    IND_SIZE = 1

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox.register("individual", tools.initRepeat, creator.Individual, generate_random_float, 2)  #every individual is a location consisting of x and y coordinates
    toolbox.register("species", tools.initRepeat, list, toolbox.individual, SPECIES_SIZE)   #species is a list of 50 locations -> (float, float)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("get_best", tools.selBest, k=1)

    NUM_SPECIES = 6

    species = [toolbox.species() for _ in range(NUM_SPECIES)]
    representatives = [random.choice(species[i]) for i in range(NUM_SPECIES)]

    # print(species)
    # print('///////////////////////')
    # print(representatives)
    g = 0
    ngen = 1000

    while g < ngen:
        #print(representatives)
        # Initialize a container for the next generation representatives
        next_repr = [None] * len(species)
        for i, s in enumerate(species):       #original code contained species_index
            # Vary the species individuals
            s = algorithms.varAnd(s, toolbox, 0.6, 1.0)

            other_representatives = representatives[:i] + representatives[i + 1:]

            for ind in s:
                # Evaluate and set the individual fitness
                ind.fitness.values = toolbox.evaluate(ind, other_representatives)

            # Select the individuals
            species[i] = toolbox.select(s, len(s))  # Tournament selection
            next_repr[i] = toolbox.get_best(s)[0]  # Best selection

        representatives = next_repr
        g = g+1

    print(representatives)
    ga = GeneticAlgorithm(car_crashes, 10)
    print(ga.calculate_fitness(representatives))

