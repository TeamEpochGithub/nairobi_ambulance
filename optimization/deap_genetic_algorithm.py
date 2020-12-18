import os

from deap import base
from deap import tools
from deap import algorithms
from deap import creator
import time
import random
import multiprocessing as mp
from tqdm import tqdm
from optimization.genetic_algorithm import GeneticAlgorithm
import psutil


def generate_random_float():
    return random.uniform(-10.0, 10.0)


def generate_random_points(n):
    print("Generate random points!!!!!!!!!")
    representatives = []
    for i in range(n):
        representatives.append([generate_random_float(), generate_random_float()])
    return representatives


def find_distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# car_crashes = generate_random_points(10)
# car_crashes = [[1.4, 4.3], [-7.3, -9.5], [-2.7, 3.4], [1.1, 7.99], [0.2, -7.2], [-3.3, -6.9], [3.9, 2.2], [3.68, -9.6],
#                [2.45, 6.8], [2.55, -4.7]]


def evaluate(individual, representatives, car_crashes):
    """
    Calculates fitness which is then taken with a negative sign by the library's toolbox. So the higher the return value
     of this function - the worse the individual is.
    :param individual: the current individual we evaluate
    :param representatives: the best individuals from the other populations
    :return: fitness
    """
    total_distance = 0
    distance_to_representatives = 0

    for r in representatives:
        distance_to_representatives += find_distance(individual, r)

    for car_crash in car_crashes:
        total_distance += find_distance(car_crash, individual)
    return total_distance, distance_to_representatives


toolbox = base.Toolbox()

SPECIES_SIZE = 100
IND_SIZE = 1

creator.create("Fitness", base.Fitness, weights=(-0.5, 1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox.register("individual", tools.initRepeat, creator.Individual, generate_random_float,
                 2)  # every individual is a location consisting of x and y coordinates
toolbox.register("species", tools.initRepeat, list, toolbox.individual,
                 SPECIES_SIZE)  # species is a list of 50 locations -> (float, float)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register('select', tools.selNSGA2)
toolbox.register("evaluate", evaluate)
toolbox.register("get_best", tools.selBest, k=1)

NUM_SPECIES = 6


def evolve_species(species, list_of_representatives, index, car_crashes):
    """

    :param species: the current population(species)
    :param list_of_representatives: the best individuals from each population
    :param index: the index of the best individual of the current species
    We need this so we can ignore the best individual from the current species and
    use only the ones from the other populations.
    :return: the best individual from this population after all of the evolution has been done
    """
    representative = toolbox.get_best(species)
    list_of_representatives[index] = representative[0]

    ngen = 1500

    for _ in tqdm(range(ngen)):
        # Vary the species individuals
        species = algorithms.varAnd(species, toolbox, 0.6, 1.0)

        other_representatives = list_of_representatives[:index] + list_of_representatives[index + 1:]

        for ind in species:
            # Evaluate and set the individual fitness
            ind.fitness.values = toolbox.evaluate(ind, other_representatives, car_crashes)

        # Select the individuals
        species = toolbox.select(species, len(species))  # Tournament selection
        next_repr = toolbox.get_best(species)[0]  # Best selection

        representative = next_repr
        list_of_representatives[index] = representative

    return representative


def run(car_crashes):
    p = psutil.Process()
    all_cpus = list(range(psutil.cpu_count()))
    p.cpu_affinity(all_cpus)

    start_time = time.time()

    pool = mp.Pool(6)

    with mp.Manager() as manager:
        representatives = []
        list_of_species = []
        for _ in range(NUM_SPECIES):
            list_of_species.append(toolbox.species())
        for i in range(len(list_of_species)):
            representatives.append(toolbox.get_best(list_of_species[i])[0])
            # taking the best individuals from every species

        list_of_representatives = manager.list(representatives)

        results_from_processes = [pool.apply_async(evolve_species, args=(list_of_species[i], list_of_representatives,
                                                                         i, car_crashes)) for i in range(NUM_SPECIES)]

        results = [p.get() for p in results_from_processes]

        print(results)

    pool.close()

    ga = GeneticAlgorithm(crashes=car_crashes)
    print("Fitness of the result:")
    print(ga.calculate_fitness(results))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    car_crashes = generate_random_points(10)
    run(car_crashes)
