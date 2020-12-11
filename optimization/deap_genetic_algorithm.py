from deap import base
from deap import tools
from deap import algorithms
from deap import creator
import time
import random
import multiprocessing as mp
from tqdm import tqdm
from optimization.genetic_algorithm import GeneticAlgorithm



def generate_random_float():
    return random.uniform(-10.0, 10.0)


def generate_random_points(n):
    representatives = []
    for i in range(n):
        representatives.append([generate_random_float(), generate_random_float()])
    return representatives


def find_distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


car_crashes = generate_random_points(10)


def evaluate(individual, representatives):
    # Compute the collaboration fitness
    # print('eval::::::::::')
    # print(representatives)
    total_distance = 0
    distance_to_representatives = 0

    for r in representatives:
        distance_to_representatives += find_distance(individual, r)

    for car_crash in car_crashes:
        total_distance += find_distance(car_crash, individual)
    return [total_distance - distance_to_representatives]


toolbox = base.Toolbox()

SPECIES_SIZE = 100
IND_SIZE = 1

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("individual", tools.initRepeat, creator.Individual, generate_random_float,
                 2)  # every individual is a location consisting of x and y coordinates
toolbox.register("species", tools.initRepeat, list, toolbox.individual,
                 SPECIES_SIZE)  # species is a list of 50 locations -> (float, float)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
toolbox.register("get_best", tools.selBest, k=1)

NUM_SPECIES = 6


def evolve_species(species, arr_representatives, index):
    # print(species)
    # print(arr_representatives)
    # print(i)

    representative = toolbox.get_best(species)
    arr_representatives[index] = representative

    ngen = 3000

    for i in tqdm(range(ngen)):
        # Vary the species individuals
        species = algorithms.varAnd(species, toolbox, 0.6, 1.0)

        other_representatives = arr_representatives[:index] + arr_representatives[index + 1:]

        for ind in species:
            # Evaluate and set the individual fitness
            ind.fitness.values = toolbox.evaluate(ind, other_representatives)

        # Select the individuals
        species = toolbox.select(species, len(species))  # Tournament selection
        next_repr = toolbox.get_best(species)[0]  # Best selection

        representative = next_repr
        arr_representatives[index] = representative

    return representative


if __name__ == '__main__':
    start_time = time.time()

    pool = mp.Pool(6)

    with mp.Manager() as manager:
        arr_representatives = manager.list(generate_random_points(NUM_SPECIES))
        results = [pool.apply(evolve_species, args=(toolbox.species(), arr_representatives, i)) for i in
                   range(NUM_SPECIES)]

        print(results)

    pool .close()

    ga = GeneticAlgorithm()
    print("Fitness of the result:")
    print(ga.calculate_fitness(results))
    print("--- %s seconds ---" % (time.time() - start_time))
