from optimization.genetic_algorithm import GeneticAlgorithm
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import random
import numpy as np


def main():
    # df_crashes = pd.read_csv("../resources/Train.csv", parse_dates=['datetime'])
    # del df_crashes['uid']
    # del df_crashes['datetime']
    # car_crashes = df_crashes.to_numpy()
    ga = GeneticAlgorithm(max_num_iterations=3000)
    ga.run()
    # car_crash = [2.0, 2.0]
    # car_crash = np.reshape(car_crash, (1, 2))
    # print("car crash: " + str(car_crash))
    # solution = [[2.0, 2.0], [1.7, 3.8], [4.8, 2.9], [9.1, 7.2], [9.8, 2.5], [0.1, 6.8]]
    # distances = euclidean_distances(solution, car_crash)
    # print(distances)
    # print(min(distances))

if __name__ == "__main__":
    main()
    # print(random.uniform(0, 10))