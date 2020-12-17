from optimization.genetic_algorithm import GeneticAlgorithm
import pandas as pd
import random


def main():
    # df_crashes = pd.read_csv("../resources/Train.csv", parse_dates=['datetime'])
    # del df_crashes['uid']
    # del df_crashes['datetime']
    # car_crashes = df_crashes.to_numpy()
    ga = GeneticAlgorithm(max_num_iterations=3000)
    ga.run()


if __name__ == "__main__":
    main()
    # print(random.uniform(0, 10))