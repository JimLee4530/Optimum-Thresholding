import numpy as np
from otsu import otsu, fast_ostu

np.random.seed(8)

class genetic:
    def __init__(self, image, N = 4, index = 4, max_iteration=10):
        """
        genetic algorithm
        :param image: image feature
        :param N: num of population
        :param population: N population
        :param index: index of crossover
        """
        self.image = image
        self.N = N
        self.population = np.random.randint(0, 256, self.N)
        self.index = index
        self.max_iteration = max_iteration

    def get_fitness(self):
        # fitness = [otsu(self.image, i) for i in self.population]
        fitness = [fast_ostu(self.image, i) for i in self.population]
        return fitness

    def select(self):
        fitness = self.get_fitness()
        sum_fitness = np.sum(fitness)
        probability = fitness / sum_fitness
        new_population = np.random.choice(self.population, self.N, True, probability)
        return new_population

    def crossover_base(self, a, b, index):
        a_1 = a % pow(2, index)
        a_0 = a - a_1
        b_1 = b % pow(2, index)
        b_0 = b - b_1
        new_a = a_0+b_1
        new_b = b_0+a_1
        return new_a, new_b

    def crossover(self):
        np.random.shuffle(self.population) # random shuffle
        new_population = np.zeros_like(self.population)
        for i in range(0, self.N, 2):
            new_population[i], new_population[i+1] = self.crossover_base(self.population[i], self.population[i+1], self.index)
        return new_population

    def mutate(self):
        m = np.random.randint(low=0, high=8, size=self.N) # random mutate for every one in population
        new_population = self.population ^ pow(2, m)
        return new_population

    def get_threshold(self):
        flag = 0
        fitness = self.get_fitness()
        best_fitness = np.max(fitness)
        best_threshold = self.population[np.argmax(fitness)]
        while True:
            self.population = self.crossover()
            self.population = self.mutate()
            self.population = self.select()

            fitness = self.get_fitness()
            if best_fitness < np.max(fitness):
                best_fitness = np.max(fitness)
                best_threshold = self.population[np.argmax(fitness)]
                flag = 0
            else:
                flag += 1

            if flag > self.max_iteration:
                break
            print("best threshold is: ")
            print(best_threshold)
            print("best fitness is: ")
            print(best_fitness)
        return best_threshold