import random
import math

class Genetic:
    def __init__(self, alpha, beta, theta):
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

        self.best_fitness = math.inf

        self.ACC = 1e4
        self.BIT_WIDTH = int(math.log2(self.ACC * 20)) + 1
        self.FMT = f"0{3 * self.BIT_WIDTH}b"

        self.N_POP       = 6000
        self.N_SELECTION = 4000
        self.N_RANDOM    = 100

        self.MUTATION_RATE = 0.05
        self.EPOCH = 200
        self.THRESHOLD = 1e-6

        self.population = [
            format(random.getrandbits(3 * self.BIT_WIDTH), self.FMT)
                for _ in range(self.N_POP)
        ]


    def equation_1(self, x, y, z):
        return self.alpha * x + y * x ** 2 + y ** 3 + z ** 3
    
    def equation_2(self, x, y, z):
        return self.beta * y + math.sin(y) + math.exp2(y) - z + math.log10(abs(x) + 1)
    
    def equation_3(self, x, y, z):
        return self.theta * z + y - (math.cos(x + y))/(math.sin(z * y - y ** 2 + z) + 2)
    
    def to_int(self, binary):
        if binary[0] == '1':
            binary = ''.join('1' if bit == '0' else '0' for bit in binary)
            return -(int(binary, 2) + 1)
        else:
            return int(binary, 2)
    
    def unpack(self, chromosome):
        x = self.to_int(chromosome[:self.BIT_WIDTH])
        y = self.to_int(chromosome[self.BIT_WIDTH:2 * self.BIT_WIDTH])
        z = self.to_int(chromosome[2 * self.BIT_WIDTH:])
        return x, y, z

    def fitness(self, chromosome):
        x, y, z = self.unpack(chromosome)
        x_norm = x / self.ACC
        y_norm = y / self.ACC
        z_norm = z / self.ACC
        return self.equation_1(x_norm, y_norm, z_norm) ** 2 +\
            self.equation_2(x_norm, y_norm, z_norm) ** 2 +\
            self.equation_3(x_norm, y_norm, z_norm) ** 2
    
    def select_best(self):
        return self.population[0]
    
    def selection(self):
        selection = self.population[:self.N_SELECTION]
        selection.extend([
            format(random.getrandbits(3 * self.BIT_WIDTH - 1), self.FMT)
                for _ in range(self.N_RANDOM)
        ])
        return selection

    def co_mutate(self):
        selection = self.selection()
        shuffle = random.sample(range(len(selection)), len(selection))
        pairs = [(shuffle[2 * i], shuffle[2 * i + 1]) for i in range(self.N_SELECTION // 2)]
        for c1, c2 in pairs:
            chromosome1 = selection[c1]
            chromosome2 = selection[c2]
            cross_point = random.randint(0, 3 * self.BIT_WIDTH)

            # Cross over
            new_chromosome1 = chromosome1[:cross_point] + chromosome2[cross_point:]
            new_chromosome2 = chromosome2[:cross_point] + chromosome1[cross_point:]

            self.population.append(new_chromosome1)
            self.population.append(new_chromosome2)

            # Mutation
            for j in range(3 * self.BIT_WIDTH):
                if random.random() < self.MUTATION_RATE:
                    mute_bit = '0' if new_chromosome1[j] == '1' else '0'
                    self.population.append(
                        new_chromosome1[:j] + mute_bit + new_chromosome1[j + 1:]
                    )
                
                if random.random() < self.MUTATION_RATE:
                    mute_bit = '0' if new_chromosome2[j] == '1' else '0'
                    self.population.append(
                        new_chromosome2[:j] + mute_bit + new_chromosome2[j + 1:]
                    )

        self.population.sort(key=lambda chromosome: self.fitness(chromosome))
        self.population = self.population[:self.N_POP]

    def solve(self):
        for _ in range(self.EPOCH):
            self.best_fitness = self.fitness(self.select_best())

            if _ % 5 == 0:
                print(self.best_fitness)

            if self.best_fitness < self.THRESHOLD:
                x, y, z = self.unpack(self.select_best())
                return x / self.ACC, y / self.ACC, z / self.ACC
            self.co_mutate()

        x, y, z = self.unpack(self.select_best())
        return x / self.ACC, y / self.ACC, z / self.ACC


# Main function
def solver(alpha, beta, teta):
    genetic = Genetic(alpha, beta, teta)
    x, y, z = genetic.solve()
    return x, y, z