#!/usr/bin/env python

#    This program tries to find a optimal location for a clinic between 3 cities.
#    Cities have a location and a population related to them.
#    Although one might simply calculate a weighted average and find the solution,
#        this is more of a 'demonstration' to optimization algorithms.


__author__ = "Mahyar Sahraei"
__email__  = "sahraei19@gmail.com"


import numpy as np
import matplotlib.pyplot as plt

class place:
    def __init__(self, x, y, population):
        self.x = x
        self.y = y
        self.pop = population

    def distance(self, place):
        return np.sqrt((self.x - place.x)**2 + (self.y - place.y)**2)


# Target function that has to be minimized
def target_function(Clinic, A, B, C, A_visitors, B_visitors, C_visitors):
    return (Clinic.distance(A) * A_visitors,
            Clinic.distance(B) * B_visitors,
            Clinic.distance(C) * C_visitors)


# City's location
A = place(2.0, 4.0, 45000)
B = place(4.0, 1.0, 12000)
C = place(1.0, 0.5, 27000)

# Clinic's location, starting at the center of the ABC triangle
Clinic = place(2.3, 1.5, 0)

VISIT_RATE = 0.011 # Four visits per yearS
EPOCH = 40         # Number of iterations
DELTA = 1e-4       # Location change rate for each iteration

# List of cost values for visualization
cost_values = []

# Estimation of average visitors per day
A_visitors = int(A.pop * VISIT_RATE)
B_visitors = int(B.pop * VISIT_RATE)
C_visitors = int(C.pop * VISIT_RATE)

# Simulation
for i in range(EPOCH):

    cost = target_function(Clinic, A, B, C, A_visitors, B_visitors, C_visitors)
    dx = DELTA * (
        cost[0] * (A.x - Clinic.x) +
        cost[1] * (B.x - Clinic.x) +
        cost[2] * (C.x - Clinic.x))
    Clinic.x += dx

    dy = DELTA * (
        cost[0] * (A.y - Clinic.y) +
        cost[1] * (B.y - Clinic.y) +
        cost[2] * (C.y - Clinic.y))
    Clinic.y += dy

    cost_values.append(sum(cost))

# Result visualization
plt.plot(cost_values)
plt.xlabel("Iteration #")
plt.ylabel("Cost (Total distance traveled)")
plt.show()
