"""

Danny Bold, 11/11/2021

A project that generates fractals by applying Newton's Method to the complex plane.

If the roots of a polynomial are specified, the program will automatically differentiate the polynomial.
Other functions will work, but the derivative must be manually entered.

"""

import math
import cmath
import sys
from functools import reduce

import pygame


def newtonsMethod(array, f, fPrime, iterations):
    """

    Carry out Newton-Raphson to each of the points in the 2d array parameter, with the number of iterations given by the iterations parameter.

    We sometimes run into ZeroDivisionErrors, hence the try-except clause is needed.

    """

    out = []

    for x in range(len(array)):
        row = []

        for y in range(len(array[x])):
            element = array[x][y]

            for i in range(iterations):
                try:
                    element = element - f(element) / fPrime(element)

                except ZeroDivisionError:  # element is a stationary point of f
                    pass

            row.append(element)

        out.append(row)

    return out


def genInputs(xRange, yRange, step):
    """

    xRange and yRange need to take the form [start, end] and step must be a float, preferably 0.01 or similarly small.

    """
    out = []

    for x in range(math.floor((xRange[1] - xRange[0]) / step)):
        row = []

        for y in range(math.floor((yRange[1] - yRange[0]) / step)):
            row.append(xRange[0] + x * step + (yRange[0] + y * step) * 1j)

        out.append(row)

    return out


def genFunction(*roots):
    """

    Given roots = [r1, r2, ...], this function returns a lambda expression equivalent to (x-r1)(x-r2)...,
    and a lambda expression equivalent to it's derivative.

    """

    # The next two for-loops are prep for the derivative expression - writing it in one line would be very complicated.

    tempIndices = []

    for k in range(len(roots)):  # k = index to miss
        t = []

        for n in range(len(roots)):
            if n != k:
                t.append(n)

        tempIndices.append(t)

    temp = []

    for i in tempIndices:
        t = []

        for h in i:
            t.append(roots[h])

        temp.append(t)

    return (lambda x: reduce(lambda z, y: z * y, [(x - y) for y in roots]),  # Function
            lambda x: sum([reduce(lambda z, y: z * y, [(x - y) for y in g]) for g in temp]))  # Derivative


def main():
    """

    Adjust ITERATIONS to develop a more complex fractal (useful if you're displaying at higher resolutions).
    Adjust RANGE to see the fractal at different values.
    Adjust STEP with RANGE to get the required size of the array.
    Adjust ROOTS to alter the fractal generated, or comment that and the 'genFunction' line and add your own function.

    """
    colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255),
        (0, 0, 0)
    ]

    ITERATIONS = 20
    RANGE = ([-5, 5], [-5, 5])
    STEP = 0.01

    ROOTS = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j, 1, -1]

    f, fPrime = genFunction(*ROOTS)
    array = genInputs(RANGE[0], RANGE[1], STEP)

    # array = newtonsMethod(array, cmath.sin, cmath.cos, ITERATIONS)
    array = newtonsMethod(array, f, fPrime, ITERATIONS)

    s = pygame.Surface((len(array[0]), len(array)))
    sP = pygame.PixelArray(s)

    # Set pixels to the colour they're nearest to
    for x in range(len(array)):
        row = array[x]

        for y in range(len(row)):
            element = row[y]
            distances = [abs(element - x) for x in ROOTS]
            sP[x, y] = colours[distances.index(min(distances))]  # Setting the colour of the pixel to the nearest root.

    del sP

    screen = pygame.display.set_mode((len(array[0]), len(array)))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()

        screen.blit(s, (0, 0))

        pygame.display.flip()


if __name__ == '__main__':
    main()

