"""

Danny Bold, 12/11/2021

A project that generates fractals by applying Newton's Method to the complex plane.

This version is CUDA-accelerated, and so ideally requires a CUDA-enabled GPU.

If the roots of a polynomial are specified, the program will automatically differentiate the polynomial.
Other functions are not supported due to the limited functions usable by CUDA kernels, but can be programmed in.

"""

import math
import sys

import pygame
from numba import cuda
import numpy as np


@cuda.jit
def newtonsMethod(array, roots, iterations, temp):
    """

    Carry out Newton-Raphson to each of the points in the 2d array parameter,
    with the number of iterations given by the iterations parameter.

    """
    pos = cuda.grid(1)

    if pos < array.size:  # Check array boundaries
        for __ in range(iterations):
            fPrime = 0

            for j in temp:
                h = 1
                for y in j:
                    h *= (array[pos] - y)
                fPrime += h

            f = 1
            for y in roots:
                f *= (array[pos] - y)

            if fPrime != 0:
                array[pos] = array[pos] - f / fPrime


def genInputs(xRange, yRange, step):
    """

    xRange and yRange need to take the form [start, end] and step must be a float, preferably 0.01 or similarly small.

    """
    out = np.zeros((math.floor((xRange[1] - xRange[0]) / step), math.floor((yRange[1] - yRange[0]) / step)), dtype=np.complex_)
    for x in range(math.floor((xRange[1] - xRange[0]) / step)):
        for y in range(math.floor((yRange[1] - yRange[0]) / step)):
            out[x][y] = xRange[0] + x * step + (yRange[0] + y * step) * 1j

    return out


def main():
    """

    Adjust ITERATIONS to develop a more complex fractal (useful if you're displaying at higher resolutions).
    Adjust RANGE to see the fractal at different values.
    Adjust STEP with RANGE to get the required size of the array.
    Adjust ROOTS to alter the fractal generated, or comment that and the 'genFunction' line and add your own function.

    """

    colours = [
        (45, 225, 252),
        (9, 232, 94),
        (33, 79, 75),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255),
        (0, 0, 0)
    ]

    ITERATIONS = 40
    RANGE = ([-28.4, 10], [-6, 6])
    STEP = 0.01

    ROOTS = np.array((1, (-1 + math.sqrt(3) * 1j) / 2, (-1 - math.sqrt(3) * 1j) / 2))

    array = genInputs(RANGE[0], RANGE[1], STEP).flatten()

    # Precalculating information for the CUDA kernel to use

    tempIndices = []

    for k in range(len(ROOTS)):  # k = index to miss
        t = []

        for n in range(len(ROOTS)):
            if n != k:
                t.append(n)

        tempIndices.append(t)

    temp = []

    for i in tempIndices:
        t = []

        for h in i:
            t.append(ROOTS[h])

        temp.append(t)

    temp = np.array(temp)

    # Processing the array inputs

    threadsperblock = 32
    blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock

    newtonsMethod[blockspergrid, threadsperblock](array, ROOTS, ITERATIONS, temp)

    array = np.reshape(array, (math.floor((RANGE[0][1] - RANGE[0][0]) / STEP), math.floor((RANGE[1][1] - RANGE[1][0]) / STEP)))

    # Generating the image

    s = pygame.Surface(array.shape)
    sP = pygame.PixelArray(s)

    # Set pixels to the colour they're nearest to
    for x in range(len(array)):
        row = array[x]

        for y in range(len(row)):
            element = row[y]
            distances = [abs(element - x) for x in ROOTS]
            sP[x, y] = colours[distances.index(min(distances))]  # Setting the colour of the pixel to the nearest root.

    del sP

    # Displaying the image

    screen = pygame.display.set_mode((len(array), len(array[0])))

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
