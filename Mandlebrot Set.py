import math
import sys

import pygame
from numba import cuda
import numpy as np


@cuda.jit
def mandlebrotIteration(array, iterations):
    """

    Iterate z_(n+1) = z_n^2 + c for each element c in the array.
    Returns 1 if |z_n| < 2 after iterating and 0 otherwise.

    """
    pos = cuda.grid(1)

    initialVal = array[pos]

    if pos < array.size:  # Check array boundaries
        for __ in range(iterations):
            array[pos] = array[pos] ** 2 + initialVal

        array[pos] = 1 if abs(array[pos]) < 2 else 0


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
        (255, 255, 255),
        (0, 0, 0)
    ]

    ITERATIONS = 20
    RANGE = ([-2.5, 1], [-1.2, 1.2])
    STEP = 0.0025

    array = genInputs(RANGE[0], RANGE[1], STEP).flatten()

    # Processing the array inputs

    threadsperblock = 32
    blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock

    mandlebrotIteration[blockspergrid, threadsperblock](array, ITERATIONS)

    array = np.reshape(array, (math.floor((RANGE[0][1] - RANGE[0][0]) / STEP), math.floor((RANGE[1][1] - RANGE[1][0]) / STEP)))

    # Generating the image

    s = pygame.Surface(array.shape)
    sP = pygame.PixelArray(s)

    for x in range(len(array)):
        row = array[x]
        for y in range(len(row)):
            element = row[y]
            sP[x, y] = colours[int(element)]

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
