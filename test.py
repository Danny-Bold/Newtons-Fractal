# from numba import cuda
# import numpy as np
#
#
# @cuda.jit
# def my_kernel(io_array):
#     x = cuda.grid(1)
#     if x < io_array.size:  # Check array boundaries
#         io_array[x] *= 2  # do the computation
#
#
# def myFunc(n):
#     return n * 2
#
# data = np.ones(255)
#
# threadsperblock = 32
#
# # Calculate the number of thread blocks in the grid
# blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
#
# my_kernel[blockspergrid, threadsperblock](data)
#
# data.reshape((51, 5))
#
# print(data)


from numba import cuda