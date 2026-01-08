# create an array and calculate mean and median and maximum and minimumand sum of the array elements
import numpy as np
array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mean = np.mean(array)
median = np.median(array)
maximum = np.max(array)
minimum = np.min(array)
sum_array = np.sum(array)  
print("Mean:", mean)
print("Median:", median)
print("Maximum:", maximum)
print("Minimum:", minimum)
print("Sum:", sum_array)