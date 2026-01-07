# # swap two variable without using third variable in python 
# import numpy as np
# a = 5
# b = 10      
# a = a + b
# b = a - b   
# a = a - b
# print("After swapping: a =", a, ", b =", b)
  

# find power of a number using math library
# import math 
# base = 2
# exponent = 3
# result = math.pow(base, exponent)
# print(f"{base} raised to the power of {exponent} is {result}")


# perform addition of two numbers using matrix
# import numpy as np
# matrix1 = np.array([[1, 2], [3, 4]])
# matrix2 = np.array([[5, 6], [7, 8]])
# result = np.add(matrix1, matrix2)
# print("Resultant Matrix:\n", result)

# create an array and calculate mean and median and maximum and minimumand sum of the array elements
# import numpy as np
# array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# mean = np.mean(array)
# median = np.median(array)
# maximum = np.max(array)
# minimum = np.min(array)
# sum_array = np.sum(array)  
# print("Mean:", mean)
# print("Median:", median)
# print("Maximum:", maximum)
# print("Minimum:", minimum)
# print("Sum:", sum_array)

# create a ist and perform sorting and reversing the list

# how many type of plotting library are there in python?
# Matplotlib
# import matplotlib.pyplot as plt
# y=[2, 4, 6, 8, 10]
# x=[1, 2, 34, 4, 20]
# plt.plot(x, y)
# plt.xlabel('X-axis')    
# plt.ylabel('Y-axis')
# plt.title('Simple Line Plot')
# plt.show()  
 
#  print prime no upto 25 
# import math

# for num in range(2, 26):
#     is_prime = True
#     for i in range(2, int(num**0.5) + 1):
#         if num % i == 0:
#             is_prime = False
#             break
#     if is_prime:
#         print(num, end=' ')

# import math.floor
# import math.sqrt
# import math.ceil
# import math.e



#use math.ceil and math.floor
# import math

# number = 5.3
# ceiling_value = math.ceil(number)
# floor_value = math.floor(number)

# print("Ceiling value of", number, "is", ceiling_value)
# print("Floor value of", number, "is", floor_value)



# surt
# import math 
# number = 16
# sqrt_value = math.sqrt(number)
# print("Square root of", number, "is", sqrt_value)

#scipy mean percentile 
import numpy as np
from scipy import stats

data = np.random.normal(loc=0, scale=1, size=1000)
mean = stats.tmean(data)
percentile_50 = np.percentile(data, 50)
print("Mean:", mean)
print("50th Percentile:", percentile_50)    