#1.  write a program to swap two numbers without using third variable 
x=45 
y=25 
x,y=y,x 
print(x) 
print(y) 
 
#2. add two numbers by numpy lib 
import numpy as np 
a= np.array([1,2,3]) 
b= np.array([4,5,6]) 
x=a+b 
print(x) 
 
# 3. write a program to draw a line using matplotlib 
import matplotlib.pyplot as plt 
import numpy as np 
x=np.array([0,20]) 
y=np.array([10,30]) 
plt.plot(x,y) 
plt.show() 
 
 
# 4. write a program for scatterplot 
import matplotlib.pyplot as plt 
x=[2,4,6,8,10] 
y=[3,5,7,9,11] 
plt.scatter(x,y) 
plt.show() 
   
 
# 5. Write a program to sort the array using numpy library 
import numpy as np 
x= np.array([4,7,9,6]) 
mean=np.sort(x) 
print(mean) 
 
# 6. Write a program to print the head and tail of the iris dataset. 
import seaborn as sns 
import matplotlib.pyplot as plt 
iris=sns.load_dataset("iris") 
print(iris.head()) 
print(iris.tail()) 
 
# 7. Program to print square root ,power   
import matplotlib.pyplot as plt 
import numpy as np 
import math 
x=np.array([2,4,9]) 
y=np.sqrt(x) 
z=np.power(x,3) 
print(y,z)
 
#8. Using random values and condition check the number is less than 30 or not 
from numpy import random 
import numpy as np 
x=random.randint(100) 
print(x) 
if x <30: 
    print("x is less than 30") 
else: 
    print("x is greater than 30") 
 
#9. Program to sort first five integers from random 10 integers 
import numpy as np 
arr = np.random.randint(1, 101, size=10) 
print("Original 10 integers:", arr) 
first_five = arr[:5] 
sorted_five = np.sort(first_five) 
print("First 5 integers:", first_five) 
print("Sorted first 5:", sorted_five)