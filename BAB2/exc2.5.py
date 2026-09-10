import math

print('Input float numbers separated by spaces: ')
values = [float(value) for value in input().split()]

for value in values:
    sine = math.sin(value)
    print('The sine of ' + str(value) + ' is ' + str(sine))
