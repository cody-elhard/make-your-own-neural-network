import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot

def avg(x, y):
    average = (x + y) / 2
    return average

print(avg(5, 15))

# matplotlib.test()

a = numpy.zeros([3, 2])

print(matplotlib.pyplot.imshow(a, interpolation="nearest"))
