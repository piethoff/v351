import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mathe import matheplot as mp

plt.figure()
plt.savefig("build/plot.pdf")
plt.clf()

lin = np.linspace(-7, 7, 1000)


def fourierc(data, lin):
    sum = np.zeros(lin.size)
    for i in range(int(data[0].size)):
        sum += data[1][i]*np.cos(data[0][i]*lin)
    return sum

def fouriers(data, lin):
    sum = np.zeros(lin.size)
    for i in range(int(data[0].size)):
        sum += data[1][i]*np.sin(data[0][i]*lin)
    return sum

U0 = 20.59

data = np.genfromtxt("content/sägezahntab.txt", unpack=True)
print("Sägezahn")
for i in range(int(data[0].size)):
    print(data[0][i], " & ", data[1][i], " & ", 100*np.abs(data[1][i]/((2*U0)/(np.pi*data[0][i]))), r" \\", "\n", sep="", end="")
print()
data = np.genfromtxt("content/dreiecktab.txt", unpack=True)
print("Dreieck")
for i in range(int(data[0].size)):
    print(data[0][i], " & ", data[1][i], " & ", 100*np.abs(data[1][i]/((2*U0)/(np.pi*data[0][i]))), r" \\", "\n", sep="", end="")
print()
data = np.genfromtxt("content/rechtecktab.txt", unpack=True)
print("Rechteck")
for i in range(int(data[0].size)):
    print(data[0][i], " & ", data[1][i], " & ", 100*np.abs(data[1][i]/((2*U0)/(np.pi*data[0][i]))), r" \\", "\n", sep="", end="")
print()

