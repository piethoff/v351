import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants.constants as const
from uncertainties import ufloat
from uncertainties import unumpy

mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

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

def f(x, A, B):
    return A*x**B


U0 = 20.59

data = np.genfromtxt("content/sägezahntab.txt", unpack=True)
print("Sägezahn")
for i in range(int(data[0].size)):
    print(data[0][i], " & ", data[1][i], " & ", 100*np.abs(data[1][i]/((2*U0)/(np.pi*data[0][i]))-1), r" \\", "\n", sep="", end="")
print()
params, covar = curve_fit(f, data[0], data[1])
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print(uparams, "\n")
lin = np.linspace(data[0][0], data[0][-1], 10000)
plt.plot(lin, f(lin, *params), label="Regression")
plt.plot(data[0], data[1], ".", label="Messwerte")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel(r"$b_n/\si{\volt}$")
plt.grid(which="both")
plt.legend()
plt.tight_layout()
plt.savefig("build/sagezahn.pdf")
plt.clf()
data = np.genfromtxt("content/dreiecktab.txt", unpack=True)
print("Dreieck")
for i in range(int(data[0].size)):
    print(data[0][i], " & ", data[1][i], " & ", 100*np.abs(data[1][i]/((8*U0)/(np.pi**2*(data[0][i])**2))-1), r" \\", "\n", sep="", end="")
print()
params, covar = curve_fit(f, data[0], data[1])
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print(uparams, "\n")
lin = np.linspace(data[0][0], data[0][-1], 10000)
plt.plot(lin, f(lin, *params), label="Regression")
plt.plot(data[0], data[1], ".", label="Messwerte")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel(r"$a_n/\si{\volt}$")
plt.grid(which="both")
plt.tight_layout()
plt.savefig("build/dreieck.pdf")
plt.clf()
data = np.genfromtxt("content/rechtecktab.txt", unpack=True)
print("Rechteck")
for i in range(int(data[0].size)):
    print(data[0][i], " & ", data[1][i], " & ", 100*np.abs(data[1][i]/((4*U0)/(np.pi*data[0][i]))-1), r" \\", "\n", sep="", end="")
print()
params, covar = curve_fit(f, data[0], data[1])
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print(uparams, "\n")
lin = np.linspace(data[0][0], data[0][-1], 10000)
plt.plot(lin, f(lin, *params), label="Regression")
plt.plot(data[0], data[1], ".", label="Messwerte")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel(r"$b_n/\si{\volt}$")
plt.grid(which="both")
plt.tight_layout()
plt.savefig("build/rechteck.pdf")
plt.clf()
