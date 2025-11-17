from matplotlib.pylab import norm
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line
import scipy
from scipy.optimize import curve_fit
from scipy import integrate
#from scipy.optimize import minimize
import iminuit
from iminuit import minimize  # has same interface as scipy.optimize.minimize
from iminuit import Minuit, describe
from iminuit.cost import LeastSquares
from iminuit.typing import Annotated, Gt
from scipy.stats import crystalball
   from scipy.special import erf

#print("iminuit version:", iminuit.__version__)
# import data
#xmass = np.loadtxt(sys.argv[1])
f = open('ups-15-small.bin',"r")
datalist = np.fromfile(f,dtype=np.float32)

# number of events
nevent = int(len(datalist)/6)

xdata = np.split(datalist,nevent)
print(xdata[0])


xdata = datalist.reshape(nevent, 6)

# extract variable arrays

def extract_variables(data):
    """Extracts the 6 variable arrays (columns) from event data."""
    list_array = [data[:, i] for i in range(data.shape[1])]
    return list_array


list_array = extract_variables(xdata)

xmass = list_array[0]
xpair_trans_mom = list_array[1]
xrapidity = list_array[2]
xpair_mom = list_array[3]
xfirst_mom = list_array[4]
xsecond_mom = list_array[5]




Ilower_bound, Iupper_bound = 9.3, 9.6
inv_mass_regionI = np.array([m for m in xmass if Ilower_bound <= m <= Iupper_bound])




def normalized_gauss(x, mu, sigma, a, b):
    # Normalization constant for truncated Gaussian

    A = (a - mu) / (sigma * np.sqrt(2))
    B = (b - mu) / (sigma * np.sqrt(2))
    Z = 0.5 * (erf(B) - erf(A))

    return (np.exp(-0.5*((x-mu)/sigma)**2) /
            (sigma * np.sqrt(2*np.pi) * Z))


def normalized_exp(x, lamb, a, b):
    # Normalized exponential on [a,b]
    Z = 1 - np.exp(-lamb * (b - a))
    return (lamb * np.exp(-lamb * (x - a))) / Z

def comp_model(x, mu, sigma, lamb, f_s, a, b):
    return f_s * normalized_gauss(x, mu, sigma, a, b) + \
           (1 - f_s) * normalized_exp(x, lamb, a, b)




def nll(mu, sigma, lamb, f_s):
    pdf_vals = comp_model(inv_mass_regionI, mu, sigma, lamb, f_s,
                          Ilower_bound, Iupper_bound)

    # avoid log(0)
    if np.any(pdf_vals <= 0):
        return 1e12

    return -np.sum(np.log(pdf_vals))

m = Minuit(nll, mu=9.46, sigma=0.05, lamb=2.0, f_s=0.5)
m.limits = ((9.3, 9.6), (0.01, 0.2), (0.1, 10.0), (0.0, 1.0))

m.migrad()
m.hesse()

print(m.values)
print(m.errors)

plt.hist(inv_mass_regionI, bins=200, range=(Ilower_bound, Iupper_bound),
         density=True, alpha=0.4)



# Overlay fitted PDF
xx = np.linspace(Ilower_bound, Iupper_bound, 2000)
yy = comp_model(xx, *m.values, Ilower_bound, Iupper_bound)
plt.plot(xx, yy, 'r-', lw=2)
plt.show()


check = np.trapz(yy, xx)
print("Integral of fitted PDF over fit range:", check)
