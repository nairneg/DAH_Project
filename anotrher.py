import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.optimize import minimize
# import data
#xmass = np.loadtxt(sys.argv[1])
f = open("ups-15-small.bin","r")
datalist = np.fromfile(f,dtype=np.float32)

# number of events
nevent = int(len(datalist)/6)

xdata = np.split(datalist,nevent)

xdata = datalist.reshape(nevent, 6)


def extract_variables(data):

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

entries, bedges, _ = plt.hist(inv_mass_regionI, bins=500, color='cyan')
plt.xlim(9.3, 9.6)
plt.xlabel(r"Invariant mass ($\frac{GeV}{c^2}$)")
plt.ylabel("Entries")
plt.title(r"$\Upsilon$ (1S) region (data)")
plt.show()

bin_centers = 0.5 * (bedges[:-1] + bedges[1:])  #midpoints of bins


def normalized_gauss(x, mu, sigma, a, b):

    integral, _ = integrate.quad(lambda xx: np.exp(-0.5*((xx - mu)/sigma)**2), a, b)
    N = 1.0 / (sigma * np.sqrt(2*np.pi) * integral)
    return N * np.exp(-0.5 * ((x - mu)/sigma)**2)

def normalized_exp(x, lamb, a, b):

    if abs(lamb) < 1e-9: 
        return np.full_like(x, 1/ (b - a))
    integral, _ = integrate.quad(lambda xx: np.exp(-lamb * xx), a, b)
    norm = lamb * np.exp(-lamb*x) / integral
    return norm

def comp_model(x, mu, sigma, lamb, f_s, a=Ilower_bound, b=Iupper_bound):
    
    return f_s * normalized_gauss(x, mu, sigma, a, b) + (1 - f_s) * normalized_exp(x, lamb, a, b)


ydata = entries / np.trapz(entries, bin_centers) # normalize histogram to form a PDF

#initial parameter guesses and bounds
p0 = [9.46, 0.05, 2.0, 0.5]
bounds = ([9.3, 0.01, 0.1, 0.0], [9.6, 0.2, 10.0, 1.0])


parameters, pcov = curve_fit(lambda x, mu, sigma, lamb, f_s: comp_model(x, mu, sigma, lamb, f_s), bin_centers, ydata, p0=p0, bounds=bounds)
mu_fit, sigma_fit, lamb_fit, f_s_fit = parameters



print(f"μ = {mu_fit:.4f} GeV")
print(f"σ = {sigma_fit:.4f} GeV")
print(f"λ = {lamb_fit:.3f} GeV⁻¹")
print(f"f_s = {f_s_fit:.3f}")


m_plot = np.linspace(Ilower_bound, Iupper_bound, 500)
#creates an array of mass values between the lower and upper bounds for plotting

plt.hist(inv_mass_regionI, bins=500, density=True, color='cyan', label='Data')
plt.plot(m_plot, comp_model(m_plot, mu_fit, sigma_fit, lamb_fit, f_s_fit),'black', label='Gaussian + Exponential (fit)')
plt.xlim(9.3, 9.6)
plt.xlabel(r"Invariant mass ($\frac{GeV}{c^2}$)")
plt.ylabel("Probability density")
plt.title(r"$\Upsilon$(1S) composite normalized fit")
plt.legend()
plt.show()


#checks for normalization of composite PDF
area, error = integrate.quad(lambda xx: comp_model(xx, mu_fit, sigma_fit, lamb_fit, f_s_fit, Ilower_bound, Iupper_bound), Ilower_bound, Iupper_bound)
#print(f"Composite PDF integral = {area} ± {error}")



def neg_log_likelihood(params, data, a, b):
    mu, sigma, lamb, f_s = params


    if sigma <= 0 or f_s < 0 or f_s > 1 or lamb < 0:
        return np.inf

    pdf_vals = comp_model(data, mu, sigma, lamb, f_s, a, b)

    # Avoid log(0)
    pdf_vals = np.clip(pdf_vals, 1e-12, None)
    nll = -np.sum(np.log(pdf_vals))
    return nll

# Initial guesses and bounds
init = [9.46, 0.05, 1.5, 0.5]  
bounds = [(9.3, 9.6), (0.005, 0.2), (0.001, 10.0), (0.0, 1.0)]


res = minimize(neg_log_likelihood, init,
               args=(inv_mass_regionI, Ilower_bound, Iupper_bound),
               bounds=bounds, method='L-BFGS-B')

# Extract fitted parameters
mu_fit, sigma_fit, lamb_fit, f_s_fit = res.x
print("Minimized negative log-likelihood parameters:")
print(f"  μ     = {mu_fit:.5f} GeV")
print(f"  σ     = {sigma_fit:.5f} GeV")
print(f"  λ     = {lamb_fit:.5f} GeV⁻¹")
print(f"  f_s   = {f_s_fit:.5f}")
print("Negative log-likelihood =", res.fun)


xplot = np.linspace(0, 1000, 800)
plt.plot(xplot, comp_model(xplot, mu_fit, sigma_fit, lamb_fit, f_s_fit),
         'r-', label='Composite Fit')
plt.plot(xplot, f_s_fit * normalized_gauss(xplot, mu_fit, sigma_fit, 9.3, 9.6), 'b--', label='Signal Gaussian')
plt.plot(xplot, (1 - f_s_fit) * normalized_exp(xplot, lamb_fit, 9.3, 9.6), 'g--', label='Background Exp')

plt.xlabel("Invariant Mass (GeV/c²)")
plt.ylabel("Probability Density")
plt.title("Υ(1S) Maximum-Likelihood Fit")
plt.legend()
plt.tight_layout()
plt.show()