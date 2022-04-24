#-----------------------------------------------------------------------------#
# Import relevant python packages
#-----------------------------------------------------------------------------#
import math
import numpy as np
from numpy.fft import fft
from scipy.integrate import quad
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

#-----------------------------------------------------------------------------#
# Set your own model parameters
#-----------------------------------------------------------------------------#

S0 = 100.00 # initial index level
K = 100.00 # strike level
T = 1 # call option maturity
r = 0.02 # constant short rate
sigma = 0.3 # constant volatility of diffusion

#-----------------------------------------------------------------------------#
# Determine BSM option value by using the analytical formula
#-----------------------------------------------------------------------------#

def BSM_call_value(S0, K, T, r, sigma):
    ''' Valuation of European call option in BSM Model.
    --> Analytical Formula.
    Parameters
    ==========
    S0: float
    initial stock/index level
    K: float
    strike price
    T: float
    time-to-maturity (for t=0)
    r: float
    constant risk-free short rate
    sigma: float
    volatility factor in diffusion term
    Returns
    =======
    call_value: float
    European call option present value
    '''
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) \
    / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) \
    / (sigma * np.sqrt(T))
    BS_C = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
    - K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    return BS_C
print(f'BSM call value: ${round(BSM_call_value(S0, K, T, r, sigma),2)}')

def BSM_put_value(S0, K, T, r, sigma):
    c_value = BSM_call_value(S0, K, T, r, sigma)
    BS_P = c_value + np.exp(-r*T)*K - S0
    return BS_P

print(f'BSM put value: ${round(BSM_put_value(S0, K, T, r, sigma),2)}')

#-----------------------------------------------------------------------------#
# Fourier Transform with Numerical int_valueegration
#-----------------------------------------------------------------------------#

def BSM_call_value_INT(S0, K, T, r, sigma):
    ''' Valuation of European call option in BSM model via Lewis (2001)
    --> Fourier-based approach (integral).
    Parameters
    ==========
    S0: float
    initial stock/index level
    K: float
    strike price
    T: float
    time-to-maturity (for t=0)
    r: float
    constant risk-free short rate
    sigma: float
    volatility factor in diffusion term
    Returns
    =======
    call_value: float
    European call option present value
    '''
    int_value = quad(lambda u:
        BSM_integral_function(u, S0, K, T, r, sigma), 0, 100)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
    / np.pi * int_value)
    return call_value

def BSM_integral_function(u, S0, K, T, r, sigma):
    ''' Valuation of European call option in BSM model via Lewis (2001)
    --> Fourier-based approach: integral function. '''
    cf_value = BSM_characteristic_function(u - 1j * 0.5, 0.0, T, r, sigma)
    int_value = 1 / (u ** 2 + 0.25) \
    * (np.exp(1j * u * np.log(S0 / K)) * cf_value).real
    return int_value

def BSM_characteristic_function(v, x0, T, r, sigma):
    ''' Valuation of European call option in BSM model via
    Lewis (2001) and Carr-Madan (1999)
    --> Fourier-based approach: characteristic function. '''
    cf_value = np.exp(((x0 / T + r - 0.5 * sigma ** 2) * 1j * v
    - 0.5 * sigma ** 2 * v ** 2) * T)
    return cf_value

def BSM_put_value_INT(S0, K, T, r, sigma):
    C_value = BSM_call_value_INT(S0, K, T, r, sigma)
    put_value = C_value + np.exp(-r*T)*K - S0
    return put_value
print(f'BSM call value with numerical integration: ${round(BSM_call_value_INT(S0, K, T, r, sigma),2)}')
print(f'BSM put value with numerical integration: ${round(BSM_put_value_INT(S0, K, T, r, sigma),2)}')

#-----------------------------------------------------------------------------#
# Fourier Transform with FFT
#-----------------------------------------------------------------------------#

def BSM_call_value_FFT(S0, K, T, r, sigma):
    ''' Valuation of European call option in BSM model via Lewis (2001)
    --> Fourier-based approach (integral).
    Fourier-Based Option Pricing 117
    Parameters
    ==========
    S0: float
    initial stock/index level
    K: float
    strike price
    T: float
    time-to-maturity (for t=0)
    r: float
    constant risk-free short rate
    sigma: float
    volatility factor in diffusion term
    Returns
    =======
    call_value: float
    European call option present value
    '''
    k = np.log(K / S0)
    x0 = np.log(S0 / S0)
    g = 1 # factor to increase accuracy
    N = g * 4096
    eps = (g * 150.) ** -1
    eta = 2 * np.pi / (N * eps)
    b = 0.5 * N * eps - k
    u = np.arange(1, N + 1, 1)
    vo = eta * (u - 1)
    # Modifications to Ensure int_valueegrability
    if S0 >= 0.95 * K: # ITM case
        alpha = 1.5
        v = vo - (alpha + 1) * 1j
        modcharFunc = np.exp(-r * T) * (BSM_characteristic_function(
        v, x0, T, r, sigma) /
        (alpha ** 2 + alpha
        - vo ** 2 + 1j * (2 * alpha + 1) * vo))
    else: # OTM case
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        modcharFunc1 = np.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha))
        - np.exp(r * T) / (1j * (vo - 1j * alpha))
        - BSM_characteristic_function(
        v, x0, T, r, sigma) /
        ((vo - 1j * alpha) ** 2
        - 1j * (vo - 1j * alpha)))
        v = (vo + 1j * alpha) - 1j
        modcharFunc2 = np.exp(-r * T) * (1 / (1 + 1j * (vo + 1j * alpha))
        - np.exp(r * T) / (1j * (vo + 1j * alpha))
        - BSM_characteristic_function(
        v, x0, T, r, sigma) /
        ((vo + 1j * alpha) ** 2
        - 1j * (vo + 1j * alpha)))
    # Numerical FFT Routine
    delt = np.zeros(N, dtype=float)
    delt[0] = 1
    j = np.arange(1, N + 1, 1)
    SimpsonW = (3 + (-1) ** j - delt) / 3
    if S0 >= 0.95 * K:
        FFTFunc = np.exp(1j * b * vo) * modcharFunc * eta * SimpsonW
        payoff = (fft(FFTFunc)).real
        CallValueM = np.exp(-alpha * k) / np.pi * payoff
    else:
        FFTFunc = (np.exp(1j * b * vo)
        * (modcharFunc1 - modcharFunc2)
        * 0.5 * eta * SimpsonW)
        payoff = (fft(FFTFunc)).real
        CallValueM = payoff / (np.sinh(alpha * k) * np.pi)
    pos = int((k + b) / eps)
    CallValue = CallValueM[pos] * S0
    # klist = np.exp((np.arange(0, N, 1) - 1) * eps - b) * S0
    return CallValue #, klist[pos - 50:pos + 50]

def BSM_put_value_FFT(S0, K, T, r, sigma):
    call = BSM_call_value_FFT(S0, K, T, r, sigma)
    put = call + np.exp(-r*T)*K - S0
    return put
print(f'BSM call value with fourier transform: ${round(BSM_call_value_FFT(S0, K, T, r, sigma),2)}')
print(f'BSM put value with fourier transform: ${round(BSM_put_value_FFT(S0, K, T, r, sigma),2)}')

#-----------------------------------------------------------------------------#
# Plot the value differences
#-----------------------------------------------------------------------------#

def call_value_differences(vtype='fft'):
    print('Type any of the determinants to get the absolute difference (K,T,sigma)')
    det = input()
    if det == 'K' or det == 'k':
        k_list = np.linspace(S0 * 0.8, S0 * 1.2, 5)
        ana_values = BSM_call_value(S0, k_list, T, r, sigma)
        plt.figure(figsize=(8, 6))
        plt.subplot(311)
        plt.plot(k_list, ana_values, 'b', label='analytical', lw=1.5)
        if vtype == 'int':
            int_values = np.array([BSM_call_value_INT(S0, K, T, r, sigma)
            for K in k_list])
            plt.plot(k_list, int_values, 'r-.', label='Fourier (integral)',
            lw=1.5)
            diffs = int_values - ana_values
            rdiffs = (int_values - ana_values) / ana_values
        else:
            fft_values = np.array([BSM_call_value_FFT(S0, K, T, r, sigma)
            for K in k_list])
            plt.plot(k_list, fft_values, 'r-.', label='Fourier (FFT)', lw=1.5)
            diffs = fft_values - ana_values
            rdiffs = (fft_values - ana_values) / ana_values
            plt.legend()
            plt.grid()
            plt.subplot(312)
            plt.plot(k_list, diffs, 'g', label='abs. difference', lw=1.5)
        plt.legend(loc=0)
        plt.grid()
        plt.subplot(313)
        plt.plot(k_list, rdiffs, 'r', label='rel. difference', lw=1.5)
        plt.legend(loc=0)
        plt.xlabel('strike')
        plt.grid()
        plt.tight_layout()
    elif det == 'T' or det == 't':
        t_list = np.linspace(0.05, 2, 40)
        ana_values = BSM_call_value(S0, K, t_list, r, sigma)
        plt.figure(figsize=(8, 6))
        plt.subplot(311)
        plt.plot(t_list, ana_values, 'b', label='analytical', lw=1.5)
        if vtype == 'int':
            int_values = np.array([BSM_call_value_INT(S0, K, T, r, sigma)
            for T in t_list])
            plt.plot(t_list, int_values, 'r-.', label='Fourier (integral)',
            lw=1.5)
            diffs = int_values - ana_values
            rdiffs = (int_values - ana_values) / ana_values
        else:
            fft_values = np.array([BSM_call_value_FFT(S0, K, T, r, sigma)
            for T in t_list])
            plt.plot(t_list, fft_values, 'r-.', label='Fourier (FFT)', lw=1.5)
            diffs = fft_values - ana_values
            rdiffs = (fft_values - ana_values) / ana_values
            plt.legend()
            plt.grid()
            plt.subplot(312)
            plt.plot(t_list, diffs, 'g', label='abs. difference', lw=1.5)
        plt.legend(loc=0)
        plt.grid()
        plt.subplot(313)
        plt.plot(t_list, rdiffs, 'r', label='rel. difference', lw=1.5)
        plt.legend(loc=0)
        plt.xlabel('time to maturity')
        plt.grid()
        plt.tight_layout()
    elif det == 'Sigma' or det == 'sigma':
        s_list = np.linspace(0.05,0.5, 10)
        ana_values = BSM_call_value(S0, K, T, r, s_list)
        plt.figure(figsize=(8, 6))
        plt.subplot(311)
        plt.plot(s_list, ana_values, 'b', label='analytical', lw=1.5)
        if vtype == 'int':
            int_values = np.array([BSM_call_value_INT(S0, K, T, r, sigma)
            for sigma in s_list])
            plt.plot(s_list, int_values, 'r-.', label='Fourier (integral)',
            lw=1.5)
            diffs = int_values - ana_values
            rdiffs = (int_values - ana_values) / ana_values
        else:
            fft_values = np.array([BSM_call_value_FFT(S0, K, T, r, sigma)
            for sigma in s_list])
            plt.plot(s_list, fft_values, 'r-.', label='Fourier (FFT)', lw=1.5)
            diffs = fft_values - ana_values
            rdiffs = (fft_values - ana_values) / ana_values
            plt.legend()
            plt.grid()
            plt.subplot(312)
            plt.plot(s_list, diffs, 'g', label='abs. difference', lw=1.5)
        plt.legend(loc=0)
        plt.grid()
        plt.subplot(313)
        plt.plot(s_list, rdiffs, 'r', label='rel. difference', lw=1.5)
        plt.legend(loc=0)
        plt.xlabel('stock volatilities')
        plt.grid()
        plt.tight_layout()
    print('\n')
    msg = f' Absolute differences with put-call parity: {diffs}'
    print(msg)

call_value_differences(vtype='fft')

def put_value_difference(S0, K, T, r, sigma):
    BSM_p = BSM_put_value(S0, K, T, r, sigma)
    for_p = BSM_put_value_FFT(S0, K, T, r, sigma)
    print('\n')
    msg = f' Absolute difference with put-call parity is: {abs(BSM_p - for_p)}'
    print(msg)
















