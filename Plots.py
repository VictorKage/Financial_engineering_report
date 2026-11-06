# Vi skal nok lave plots på enkelte maturities og strikes. 
# vi skal nok og lave en 

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


class FE_Plots:
    def __init__(self, plot_df):
        self.plot_df = plot_df

    def bs_sholes_prices(self, S_0, K, T, r, sigma, q, type):
        d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if type == "call":
            price = S_0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif type == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * np.exp(-q * T) * norm.cdf(-d1)
        return price



   

    def cross_section_plot(self, maturity, strike):
        pass

    def surface_plot(self):
        pass