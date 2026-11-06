# Vi skal nok lave plots på enkelte maturities og strikes. 
# vi skal nok og lave en 

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from DataCleaner import DataCleaner


class Imp_Vol_Plots:
    def __init__(self, df, S_0):
        self.df = df
        self.S_0 = S_0

    def bs_sholes_prices(self, S_0, K, T, r, sigma, q, type):
        d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if type == "C":
            price = S_0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif type == "P":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * np.exp(-q * T) * norm.cdf(-d1)
        return price

    def solve_for_iv(self, S, K, T,  market_price, type, r, q):
        # Initial guess for implied volatility
        sigma = 0.2
        tolerance = 1e-6
        max_iterations = 100

        for i in range(max_iterations):
            price = self.bs_sholes_prices(S, K, T, r, sigma, q, type)
            vega = S * np.exp(-q * T) * norm.pdf((np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)

            price_diff = price - market_price

            if abs(price_diff) < tolerance:
                return sigma

            sigma -= price_diff / vega

        return None 

    def plot_implied_volatility(self):
        ivs = []
        moneyness = []
        dtes = []
        S = self.S_0

        for idx, row in self.df.iterrows():
            K = row['strike']
            T = row['time_to_maturity']
            market_price = row['mid']
            type = row['cp_flag']
            r = 0.2
            q = 0.2

            iv = self.solve_for_iv(S, K, T, market_price, type, r, q)
            if iv is not None:
                ivs.append(iv)
                moneyness.append(K / S)
                dtes.append(T)

        # 3D plot of implied volatility
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(moneyness, dtes, ivs, c=ivs, cmap='viridis', marker='o')
        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Implied Volatility (IV)')
        ax.set_title('Implied Volatility Surface')
        plt.show()

if __name__ == "__main__":
    data_cleaner = DataCleaner()
    df, S = data_cleaner.clean_data_for_calibration()
    # Only keeping call options for the implied volatility plot
    df = df[df['cp_flag'] == 'C']
    iv_plotter = Imp_Vol_Plots(df, S)
    iv_plotter.plot_implied_volatility()