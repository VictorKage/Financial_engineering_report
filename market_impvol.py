import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata


class MarketImpVol:
    def __init__(self):
        self.S = self.spx_close()
        self.data = pd.read_csv("cleaned_data.csv", index_col=0)
        self.q = pd.read_csv("calibrated_parameters.csv")['calibrated_q'].values[0]

    @staticmethod
    def spx_close() -> float:
        data = pd.read_excel("Data20191113.xlsx", header=None)
        spx_close = data.iat[2, 6]
        return spx_close
    
    @staticmethod
    def bs_price(S,K,T,r,q,sigma, cp_flag):
        d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1-sigma*np.sqrt(T)
        if cp_flag == "C":
            return S* np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)

    @staticmethod  
    def bs_vega(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    def implied_volatility(self,c_obs,S,K,T,r,q,cp):
        try:
            return brentq(lambda s: self.bs_price(S,K,T,r,q,s,cp) - c_obs, 1e-4, 5.0, xtol=1e-10)
        except ValueError:
            return np.nan
        
    def create_iv_col(self, data, col, colname):
        data[colname] = data.apply(lambda x: self.implied_volatility(x[col], self.S, x["strike"], x["time_to_maturity"], x["r"], self.q, x["cp_flag"]), axis=1)
        return data
    
    def infer_impvols(self) -> pd.DataFrame:
        cleaned_data = self.data
        cleaned_data = self.create_iv_col(cleaned_data, "original_bid", "iv_bid")
        cleaned_data = self.create_iv_col(cleaned_data, "original_offer", "iv_offer")
        cleaned_data = self.create_iv_col(cleaned_data, "original_mid", "iv_mid")
        cleaned_data["vega"] = self.bs_vega(self.S, cleaned_data["strike"], cleaned_data["time_to_maturity"],
                       cleaned_data["r"], self.q, cleaned_data["iv_mid"])
        return cleaned_data
    
    def save_ivdata(self):
        return self.infer_impvols().to_csv("iv_data.csv")

    def plot_volatility_surface(self, interpolate: bool = False, include_market_obs: bool= False, forward_lm: bool = False):
        iv_data = self.infer_impvols()
        F = self.S * np.exp(
            (iv_data["r"] - self.q) * iv_data["time_to_maturity"]
        )
        if forward_lm:
            iv_data["moneyness"] = np.log(
                iv_data["strike"] / F
            )
        else:
            iv_data["moneyness"] = np.log(
                iv_data["strike"] / self.S
            )
        iv_data = iv_data.dropna(
        subset=["strike", "time_to_maturity", "iv_mid"]
        )

        K = iv_data["moneyness"].values
        T = iv_data["time_to_maturity"].values
        IV = iv_data["iv_mid"].values
        if interpolate:
            K_grid = np.linspace(K.min(), K.max(), 100)
            T_grid = np.linspace(T.min(), T.max(), 100)

            K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

            IV_mesh = griddata(
                (K, T),
                IV,
                (K_mesh, T_mesh),
                method="linear"
            )

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

            surface = ax.plot_surface(
                K_mesh,
                T_mesh,
                IV_mesh,
                cmap="viridis",
                edgecolor="none",
                alpha=0.9
            )
        else:
            points = pd.DataFrame({
                "K": K,
                "T": T,
                "IV": IV
            }).groupby(["K", "T"], as_index=False)["IV"].mean()

            K = points["K"].values
            T = points["T"].values
            IV = points["IV"].values
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

            surface = ax.plot_trisurf(
                K, T, IV,
                cmap="viridis",
                edgecolor="none",
                alpha=0.9
            )
        if include_market_obs:
            ax.scatter(
                K, T, IV,
                color="red",
                s=12,
                alpha=0.9,
                label="Market observations",
                depthshade=False
            )

            ax.legend()

        if forward_lm:
            ax.set_xlabel("Forward log moneyness log(K/F)")
        else:
            ax.set_xlabel("Log moneyness log(K/S)")
        ax.set_ylabel("Time to Maturity (T)")
        ax.set_zlabel("Implied Volatility")

        ax.set_title("SPX Options Implied Volatility Surface")

        fig.colorbar(surface, ax=ax, shrink=0.6, label="Implied Volatility")

        plt.tight_layout()
        plt.savefig("impvolsurface.png")
        plt.show()
        return
    
if __name__ == "__main__":
    market_iv = MarketImpVol()
    market_iv.plot_volatility_surface(interpolate=True, include_market_obs=False, forward_lm=False)
    market_iv.save_ivdata()

