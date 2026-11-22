import time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq
from scipy.special import gamma
from scipy.stats import norm
from fft_pricer import FFTPricer


def bs_price(S, K, T, r, q, sigma, cp):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, q, cp):
    try:
        return brentq(lambda s: bs_price(S, K, T, r, q, s, cp) - price, 1e-4, 5.0)
    except ValueError:
        return np.nan

class Fitter:
    PARAMS = {
        "VG":   ["sigma", "nu", "theta"],
        "CGMY": ["C", "G", "M", "Y"],
    }
    BOUNDS = {
        "VG":   ([0.01, 0.001, -1.0], [1.0, 5.0, 1.0]),
        "CGMY": ([1e-4, 0.5, 1.5, 0.01], [100.0, 100.0, 100.0, 1.9]),
    }
    PENALTY = 1e3

    def __init__(self, pricer: FFTPricer, data: pd.DataFrame, vega_floor: float = 1.0):
        self.pricer = pricer
        self.S, self.q = pricer.S, pricer.q
        self.data = data.reset_index(drop=True).copy()
        self.vega_floor = vega_floor
        self._add_iv_and_vega()

    def _add_iv_and_vega(self):
        d = self.data
        d["iv_mkt"] = [implied_vol(row.original_mid, self.S, row.strike, row.time_to_maturity,
                                   row.r, self.q, row.cp_flag) for row in d.itertuples()]
        d["vega"] = bs_vega(self.S, d["strike"], d["time_to_maturity"], d["r"], self.q, d["iv_mkt"])
        d["w"] = 1.0 / np.maximum(d["vega"], self.vega_floor) # vega loss weight
        n_bad = d["iv_mkt"].isna().sum()
        if n_bad:
            print(f"  dropping {n_bad} rows with no market IV")
            self.data = d.dropna(subset=["iv_mkt"]).reset_index(drop=True)

    @staticmethod
    def is_valid(model: str, params) -> bool:
        #Vi tjekker om E[exp(X_1)]<inf
        if model == "VG":
            sigma, nu, theta = params
            return 1 - theta * nu - 0.5 * sigma**2 * nu > 0
        if model == "CGMY":
            C, G, M, Y = params
            return M > 1 and abs(Y - 1) > 1e-6
        return True

    def residuals(self, params, model: str, loss: str, data: pd.DataFrame = None):
        d = self.data if data is None else data
        if not self.is_valid(model, params):
            return np.full(len(d), self.PENALTY)
        model_prices = self.pricer.price(model, params, d)
        if not np.all(np.isfinite(model_prices)):
            return np.full(len(d), self.PENALTY)
        err = model_prices - d["C"].to_numpy()
        if loss == "SSE":
            return err
        else:
            return err*d["w"].to_numpy()

    @staticmethod
    def starting_points(model: str, target_vol: float = 0.18):
        if model == "VG":
            return [np.array([s, n, t]) for s in (0.12, 0.2)
                    for n in (0.1, 0.5) for t in (-0.3, -0.1)]
        starts = []
        for Y in (0.2, 0.6, 0.9, 1.3):
            for G, M in ((5.0, 15.0), (10.0, 30.0)):
                C = target_vol**2 / (gamma(2 - Y) * (M**(Y - 2) + G**(Y - 2)))
                starts.append(np.array([C, G, M, Y]))
        return starts

    def fit(self, model: str, loss: str, data: pd.DataFrame = None, verbose: bool = True):
        lo, hi = self.BOUNDS[model]
        best = None
        t0 = time.time()
        for x0 in self.starting_points(model):
            x0 = np.clip(x0, np.array(lo) + 1e-8, np.array(hi) - 1e-8)
            res = least_squares(self.residuals, x0, bounds=(lo, hi), args=(model, loss, data),
                                method="trf", x_scale="jac", diff_step=1e-4,
                                xtol=1e-10, ftol=1e-10, max_nfev=400)
            if best is None or res.cost < best.cost:
                best = res
        out = {"model": model, "loss": loss, "x": best.x, "objective": 2 * best.cost,
               "nfev": best.nfev, "seconds": time.time() - t0}
        out.update(dict(zip(self.PARAMS[model], best.x)))
        return out

    def model_ivs(self, model: str, params, data: pd.DataFrame = None) -> np.ndarray:
        d = self.data if data is None else data
        C = self.pricer.price(model, params, d)
        K, T, r = d["strike"].to_numpy(), d["time_to_maturity"].to_numpy(), d["r"].to_numpy()
        is_put = (d["cp_flag"] == "P").to_numpy()
        otm = np.where(is_put, C - self.S * np.exp(-self.q * T) + K * np.exp(-r * T), C)
        return np.array([implied_vol(p, self.S, k, t, rr, self.q, "P" if ip else "C")
                         for p, k, t, rr, ip in zip(otm, K, T, r, is_put)])

    def diagnostics(self, model: str, params, data: pd.DataFrame = None) -> dict:
        d = self.data if data is None else data
        C = self.pricer.price(model, params, d)
        iv = self.model_ivs(model, params, d)
        err = C - d["C"].to_numpy()
        iv_err = iv - d["iv_mkt"].to_numpy()
        inside = (C >= d["best_bid"].to_numpy()) & (C <= d["best_offer"].to_numpy())
        return {"price RMSE ($)": np.sqrt(np.mean(err**2)),
                "price MAE ($)": np.mean(np.abs(err)),
                "IV RMSE (vol pts)": 100 * np.sqrt(np.nanmean(iv_err**2)),
                "IV MAE (vol pts)": 100 * np.nanmean(np.abs(iv_err)),
                "inside bid-ask (%)": 100 * inside.mean()}

    def iv_rmse_by_maturity(self, model: str, params) -> pd.Series:
        iv = self.model_ivs(model, params)
        e2 = (iv - self.data["iv_mkt"].to_numpy())**2
        return 100 * np.sqrt(pd.Series(e2).groupby(self.data["exdate"].to_numpy()).mean())


if __name__ == "__main__":
    data = pd.read_csv("cleaned_data.csv", parse_dates=["exdate"])
    rates = pd.read_csv("calibrated_parameters.csv", parse_dates=["maturity"])
    S = pd.read_excel("Data20191113.xlsx", header=None).iat[2, 6]
    pricer = FFTPricer(S=S,
                           q=float(rates["calibrated_q"].iloc[0]),
                           rates=dict(zip(rates["maturity"], rates["calibrated_r"])))
    fitter = Fitter(pricer, data, vega_floor=1.0)

    results, by_maturity = [], {}
    for model in ("VG", "CGMY"):
        for loss in ("SSE", "VEGA"):
            res = fitter.fit(model, loss)
            res.update(fitter.diagnostics(model, res["x"]))
            results.append(res)
            by_maturity[f"{model} {loss}"] = fitter.iv_rmse_by_maturity(model, res["x"])

    table = pd.DataFrame(results).drop(columns="x")
    pd.set_option("display.width", 200)
    print("\n", table.round(4).to_string(index=False))
    print("\nIV RMSE by maturity (vol points)\n", pd.DataFrame(by_maturity).round(2))

    table.to_csv("calibration_results.csv", index=False)
    pd.DataFrame(by_maturity).to_csv("iv_rmse_by_maturity.csv")

    out = fitter.data[["exdate", "strike", "time_to_maturity", "cp_flag", "C", "iv_mkt", "vega"]].copy()
    for res in results:
        out[f"iv_{res['model']}_{res['loss']}"] = fitter.model_ivs(res["model"], res["x"])
    out.to_csv("fitted_ivs.csv", index=False)
