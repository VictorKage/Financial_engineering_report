import pandas as pd
import numpy as np
from scipy.optimize import minimize

class InitialCleaner:
    def __init__(self):
        self.params = self.initial_data_cleaned()

    @staticmethod
    def initial_data_cleaned() -> dict:
        data = pd.read_excel("Data20191113.xlsx", header=None)
        spx_close = data.iat[2, 6]
        df = data.iloc[6:].copy()
        df.columns = data.iloc[5].tolist()

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")

        df["time_to_maturity"] = (
            df["exdate"] - df["date"]
        ).dt.days / 365
        return {'data': df, 'close_price': spx_close}
    
    def select_put_call_pairs(self, n: int = 5) -> pd.DataFrame:
        data = self.params['data']
        data["strike_diff"] = abs(data["strike"] - self.params['close_price'])
        data = data.sort_values(by=["exdate", "strike_diff"])

        nearest_strikes = (
            data.groupby("exdate")["strike"]
            .unique()
            .apply(lambda s: s[:n*2])
            .explode()
            .reset_index()
        )

        put_call_pairs = pd.merge(data, nearest_strikes, on=["exdate", "strike"])
        return put_call_pairs
    

class Calibrate(InitialCleaner):
    def __init__(self):
        super().__init__()
    
    def _pairs(self):
        data = self.select_put_call_pairs()
        return data

    def loss_function(self, fitting_params):
        q = fitting_params[0]
        r_values = fitting_params[1:]
        data = self.select_put_call_pairs()

        loss = 0

        maturities = data["exdate"].unique()

        for i, maturity in enumerate(maturities):
            r = r_values[i]

            maturity_data = data[data["exdate"] == maturity]

            for strike in maturity_data["strike"].unique():
                strike_data = maturity_data[maturity_data["strike"] == strike]
                observed_put_prices = strike_data[strike_data["cp_flag"] == "P"]["mid"].values
                observed_call_prices = strike_data[strike_data["cp_flag"] == "C"]["mid"].values

                if len(observed_put_prices) > 0 and len(observed_call_prices) > 0:
                    T = strike_data["time_to_maturity"].values[0]
                    theoretical_put_prices = self.theoretical_put_price(observed_call_prices, T, strike, r, q)
                    loss += np.sum((observed_put_prices - theoretical_put_prices) ** 2)
        return loss

    
    def theoretical_put_price(self, call_price, T, K, r, q):
        put_price = call_price + K * np.exp(-r * T) - self.params['close_price'] * np.exp(-q * T)
        return put_price


    def calibrate(self):
        data = self._pairs()
        n_maturities = len(data["exdate"].unique())

        # First parameter is q, remaining parameters are maturity-specific r
        initial_guess = [0.2] + [0.2] * n_maturities

        # Bounds for q and all interest rates
        bounds = [(0, None)] * (n_maturities + 1)

        result = minimize(
            self.loss_function,
            initial_guess,
            bounds=bounds
        )

        if result.success:
            calibrated_q = result.x[0]
            calibrated_r = result.x[1:]

            maturities = data["exdate"].unique()

            results = pd.DataFrame({
                "maturity": maturities,
                "calibrated_r": calibrated_r.round(5),
                "calibrated_q": calibrated_q.round(5)
            })
            # # save to csv
            results.to_csv("calibrated_parameters.csv", index=False)
            return results

        else:
            raise ValueError("Calibration failed: " + result.message)

class DataCleaner(InitialCleaner):
    def __init__(self):
        super().__init__()
        rates = pd.read_csv("calibrated_parameters.csv")
        self.r = dict(zip(pd.to_datetime(rates["maturity"]), rates["calibrated_r"]))
        self.q = float(rates["calibrated_q"].iloc[0])

    @staticmethod
    def to_numeric(data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col in ["best_bid", "best_offer", "mid", "strike", "time_to_maturity"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["spread"] = data["best_offer"] - data["best_bid"]
        return data

    @staticmethod
    def zero_bid_removal(data: pd.DataFrame) -> pd.DataFrame:
        rows = data.shape[0]
        cleaned_data = data[data["best_bid"] != 0]
        new_rows = cleaned_data.shape[0]
        print(f"Removed {rows - new_rows} rows with bid = 0")
        return cleaned_data
    
    def itm_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = data.shape[0]
        data = data[~((data["cp_flag"] == "C") & (data["strike"] < self.params['close_price']))]
        data = data[~((data["cp_flag"] == "P") & (data["strike"] > self.params['close_price']))]
        new_rows = data.shape[0]
        print(f"Removed {rows-new_rows} ITM")
        return data
    
    def puts_to_calls(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["original_bid"] = data["best_bid"]
        data["original_offer"] = data["best_offer"]
        data["original_mid"] = data["mid"]
        S = self.params['close_price']
        T = data["time_to_maturity"]
        data["r"] = data["exdate"].map(self.r)
        is_put = data["cp_flag"] == "P"
        shift = S * np.exp(-self.q * T) - data["strike"] * np.exp(-data["r"] * T)
        data["best_bid"] = np.where(is_put, data["best_bid"] + shift, data["best_bid"])
        data["best_offer"] = np.where(is_put, data["best_offer"] + shift, data["best_offer"])
        data["C"] = (data["best_bid"] + data["best_offer"]) / 2
        data["spread"] = (data["best_offer"] - data["best_bid"]).round(2)
        data = data.sort_values(["exdate", "strike"]).reset_index(drop=True)
        print(f"Puts converted to calls: {is_put.sum()}")
        print(f"Total call observations: {data.shape[0]}")
        return data
    

    def price_bound(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = data.shape[0]
        S = self.params['close_price']
        lower = np.maximum(0, S * np.exp(-self.q * data["time_to_maturity"])
                           - data["strike"] * np.exp(-data["r"] * data["time_to_maturity"]))
        data = data[data["C"] >= lower]
        print(f"Removed by price bounds: {rows - data.shape[0]}")
        return data

    @staticmethod
    def count_violations(maturity_data: pd.DataFrame, S: float, q: float, tol: float = 1e-8):
        T = maturity_data["time_to_maturity"].iloc[0]
        disc = np.exp(-maturity_data["r"].iloc[0] * T)
        C = np.concatenate(([S * np.exp(-q * T)], maturity_data["C"].to_numpy(float)))
        K = np.concatenate(([0.0], maturity_data["strike"].to_numpy(float)))
        counts = np.zeros(len(C), dtype=int)

        slope = (C[:-1] - C[1:]) / (K[1:] - K[:-1])
        mono = (slope < -tol) | (slope > disc + tol)
        counts[:-1] += mono
        counts[1:] += mono

        conv = (slope[:-1] - slope[1:]) < -tol
        counts[:-2] += conv
        counts[1:-1] += conv
        counts[2:] += conv
        return counts[1:], int(mono.sum()), int(conv.sum())
    
    def arbitrage_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = data.shape[0]
        S = self.params['close_price']
        n_mono, n_conv = 0, 0
        cleaned = []
        for maturity, maturity_data in data.groupby("exdate"):
            maturity_data = maturity_data.sort_values("strike")
            while len(maturity_data) >= 1:
                counts, mono, conv = self.count_violations(maturity_data, S, self.q)
                if counts.max() == 0:
                    break
                n_mono += mono
                n_conv += conv
                candidates = maturity_data[counts == counts.max()]                    # most violations
                candidates = candidates[candidates["spread"] == candidates["spread"].max()]  # widest spread
                maturity_data = maturity_data.drop(candidates.index[0])                  # first = lowest strike
            cleaned.append(maturity_data)
        data = pd.concat(cleaned)
        print(f"Removed by slope/convexity: {rows - data.shape[0]}")
        print(f"Monotonicity violations found: {n_mono}")
        print(f"Convexity violations found: {n_conv}")
        return data

    @staticmethod
    def short_maturity_removal(data: pd.DataFrame, min_days: int = 7) -> pd.DataFrame:
        rows = data.shape[0]
        data = data[data["time_to_maturity"] >= min_days / 365]
        print(f"Removed by maturity filter: {rows - data.shape[0]}")
        return data

    def clean_data(self):
        data = self.params['data']
        data = self.to_numeric(self.params['data'])
        print(f"Initial observations: {data.shape[0]}")
        data = self.zero_bid_removal(data)
        data = self.itm_removal(data)
        data = self.puts_to_calls(data)
        data = self.price_bound(data)
        data = self.arbitrage_removal(data)
        data = self.short_maturity_removal(data)
        print(f"Observations left: {data.shape[0]}")
        print(f"Remaining maturities: {data['exdate'].nunique()}")
        data.to_csv('cleaned_data.csv')
        return data

if __name__ == "__main__":
    # Calibrate().calibrate()
    DataCleaner().clean_data()