from DataCleaner import DataCleaner
import pandas as pd
import numpy as np

class PC_Calibration:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.cleaned_data, self.spy_close = self.cleaner.clean_data_for_put_call_parity()


    def loss_function(self, params):
        q = params[0]
        r_values = params[1:]

        loss = 0

        maturities = self.cleaned_data["exdate"].unique()

        for i, maturity in enumerate(maturities):
            r = r_values[i]

            maturity_data = self.cleaned_data[self.cleaned_data["exdate"] == maturity]

            for strike in maturity_data["strike"].unique():
                strike_data = maturity_data[maturity_data["strike"] == strike]
                observed_put_prices = strike_data[strike_data["cp_flag"] == "P"]["mid"].values
                observed_call_prices = strike_data[strike_data["cp_flag"] == "C"]["mid"].values

                if len(observed_put_prices) > 0 and len(observed_call_prices) > 0:
                    T = strike_data["time_to_maturity"].values[0]
                    theoretical_put_prices = self.PC_Prices(observed_call_prices, T, strike, r, q)
                    loss += np.sum((observed_put_prices - theoretical_put_prices) ** 2)

        return loss

    
    def PC_Prices(self, Call_price, T, K, r, q):
        PC_prices = Call_price + K * np.exp(-r * T) - self.spy_close * np.exp(-q * T)
        return PC_prices


    def calibrate(self):
        from scipy.optimize import minimize

        n_maturities = len(self.cleaned_data["exdate"].unique())

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

            maturities = self.cleaned_data["exdate"].unique()

            results = pd.DataFrame({
                "maturity": maturities,
                "calibrated_r": calibrated_r,
                "calibrated_q": calibrated_q
            })

            # save to csv
            results.to_csv("calibrated_parameters.csv", index=False)
            return results

        else:
            raise ValueError("Calibration failed: " + result.message)


if __name__ == "__main__":
    pc_calibration = PC_Calibration()
    results = pc_calibration.calibrate()
    print(results)

