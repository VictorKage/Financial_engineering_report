import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.stats import norm

class FFTPricer:
    def __init__(self, S, q, rates, eta = 0.25, N_min: int = 2**12, N_max: int = 2**18, tol: float = 1e-3):
        self.S = S
        self.q = q
        self.eta = eta
        self.N_min = N_min
        self.rates = rates      
        self.N_max = N_max
        self.tol = tol

    @staticmethod
    def levy_exp_vg(u, sigma, nu, theta): #Samme som kappa(iu)
        return -(1/nu) * np.log(1 - 1j*theta*nu*u + (sigma**2*nu*u**2)/2)

    @staticmethod
    def levy_exp_cgmy(u, C, G, M, Y):
        return C * gamma(-Y) * ((M - 1j * u)**Y - M**Y + (G + 1j * u)**Y - G**Y)

    def levy_exponent_chooser(self, model, params):
        if model == "VG":
            return lambda u: self.levy_exp_vg(u, *params)
        elif model == "CGMY":
            return lambda u: self.levy_exp_cgmy(u, *params)
        elif model == "BS":
            (sigma,) = params
            return lambda u: -0.5 * sigma**2 * u**2
        raise ValueError("Model not found")

    def martingale_correction(self, model, params):
        psi = self.levy_exponent_chooser(model, params)
        return np.real(psi(-1j))

    # mangler bevidst e^(iu * (log S_0+(r-q)T) da det kræver r og beregnes for hver maturity
    def characteristic_function_part(self, model, params, T):
        kappa_1 = self.martingale_correction(model, params)
        psi = self.levy_exponent_chooser(model, params)
        return lambda u: np.exp(T * (psi(u) - 1j * u * kappa_1))

    def simpson_weights(self, N):
        j = np.arange(N)
        w = (self.eta / 3) * (3 + (-1) ** (j + 1))
        w[0] = self.eta / 3
        return w
    
    # Største alpha med E[S_T^(1+alpha)] < inf.
    @staticmethod
    def alpha_max(model: str, params) -> float:
        if model == "VG":
            sigma, nu, theta = params
            return np.sqrt(theta**2 / sigma**4 + 2 / (sigma**2 * nu)) - theta / sigma**2 - 1
        elif model == "CGMY":
            C, G, M, Y = params
            return M - 1
        return np.inf

    # den største mand Peter Carr siger de bruger 0.25 * bound
    def choose_alpha(self, model: str, params, alpha: float = 1.5) -> float:
        return min(alpha, 0.25 * self.alpha_max(model, params))

    def choose_N(self, cf, alpha: float) -> int:
        """Smallest power of 2 such that the truncation error at a = N*eta is below tol.

        Carr-Madan: the tail of the integral beyond a is roughly |phi(a)| / a, so the
        dollar error is about S * |phi(a - (alpha+1)i)| / (pi * a).
        """
        N = self.N_min
        while N < self.N_max:
            a = N * self.eta
            if self.S * abs(cf(a - (alpha + 1) * 1j)) / (np.pi * a) < self.tol:
                break
            N *= 2
        return N

    def carr_madan_fft(self, cf, T: float, r: float, alpha: float):
        """Call prices for S0 = 1 on a grid of log-moneyness k = log(K / S0).

        Returns (k_grid, prices_for_S0_equal_1).
        """
        eta = self.eta
        N = self.choose_N(cf, alpha)
        lam = 2 * np.pi / (N * eta) #spacing of log-strike grid
        b = N * lam / 2 #log strike levels
        j = np.arange(N)
        v = eta * j # integration grid v_j
        k = -b + lam * j # output grid

        u = v - (alpha + 1) * 1j
        phi = np.exp(1j * u * (r - self.q) * T) * cf(u)

        psi = np.exp(-r * T) * phi / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)

        w = self.simpson_weights(N)

        x = np.exp(1j * b * v) * psi * w
        prices = np.exp(-alpha * k) / np.pi * np.real(np.fft.fft(x))
        return k, prices

    def price(self, model: str, params, data: pd.DataFrame) -> np.ndarray:
        """Model call prices for every row of data (columns exdate, strike, time_to_maturity)."""
        alpha = self.choose_alpha(model, params)
        out = np.empty(len(data))
        for exdate, rows in data.groupby("exdate"):
            T = rows["time_to_maturity"].iloc[0]
            r = self.rates[exdate]
            k, c = self.carr_madan_fft(self.characteristic_function_part(model, params, T), T, r, alpha)
            keep = (k > -3) & (k < 3)                   # K/S0 between 0.05 and 20
            spline = CubicSpline(k[keep], c[keep])
            k_mkt = np.log(rows["strike"].to_numpy(float) / self.S)
            out[data.index.get_indexer(rows.index)] = self.S * spline(k_mkt)
        return out

    def price(self, model: str, params, data: pd.DataFrame) -> np.ndarray:
        alpha = self.choose_alpha(model, params)
        out = np.empty(len(data))
        for exdate, rows in data.groupby("exdate"):
            T = rows["time_to_maturity"].iloc[0]
            r = self.rates[exdate]
            cf = self.characteristic_function_part(model, params, T)
            k, c = self.carr_madan_fft(cf, T, r, alpha) # grid prices, S0 = 1
            keep = (k > -3) & (k < 3) # K/S0 between 0.05 and 20
            spline = CubicSpline(k[keep], c[keep])
            k_mkt = np.log(rows["strike"].to_numpy(float) / self.S)
            out[data.index.get_indexer(rows.index)] = self.S * spline(k_mkt)
        return out



    # OKAY BOYS NGL ALT DET HER NEDENUNDER ER 100% CLAUDE MEN IT SEEMS TO BE WORKING SÅ DET FINT 

    def benchmark_bs_call(self, K, T, r, sigma):
        d1 = (np.log(self.S / K) + (r - self.q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return self.S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def benchmark_lewis_formula(self, cf, K, T, r, u_max=1e6, n_pieces=200):
        F = self.S * np.exp((r - self.q) * T)
        k = np.log(F / K)

        def integrand(u):
            return np.real(np.exp(1j * u * k) * cf(u - 0.5j)) / (u**2 + 0.25)

        edges = np.concatenate(([0.0], np.geomspace(1.0, u_max, n_pieces)))
        integral = sum(quad(integrand, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
        return self.S * np.exp(-self.q * T) - np.sqrt(F * K) * np.exp(-r * T) / np.pi * integral

    def validate(self, data: pd.DataFrame):
        """Run the checks that go in the report's validation table."""
        T_all = sorted(data["time_to_maturity"].unique())
        exdates = sorted(data["exdate"].unique())
        results = []
        for exdate, T in [(exdates[0], T_all[0]), (exdates[-1], T_all[-1])]:
            r = self.rates[exdate]
            K = data.loc[data["exdate"] == exdate, "strike"].to_numpy(float)
            rows = data[data["exdate"] == exdate]

            # (a) BS through the FFT vs the BS formula
            fft = self.price("BS", (0.15,), rows)
            results.append(("BS: FFT vs formula", T, np.max(np.abs(fft - self.benchmark_bs_call(K, T, r, 0.15)))))

            # (b) VG through the FFT vs the Lewis formula (different formula, no alpha)
            vg = (0.12, 0.2, -0.15)
            fft = self.price("VG", vg, rows)
            cf = self.characteristic_function_part("VG", vg, T)
            lew = np.array([self.benchmark_lewis_formula(cf, Ki, T, r) for Ki in K[::10]])
            results.append(("VG: FFT vs Lewis", T, np.max(np.abs(fft[::10] - lew))))
        table = pd.DataFrame(results, columns=["check", "T", "max abs price error"])
        print(table.to_string(index=False))
        return table
    

if __name__ == '__main__':
    data = pd.read_csv("cleaned_data.csv", parse_dates=["exdate"])
    rates = pd.read_csv("calibrated_parameters.csv", parse_dates=["maturity"])
    S = pd.read_excel("Data20191113.xlsx", header=None).iat[2, 6]
    pricer = FFTPricer(S=S,
                           q=float(rates["calibrated_q"].iloc[0]),
                           rates=dict(zip(rates["maturity"], rates["calibrated_r"])))
    pricer.validate(data)