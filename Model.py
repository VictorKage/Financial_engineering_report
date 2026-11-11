import numpy as np
from scipy.integrate import simpson
import math

class FourierModels:
    def __init__(self, upper_bound, K, cf_model):
        self.upper_bound = upper_bound
        self.K = K
        self.cf_model = cf_model
        self.S_0 = cf_model.S_0
        self.r = cf_model.r
        self.T = cf_model.T
        self.characteristic_function = (
            cf_model.characteristic_function
        )
    
    def black_scholes_like_formula(self):
        self.log_K = np.log(self.K)

        def pi_1(u = np.linspace(1e-10, self.upper_bound, 1000)):
            pi_1 = 1/2 + (1/np.pi) * simpson(
                np.real(
                    np.exp(-1j * self.log_K * u) * self.characteristic_function(u - 1j ) / (1j * u * self.characteristic_function(-1j))
                ),
                u,
            )
            return pi_1
        
        def pi_2(u = np.linspace(1e-10, self.upper_bound, 1000)):
            pi_2 = 1/2 + (1/np.pi) * simpson(
                np.real(
                    np.exp(-1j * self.log_K * u) * self.characteristic_function(u) / (1j * u)
                ),
                u,
            )
            return pi_2

        call_price = self.S_0 * pi_1() - self.K * np.exp(-self.r * self.T) * pi_2()
        return call_price

    def Carr_and_Madan(self, alpha=1.5):
        self.log_K = np.log(self.K)

        def psi(u):
            psi = np.exp(-self.r * self.T) * self.characteristic_function(u - (alpha + 1) * 1j) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
            return psi

        def integrand(u = np.linspace(1e-10, self.upper_bound, 1000)):
            return simpson(np.real(np.exp(-1j * self.log_K * u) * psi(u)), u)

        call_price = (np.exp(-alpha * self.log_K) / np.pi) * integrand()
        return call_price

    def lewis_formula(self):
        k = np.log(self.K / self.S_0) - self.r *self.T

        def integrand(u = np.linspace(1e-10, self.upper_bound, 1000)):
            return simpson(np.real(np.exp(-1j * u * k) * self.characteristic_function_0(- u - 1j / 2)) * 1 / (u**2 + 1/4), u)

        call_price = self.S_0 - (np.sqrt(self.S_0 * self.K) * np.exp(-self.r * self.T / 2)) / np.pi * integrand()
        return call_price



class CharacteristicFunction:
    def __init__(self, model, T, S_0, r, **params):
        self.model = model
        self.T = T
        self.S_0 = S_0
        self.r = r
        self.params = params

        models = {
            "VarianceGamma": self.characteristic_function_VarianceGamma,
            "CGMY": self.characteristic_function_CGMY,
            "BlackScholes": self.characteristic_function_BlackScholes
        }

        if model not in models:
            raise ValueError(f"Unknown model: {model}")

        self.characteristic_function = models[model]

    def characteristic_function_VarianceGamma(self, u):
        theta = self.params["theta"]
        nu = self.params["nu"]
        sigma = self.params["sigma"]

        return (1 - 1j * theta * nu * u+ 0.5 * sigma**2 * nu * u**2) ** (self.T / nu)

    def characteristic_function_CGMY(self, u):
        C = self.params["C"]
        Y = self.params["Y"]
        M = self.params["M"]
        G = self.params["G"]

        return np.exp(
            self.T * C * math.gamma(-Y) *((M - 1j*u)**Y - M**Y+ (G + 1j*u)**Y - G**Y))

    def characteristic_function_BlackScholes(self, u):
        sigma = self.params["sigma"]

        mu = np.log(self.S_0) + (
            self.r - 0.5 * sigma**2
        ) * self.T

        return np.exp(1j * u * mu - 0.5 * sigma**2 * self.T * u**2)


if __name__ == "__main__":
    # Use case
    cf = CharacteristicFunction(
        model="CGMY",
        T=1,
        S_0=100,
        r=0.05,
        C=0.5,
        Y=0.5,
        M=5,
        G=5
    )

    fourier = FourierModels(
        upper_bound=200,
        K=100,
        cf_model=cf
    )

print(fourier.Carr_and_Madan())