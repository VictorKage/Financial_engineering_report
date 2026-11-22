import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fft_pricer import FFTPricer
from model_calibration import Fitter, implied_vol

STYLE = {
    "market": dict(color="#55534e", s=10, zorder=3, label="Market"),
    "VG":     dict(color="#2a78d6", lw=2.0, ls="-",  label="VG"),
    "CGMY":   dict(color="#eb6834", lw=2.0, ls="--", label="CGMY"),
}


def model_smile(pricer: FFTPricer, model: str, params, exdate, T: float, r: float,
                strikes: np.ndarray) -> np.ndarray:
    grid = pd.DataFrame({"exdate": exdate, "strike": strikes, "time_to_maturity": T})
    C = pricer.price(model, params, grid)
    S, q = pricer.S, pricer.q
    F = S * np.exp((r - q) * T)
    ivs = []
    for K, c in zip(strikes, C):
        if K < F:
            put = c - S * np.exp(-q * T) + K * np.exp(-r * T)
            ivs.append(implied_vol(put, S, K, T, r, q, "P"))
        else:
            ivs.append(implied_vol(c, S, K, T, r, q, "C"))
    return np.array(ivs)


def plot_smiles(fitter: Fitter, params: dict, filename: str, n_grid: int = 200):
    pricer, data = fitter.pricer, fitter.data
    exdates = sorted(data["exdate"].unique())
    n_cols = 3
    n_rows = int(np.ceil(len(exdates) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.6 * n_rows))

    for ax, exdate in zip(axes.flat, exdates):
        rows = data[data["exdate"] == exdate]
        T = rows["time_to_maturity"].iloc[0]
        r = pricer.rates[exdate]
        F = pricer.S * np.exp((r - pricer.q) * T)

        ax.scatter(np.log(rows["strike"] / F), 100 * rows["iv_mkt"], **STYLE["market"])

        strikes = np.linspace(rows["strike"].min(), rows["strike"].max(), n_grid)
        for model, x in params.items():
            iv = model_smile(pricer, model, x, exdate, T, r, strikes)
            ax.plot(np.log(strikes / F), 100 * iv, **STYLE[model])

        ax.set_title(f"{pd.Timestamp(exdate):%d %b %Y}  (T = {T:.2f})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes.flat[len(exdates):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Log-moneyness log(K / F)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied vol (%)", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(filename + ".pdf")
    fig.savefig(filename + ".png", dpi=150)
    print(f"saved {filename}.pdf and {filename}.png")
    return fig


if __name__ == "__main__":
    data = pd.read_csv("cleaned_data.csv", parse_dates=["exdate"])
    rates = pd.read_csv("calibrated_parameters.csv", parse_dates=["maturity"])
    S = pd.read_excel("Data20191113.xlsx", header=None).iat[2, 6]
    pricer = FFTPricer(S=S,
                           q=float(rates["calibrated_q"].iloc[0]),
                           rates=dict(zip(rates["maturity"], rates["calibrated_r"])))
    fitter = Fitter(pricer, data)

    res = pd.read_csv("calibration_results.csv")
    sse = res[res["loss"] == "SSE"].set_index("model")
    params = {
        "VG":   tuple(sse.loc["VG", ["sigma", "nu", "theta"]]),
        "CGMY": tuple(sse.loc["CGMY", ["C", "G", "M", "Y"]]),
    }
    print("SSE parameters:", {m: np.round(p, 4) for m, p in params.items()})

    plot_smiles(fitter, params, "smiles_sse")
