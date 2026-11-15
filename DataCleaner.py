import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self):
        sheet = pd.read_excel("Data20191113.xlsx", header=None)

        self.spy_close = sheet.iat[2, 6]
        self.df = sheet.iloc[6:].copy()
        self.df.columns = sheet.iloc[5].tolist()

        # Clean column names
        self.df.columns = (
            self.df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Convert dates
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df["exdate"] = pd.to_datetime(self.df["exdate"], errors="coerce")

        self.df["time_to_maturity"] = (
            self.df["exdate"] - self.df["date"]
        ).dt.days / 365 # er i tvivl her
        print(f"Intial length of self.df is: {self.df.shape[0]}")

        # Loader calibrated parames for yield og discount rate
        self.yield_and_discount_factor = pd.read_csv("calibrated_parameters.csv")
        print(self.yield_and_discount_factor)

        self.yield_and_discount_factor["maturity"] = pd.to_datetime(
            self.yield_and_discount_factor["maturity"]
        )
        print(self.yield_and_discount_factor)

    def remove_0_bid_prices(self):
        self.df = self.df[self.df["best_bid"] != 0]
        print(f"Length of self.df after removing 0 prices bid options: {self.df.shape[0]}")

    def remove_itm_prices(self):
        # removing in-the-money Call options
        self.df = self.df[~((self.df["cp_flag"] == "C") & (self.df["strike"] < self.spy_close))]
        # removing in-the-money Put options
        self.df = self.df[~((self.df["cp_flag"] == "P") & (self.df["strike"] > self.spy_close))]
        print(f"Length of self.df after removing itm options: {self.df.shape[0]}")

    def keep_matched_call_put_pairs(self):

        keys = ["date", "exdate", "style", "strike"]

        prices = (
            self.df.groupby(keys + ["cp_flag"])["mid"]
            .mean()
            .unstack("cp_flag")
        )

        prices = prices.dropna(subset=["C", "P"])

        prices = prices.rename(columns={
            "C": "call_market",
            "P": "put_market"
        })

        return prices

    def convert_puts_to_calls(self, paired):

        rates = self.yield_and_discount_factor

        parameters = (
            self.df.groupby(["date", "exdate", "style", "strike"])
            .agg(
                time_to_maturity=("time_to_maturity", "first")
            )
        )

        paired = paired.join(parameters)

        paired = paired.reset_index().merge(
            rates,
            left_on="exdate",
            right_on="maturity",
            how="inner",
            validate="many_to_one"
        )

        K = paired["strike"]
        T = paired["time_to_maturity"]
        r = paired["calibrated_r"]
        q = paired["calibrated_q"]

        paired["synthetic_call"] = (
            paired["put_market"]
            + self.spy_close * np.exp(-q * T)
            - K * np.exp(-r * T)
        )

        return paired

    def average_call_prices(self, paired):

        paired["call_price"] = (
            paired["call_market"]
            + paired["synthetic_call"]
        ) / 2

        return paired



    def find_vertical_spread_arbitrage(
        self, paired, tolerance=1e-8
    ):

        flagged = set()

        for _, group in paired.groupby(
            ["date", "exdate", "style"]
        ):

            group = group.sort_values("strike")

            for i in range(len(group) - 1):

                row1 = group.iloc[i]
                row2 = group.iloc[i + 1]

                k1, k2 = row1["strike"], row2["strike"]
                p1, p2 = row1["call_price"], row2["call_price"]

                if k2 <= k1:
                    continue

                Q = (p1 - p2) / (k2 - k1)

                upper_bound = np.exp(
                    -row1["calibrated_r"]
                    * row1["time_to_maturity"]
                )

                if Q < -tolerance or Q > upper_bound + tolerance:

                    flagged.add(tuple(row1[[
                        "date", "exdate", "style", "strike"
                    ]]))

                    flagged.add(tuple(row2[[
                        "date", "exdate", "style", "strike"
                    ]]))

        return flagged

    def find_butterfly_spread_arbitrage(self, paired, tolerance=1e-8):

        flagged = set()

        for _, group in paired.groupby(
            ["date", "exdate", "style"]
        ):

            group = group.sort_values("strike")

            for i in range(len(group) - 2):

                rows = group.iloc[i:i+3]

                k1, k2, k3 = rows["strike"].to_numpy()
                p1, p2, p3 = rows["call_price"].to_numpy()

                if not (k1 < k2 < k3):
                    continue

                butterfly_price = (
                    p1
                    - ((k3-k1)/(k3-k2)) * p2
                    + ((k2-k1)/(k3-k2)) * p3
                )

                if butterfly_price < -tolerance:

                    for _, row in rows.iterrows():

                        flagged.add(tuple(row[[
                            "date", "exdate", "style", "strike"
                        ]]))

        return flagged

    def find_calendar_spread_arbitrage(
        self, paired, tolerance=1e-8
    ):

        flagged = set()

        for _, group in paired.groupby(
            ["date", "strike", "style"]
        ):

            group = group.sort_values("exdate")

            for i in range(len(group) - 1):

                row1 = group.iloc[i]
                row2 = group.iloc[i + 1]

                price_difference = (
                    row2["call_price"] - row1["call_price"]
                )

                if price_difference < -tolerance:

                    flagged.add(tuple(row1[[
                        "date", "exdate", "style", "strike"
                    ]]))

                    flagged.add(tuple(row2[[
                        "date", "exdate", "style", "strike"
                    ]]))

        return flagged

    def remove_static_arbitrage(self, paired, tolerance=1e-8):

        vertical = self.find_vertical_spread_arbitrage(
            paired, tolerance
        )

        butterfly = self.find_butterfly_spread_arbitrage(
            paired, tolerance
        )

        calendar = self.find_calendar_spread_arbitrage(
            paired, tolerance
        )

        flagged = vertical | butterfly | calendar

        # Remove violations from the temporary surface only
        paired = paired[
            ~paired.set_index(
                ["date", "exdate", "style", "strike"]
            ).index.isin(flagged)
        ].copy()

        print("\nStatic arbitrage results:")
        print(f"Vertical violations: {len(vertical)}")
        print(f"Butterfly violations: {len(butterfly)}")
        print(f"Calendar violations: {len(calendar)}")
        print(f"Total unique violations: {len(flagged)}")

        return paired

    def check_minimum_quotes(self, paired):

        surface_keys = ["date", "exdate", "style"]

        valid_surfaces = (
            paired.groupby(surface_keys)
            .size()
        )

        valid_surfaces = valid_surfaces[
            valid_surfaces > 10
        ].index

        paired = paired.set_index(surface_keys)

        paired = paired.loc[
            paired.index.isin(valid_surfaces)
        ].reset_index()

        print(
            f"Surfaces with more than 10 quotes: "
            f"{len(valid_surfaces)}"
        )

        return paired

    def find_lower_bound_arbitrage(self, paired, tolerance=1e-8):
            """
            Equation (29):
            C_t(S_t; T, K_j) > max(0, S_t * exp(-q*T) - K_j * exp(-r*T))
            """
            flagged = set()

            S = self.spy_close
            K = paired["strike"]
            T = paired["time_to_maturity"]
            r = paired["calibrated_r"]
            q = paired["calibrated_q"]
            C = paired["call_price"]

            intrinsic_bound = S * np.exp(-q * T) - K * np.exp(-r * T)
            lower_bound = np.maximum(0.0, intrinsic_bound)

            # Violation if C <= lower_bound (within numerical tolerance)
            violations = paired[C <= lower_bound + tolerance]

            for _, row in violations.iterrows():
                flagged.add(tuple(row[["date", "exdate", "style", "strike"]]))

            return flagged

    def remove_thomas_artical_arb(self, paired, tolerance=1e-8):
        """
        Enforces Equations (29), (30), and (31) from Thomas Kockum:
        - (29) Call lower bound
        - (30) Vertical spread monotonicity
        - (31) Butterfly spread convexity
        """
        # Equation (29): Lower bound
        eq29_violations = self.find_lower_bound_arbitrage(paired, tolerance)

        # Equation (30): Vertical spread condition
        eq30_violations = self.find_vertical_spread_arbitrage(paired, tolerance)

        # Equation (31): Butterfly / convexity condition
        eq31_violations = self.find_butterfly_spread_arbitrage(paired, tolerance)

        flagged = eq29_violations | eq30_violations | eq31_violations

        paired = paired[
            ~paired.set_index(["date", "exdate", "style", "strike"]).index.isin(flagged)
        ].copy()

        print("\nThomas article no-arbitrage results (Eq. 29-31):")
        print(f"Eq (29) Lower bound violations: {len(eq29_violations)}")
        print(f"Eq (30) Vertical violations: {len(eq30_violations)}")
        print(f"Eq (31) Convexity violations: {len(eq31_violations)}")
        print(f"Total unique violations removed: {len(flagged)}")

        return paired

    def remove_arbitrage_prices(self):
        # 5 step plan: Thomas Kockum, section 3

        # Step 1:
        # Only use strikes where both put and call prices are quoted.
        paired = self.keep_matched_call_put_pairs()

        # Step 2:
        # Convert put prices into synthetic call prices.
        paired = self.convert_puts_to_calls(paired)

        # Step 3:
        # Compute the average of the two available call prices.
        paired = self.average_call_prices(paired)

        # Step 4:
        # Remove static arbitrage using vertical, butterfly
        # and calendar spread conditions.
        paired = self.remove_static_arbitrage(paired)
        paired = self.remove_thomas_artical_arb(paired)
        
        # Step 5:
        # Only calibrate surfaces with more than 10 quotes.
        paired = self.check_minimum_quotes(paired)

        # Create a mask for the ORIGINAL DataFrame.
        # No original prices are modified.

        keys = ["date", "exdate", "style", "strike"]

        valid_keys = pd.MultiIndex.from_frame(
            paired[keys]
        )

        original_keys = pd.MultiIndex.from_frame(
            self.df[keys]
        )

        mask = original_keys.isin(valid_keys)

        self.df = self.df.loc[mask].copy()

        print("\nOriginal option data filtered.")
        print(f"Remaining original observations: {len(self.df)}")


    def clean_data_for_calibration(self):
        self.remove_0_bid_prices()
        self.remove_arbitrage_prices()
        self.remove_itm_prices()

        return self.df, self.spy_close


    def clean_data_for_put_call_parity(self, n: int = 5):
        self.df_sorted = self.df.assign(
            strike_diff=abs(self.df["strike"] - self.spy_close)
        ).sort_values(by=["exdate", "strike_diff"])

        # Select n unique nearest strikes per exdate
        nearest_strikes = (
            self.df_sorted.groupby("exdate")["strike"]
            .unique()
            .apply(lambda strikes: strikes[:n*2])
            .explode()
            .reset_index()
        )

        # Merge back to get both Call and Put records for those strikes
        pc_self.df = pd.merge(self.df, nearest_strikes, on=["exdate", "strike"])

        return pc_self.df, self.spy_close



if __name__ == "__main__":
    cleaner = DataCleaner()
    test, spy_close_test = cleaner.clean_data_for_calibration()


