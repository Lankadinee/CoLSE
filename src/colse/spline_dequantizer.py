from typing import Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


class SplineDequantizer:
    """
    Spline‐based dequantization (via PCHIP) on discrete/categorical columns of a DataFrame.
    Now handles unseen discrete values by interpolating between the two nearest seen levels.
    """

    def __init__(self, M: int = 10000):
        """
        Parameters
        ----------
        M : int
            Number of grid points in [0, K] used to invert the PCHIP spline.
            Larger M ⇒ finer inversion accuracy (default 10 000).
        """
        self.M = M
        self.dequantizers = {}  # per‐column params

    def _fit_single_column(self, x: pd.Series, col_name: str):
        """
        Build histogram, CDF, PCHIP spline, and lookup table for one column.
        Stores in self.dequantizers[col_name]:
            'uniques':  np.ndarray of length K (sorted distinct values seen at fit)
            'K':        int, number of seen levels
            'cdf_b':    np.ndarray of length (K+1): [0, CDF(0), CDF(1), …, CDF(K−1)]
            'grid_z':   np.ndarray of length M, points in [0, K]
            'grid_c':   np.ndarray of length M, spline_cdf(grid_z)
        """
        # 1) Factorize (integer‐encode) so codes ∈ {0,…,K−1}
        codes, uniques = pd.factorize(x, sort=True)
        if (codes < 0).any():
            raise ValueError(f"Column '{col_name}' contains NaN during fit.")

        K = len(uniques)            # number of distinct seen levels
        N = len(codes)

        # 2) Histogram + relative frequencies
        counts = np.bincount(codes, minlength=K).astype(np.float64)  # length K
        p = counts / float(N)                                        # length K

        # 3) Build CDF at integer boundaries
        CDF_vals = np.cumsum(p)                                      # length K
        cdf_b = np.concatenate(([0.0], CDF_vals))                    # length K+1

        # 4) Fit monotonic PCHIP spline on points (0..K) → cdf_b
        z_b = np.arange(K + 1, dtype=np.float64)                     # [0, …, K]
        spline_cdf = PchipInterpolator(z_b, cdf_b, extrapolate=False)

        # 5) Build lookup table (M points in [0, K]) for inversion
        grid_z = np.linspace(0.0, float(K), self.M, dtype=np.float64)  # shape (M,)
        grid_c = spline_cdf(grid_z)                                    # shape (M,)

        # 6) Store parameters
        #    Convert uniques to numpy array for searchsorted later
        uniques_arr = np.array(uniques)
        self.dequantizers[col_name] = {
            'uniques': uniques_arr,   # dtype: whatever x.dtype
            'K': K,
            'cdf_b': cdf_b,           # shape (K+1,)
            'grid_z': grid_z,         # shape (M,)
            'grid_c': grid_c          # shape (M,)
        }

    def fit(self, df: pd.DataFrame, columns=None):
        """
        Fit dequantizers for each specified column.
        """
        if columns is None:
            columns = df.columns.tolist()
        for col in columns:
            self._fit_single_column(df[col], col)

    def _transform_single_column(self, x: pd.Series, col_name: str) -> np.ndarray:
        """
        Dequantize one column into a continuous array in [0,1).
        Unseen values are interpolated between nearest seen neighbors.
        """
        params = self.dequantizers[col_name]
        uniques = params['uniques']         # length K, sorted
        K = params['K']
        cdf_b = params['cdf_b']             # length K+1
        grid_z = params['grid_z']           # length M
        grid_c = params['grid_c']           # length M

        # 1) Factorize x with same uniques → codes in [0..K-1] or -1 if unseen
        cat = pd.Categorical(x, categories=uniques, ordered=True)
        codes = cat.codes                   # shape (N,), -1 if unseen
        N = len(codes)

        # Prepare arrays for "low_cdf" and "high_cdf" (in CDF‐space) for each row
        low_cdf = np.empty(N, dtype=np.float64)
        high_cdf = np.empty(N, dtype=np.float64)

        # ── Handle seen codes (0..K-1)
        seen_mask = (codes >= 0)
        if seen_mask.any():
            sc = codes[seen_mask]
            low_cdf[seen_mask] = cdf_b[sc]       # CDF(i)
            high_cdf[seen_mask] = cdf_b[sc + 1]  # CDF(i+1)

        # ── Handle unseen values (codes == -1)
        unseen_mask = (codes < 0)
        if unseen_mask.any():
            # For each unseen x[i], we need to find insertion position in uniques
            unseen_vals = np.array(x[unseen_mask])
            # Search in the sorted uniques array
            pos = np.searchsorted(uniques, unseen_vals)  # array of shape (n_unseen,)

            # For each position pos[j], determine neighbors:
            #   if pos[j] == 0: no lower neighbor → use code_lo = 0, code_hi = 0
            #   elif pos[j] == K: no upper neighbor → use code_lo = K−1, code_hi = K−1
            #   else: code_lo = pos[j]−1, code_hi = pos[j]
            pos_lo = np.clip(pos - 1, 0, K - 1)
            pos_hi = np.clip(pos, 0, K - 1)

            # For each unseen row, fetch cdf_b at boundaries for those neighbor codes:
            cdf_lo_low = cdf_b[pos_lo]         # CDF at code_lo
            cdf_lo_high = cdf_b[pos_lo + 1]    # CDF at code_lo+1
            cdf_hi_low = cdf_b[pos_hi]         # CDF at code_hi
            cdf_hi_high = cdf_b[pos_hi + 1]    # CDF at code_hi+1

            # Interpolated low_cdf = (cLo_low + cHi_low) / 2
            # Interpolated high_cdf = (cLo_high + cHi_high) / 2
            interp_low = 0.5 * (cdf_lo_low + cdf_hi_low)
            interp_high = 0.5 * (cdf_lo_high + cdf_hi_high)

            # Assign into low_cdf, high_cdf at unseen_mask positions
            low_cdf[unseen_mask] = interp_low
            high_cdf[unseen_mask] = interp_high

        # 2) Draw uniform v in [low_cdf, high_cdf] for each row
        v = np.random.uniform(low_cdf, high_cdf)   # shape (N,)

        # 3) For seen rows, invert via np.interp; for unseen, treat same way
        #    Actually, now every row has a valid (low_cdf < high_cdf) unless low_cdf==high_cdf.
        #    Even if they are equal, np.interp will return that same z. So we can unify:
        z = np.interp(v, grid_c, grid_z)           # shape (N,)

        # 4) Normalize to [0,1): divide by (K) if we want z∈[0,K], but since z≤K, dividing by K 
        #    would make unseen at exactly 1.0. Instead, divide by (K - 1e-12) or handle clamping.
        #    But simpler: since z_max = K - ε (because cdf_b[K] = 1, and grid_z max = K),
        #    we can safely divide by K. z=K only if v=1 exactly (rare). To avoid exactly 1,
        #    we cap z by K - 1e-12. Or divide by (K + 1) as before.
        #
        #    We'll divide by (K) but then clamp any 1.0 down to (K-ε)/K. For practical purposes,
        #    we can do z = np.minimum(z, K - 1e-12) before normalization.
        z = np.minimum(z, float(K) - 1e-12)
        x_cont = z / float(K)

        return x_cont

    def transform(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        Dequantize each specified column in df, return a new DataFrame of continuous columns.
        """
        if columns is None:
            columns = list(self.dequantizers.keys())
        result = pd.DataFrame(index=df.index)
        for col in columns:
            result[col] = self._transform_single_column(df[col], col)
        return result

    def get_interval(self, col_name: str, original_value) -> Tuple[float, float]:
        """
        Given one old‐value (numeric or categorical), return its continuous interval [low, high)
        in [0,1) after dequantization. For unseen numeric values, interpolate between the two nearest
        seen neighbors. For unseen categorical values (non‐numeric), treat as no numeric order: return
        an empty interval (low=high=NaN).
        """
        if col_name not in self.dequantizers:
            raise KeyError(f"Column '{col_name}' was not fit.")

        params = self.dequantizers[col_name]
        uniques = params['uniques']    # sorted array of seen values
        K = params['K']
        cdf_b = params['cdf_b']        # length K+1

        # 1) Check if original_value was seen
        #    Use pandas.Categorical to find its code if present, else code = -1
        cat = pd.Categorical([original_value], categories=uniques, ordered=True)
        code = int(cat.codes[0])  # -1 if unseen

        if code >= 0:
            # Seen case: interval in CDF‐space is [cdf_b[code], cdf_b[code+1]]
            low_cdf = cdf_b[code]
            high_cdf = cdf_b[code + 1]
        else:
            # Unseen: if original_value is numeric, attempt to interpolate
            try:
                v = float(original_value)
            except Exception:
                # Non-numeric unseen (categorical): no meaningful interpolation → return NaN interval
                return (np.nan, np.nan)

            # Locate insertion position in sorted uniques (numeric)
            pos = np.searchsorted(uniques.astype(np.float64), v)
            if pos <= 0:
                # v < uniques[0]: clamp to the first seen level
                low_cdf = cdf_b[0]
                high_cdf = cdf_b[1]
            elif pos >= K:
                # v > uniques[-1]: clamp to the last seen level
                low_cdf = cdf_b[K - 1]
                high_cdf = cdf_b[K]
            else:
                # Interpolate between neighbors at codes pos-1 and pos
                i_lo = pos - 1
                i_hi = pos
                low_cdf_lo = cdf_b[i_lo]
                low_cdf_hi = cdf_b[i_hi]
                high_cdf_lo = cdf_b[i_lo + 1]
                high_cdf_hi = cdf_b[i_hi + 1]
                # Midpoints:
                low_cdf = 0.5 * (low_cdf_lo + low_cdf_hi)
                high_cdf = 0.5 * (high_cdf_lo + high_cdf_hi)

        # 2) Convert from CDF‐space back to z‐space by inverting the spline:
        #    We use the stored lookup table (grid_c, grid_z) for fast inversion.
        grid_z = params['grid_z']
        grid_c = params['grid_c']

        # If low_cdf == high_cdf, np.interp returns the same z for both
        z_low = np.interp(low_cdf, grid_c, grid_z)
        z_high = np.interp(high_cdf, grid_c, grid_z)

        # 3) Normalize to [0,1)
        #    Clamp at K - tiny to avoid exactly 1.0
        z_low = min(z_low, float(K) - 1e-12)
        z_high = min(z_high, float(K) - 1e-12)
        low = z_low / float(K)
        high = z_high / float(K)

        return (low, high)

    def get_discrete_intervals(self, col_name: str, value_range: Tuple[float, float]) -> Tuple[float, float]:
        """
        Given iterable of old‐values, return list of (low, high) intervals for each.
        """
        if col_name not in self.dequantizers:
            raise KeyError(f"Column '{col_name}' was not fit.")
        
        range_1 = self.get_interval(col_name, value_range[0])
        range_2 = self.get_interval(col_name, value_range[1])
        return (range_1[0], range_2[1])
