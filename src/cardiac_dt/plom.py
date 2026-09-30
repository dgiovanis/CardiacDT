"""PLoM numerical routines bundled from the original research library.

Arrays in this module generally use (features, samples), unlike the public
workflow, which uses (samples, features). Numerical formulas are preserved.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.beta = None
        self.alpha = None
        self.alpha_inv = None

    def fit(self, X):
        Rmax = np.max(X, axis=1)
        Rmin = np.min(X, axis=1)

        nbx = X.shape[0]

        self.beta = np.zeros(nbx)
        self.alpha = np.zeros(nbx)
        self.alpha_inv = np.zeros(nbx)

        for k in range(nbx):
            if Rmax[k] - Rmin[k] != 0:
                self.beta[k] = Rmin[k]
                self.alpha[k] = Rmax[k] - Rmin[k]
                self.alpha_inv[k] = 1.0 / self.alpha[k]
            else:
                self.beta[k] = Rmin[k]
                self.alpha[k] = 1.0
                self.alpha_inv[k] = 1.0

        return self

    def transform(self, X):
        return self.alpha_inv[:, np.newaxis] * (X - self.beta[:, np.newaxis])

    def inverse_transform(self, Xscaled):
        return self.alpha[:, np.newaxis] * Xscaled + self.beta[:, np.newaxis]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PCA_PLom:
    def __init__(self, error_PCA=1e-5, verbose=True):
        self.error_PCA = error_PCA
        self.verbose = verbose
        self.mean = None
        self.eigenvalues = None  # RmuPCA
        self.eigenvectors = None  # MatRVectPCA
        self.nu = None
        self.nnull = None
        self.trace_cov = None
        self.error = None

    # ------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------
    def fit(self, X):
        """
        X: shape (n_x, n_d)
        """
        n_x, n_d = X.shape

        RXmean = np.mean(X, axis=1)
        self.mean = RXmean
        MatRXmean = RXmean[:, None]

        # trace of covariance
        MatRtemp = (X - MatRXmean) ** 2
        Rtemp = np.sum(MatRtemp, axis=1) / (n_d - 1)
        traceMatRXcov = np.sum(Rtemp)
        self.trace_cov = traceMatRXcov

        # ============================================================
        # CASE 1: n_x <= n_d
        # ============================================================
        if n_x <= n_d:
            MatRXcov = np.cov(X)
            MatRXcov = 0.5 * (MatRXcov + MatRXcov.T)

            RmuTemp, MatRVectTemp = np.linalg.eigh(MatRXcov)

            # sign consistency
            for i in range(MatRVectTemp.shape[1]):
                if MatRVectTemp[0, i] < 0:
                    MatRVectTemp[:, i] *= -1

            idx = np.argsort(RmuTemp)[::-1]
            RmuPCA = RmuTemp[idx]
            RmuPCA[RmuPCA < 0] = 0
            MatRVectPCA = MatRVectTemp[:, idx]

            RerrPCA = 1 - np.cumsum(RmuPCA) / traceMatRXcov
            RerrPCA[RerrPCA < 0] = RerrPCA[0] * 1e-14

            Ind = np.where(RerrPCA <= self.error_PCA)[0]

            if Ind.size > 0:
                nu = n_x - Ind.size + 1
                if nu + 1 <= n_x:
                    RmuPCA = RmuPCA[:nu]
                    MatRVectPCA = MatRVectPCA[:, :nu]
            else:
                nu = n_x

            nnull = n_x - nu

        # ============================================================
        # CASE 2: n_x > n_d
        # ============================================================
        else:
            MatRVectTemp, RSigmaTemp, _ = np.linalg.svd(
                X - MatRXmean, full_matrices=False
            )

            for i in range(MatRVectTemp.shape[1]):
                if MatRVectTemp[0, i] < 0:
                    MatRVectTemp[:, i] *= -1

            idx = np.argsort(RSigmaTemp)[::-1]
            RSigma = RSigmaTemp[idx]

            RmuPCA = (RSigma**2) / (n_d - 1)
            MatRVectPCA = MatRVectTemp[:, idx]

            RerrPCA = 1 - np.cumsum(RmuPCA) / traceMatRXcov
            RerrPCA[RerrPCA < 0] = RerrPCA[0] * 1e-14

            Ind = np.where(RerrPCA <= self.error_PCA)[0]

            if Ind.size > 0:
                nu = n_d - Ind.size + 1
                if nu + 1 <= n_d:
                    RmuPCA = RmuPCA[:nu]
                    MatRVectPCA = MatRVectPCA[:, :nu]
            else:
                nu = n_d

            nnull = n_d - nu

        self.eigenvalues = RmuPCA
        self.eigenvectors = MatRVectPCA
        self.nu = nu
        self.nnull = nnull
        self.error = RerrPCA

        return self

    # ------------------------------------------------------------
    # STANDARD PCA (no whitening)
    # ------------------------------------------------------------
    def transform(self, X):
        """
        Standard PCA projection (NOT PLoM)
        """
        Xc = X - self.mean[:, None]
        return self.eigenvectors.T @ Xc

    def inverse_transform(self, Xpca):
        """
        Standard PCA inverse
        """
        return self.eigenvectors @ Xpca + self.mean[:, None]

    # ------------------------------------------------------------
    # PLoM COORDINATES (WHITENED)
    # ------------------------------------------------------------
    def transform_plom(self, X):
        """
        Compute MatReta (PLoM coordinates)

        Output: (nu, n_samples)
        """
        Xc = X - self.mean[:, None]
        Rcoef = 1.0 / np.sqrt(self.eigenvalues)
        MatRcoef = np.diag(Rcoef)

        return MatRcoef @ self.eigenvectors.T @ Xc

    def inverse_transform_plom(self, MatReta):
        """
        Reconstruct from PLoM coordinates

        Input: (nu, n_samples)
        Output: (n_x, n_samples)
        """
        return (
            self.mean[:, None]
            + self.eigenvectors @ np.diag(np.sqrt(self.eigenvalues)) @ MatReta
        )

    # ------------------------------------------------------------
    # ERROR CHECK
    # ------------------------------------------------------------
    def reconstruction_error(self, X):
        """
        Compute reconstruction error in PLoM coordinates
        """
        MatReta = self.transform_plom(X)
        Xrec = self.inverse_transform_plom(MatReta)

        error = np.linalg.norm(X - Xrec, "fro") / np.linalg.norm(X, "fro")

        if self.verbose:
            print(f"error_PCA target       = {self.error_PCA:14.7e}")
            print(f"n_x                   = {X.shape[0]:4d}")
            print(f"n_d                   = {X.shape[1]:4d}")
            print(f"nu                    = {self.nu:4d}")
            print(f"L2 reconstruction err = {error:14.7e}")

        return error


class DiffusionMapsBasis:
    def __init__(
        self,
        nbmDMAP=10,
        mDP=20,
        comp_ref=1.0,
        n_search=30,
        log10_span=3.0,
        verbose=True,
    ):
        """
        Parameters
        ----------
        nbmDMAP : int
            Number of diffusion basis vectors to retain
        mDP : int
            Number of eigenpairs kept before truncation to nbmDMAP
        comp_ref : float
            Reference threshold for spectral ratio lambda_{k+1}/lambda_k
        n_search : int
            Number of epsilon values in the log-scale search
        log10_span : float
            Search over median_epsilon * 10^s, with s in [-log10_span, +log10_span]
        """
        self.nbmDMAP = nbmDMAP
        self.mDP = mDP
        self.comp_ref = comp_ref
        self.n_search = n_search
        self.log10_span = log10_span
        self.verbose = verbose

        # Outputs
        self.epsilonDIFF = None
        self.iter_conv = None
        self.epsilon_grid = None
        self.epsilon_median = None
        self.Rcomp = None
        self.Rlambda = None
        self.MatRphi = None
        self.MatRg = None
        self.MatRa = None
        self.Rd = None
        self.Rdm1s2 = None
        self.distance_sq = None

    def _compute_pairwise_sqdist(self, MatReta_d):
        """
        MatReta_d : (nu, n_d)
        returns D : (n_d, n_d), where D[i,j] = ||eta_i - eta_j||^2
        """
        diff = MatReta_d[:, :, None] - MatReta_d[:, None, :]
        D = np.sum(diff**2, axis=0)
        return D

    def _build_kernel_and_spectrum(self, D, epsilon):
        n_d = D.shape[0]
        co = 1.0 / (4.0 * epsilon)
        K = np.exp(-co * D)
        scaling = np.max(K)
        K = K + scaling * 1e-12 * np.eye(n_d)

        Rd = np.sum(K, axis=1)
        Rdm1s2 = 1.0 / np.sqrt(Rd)

        Ps = (Rdm1s2[:, None] * K) * Rdm1s2[None, :]
        Ps = 0.5 * (Ps + Ps.T)

        Rlambda, MatRphi = np.linalg.eigh(Ps)

        idx = np.argsort(Rlambda)[::-1]
        Rlambda = Rlambda[idx]
        MatRphi = MatRphi[:, idx]

        return K, Rd, Rdm1s2, Rlambda, MatRphi

    def fit(self, MatReta_d):
        """
        Compute diffusion maps basis from latent data MatReta_d (nu, n_d)
        """
        nu, n_d = MatReta_d.shape

        if self.nbmDMAP >= n_d:
            raise ValueError("nbmDMAP must be strictly smaller than n_d")
        if self.mDP > n_d:
            raise ValueError("mDP must be <= n_d")
        if self.nbmDMAP > self.mDP:
            raise ValueError("nbmDMAP must be <= mDP")

        # ------------------------------------------------------------
        # 1. Pairwise squared distances
        # ------------------------------------------------------------
        D = self._compute_pairwise_sqdist(MatReta_d)
        self.distance_sq = D

        # use only off-diagonal distances for the median
        mask_offdiag = ~np.eye(n_d, dtype=bool)
        D_offdiag = D[mask_offdiag]

        positive_D = D_offdiag[D_offdiag > 0]
        if positive_D.size == 0:
            raise ValueError(
                "All pairwise distances are zero; diffusion maps cannot be built."
            )

        epsilon_median = np.median(positive_D)
        self.epsilon_median = epsilon_median

        # ------------------------------------------------------------
        # 2. Log-scale epsilon grid around the median
        # ------------------------------------------------------------
        log_shifts = np.linspace(-self.log10_span, self.log10_span, self.n_search)
        epsilon_grid = epsilon_median * (10.0**log_shifts)

        self.epsilon_grid = epsilon_grid
        Rcomp = np.zeros(self.n_search)

        found = False

        # ------------------------------------------------------------
        # 3. Search over epsilon
        # ------------------------------------------------------------
        for it, epsilon in enumerate(epsilon_grid):
            _, Rd, Rdm1s2, Rlambda, MatRphi = self._build_kernel_and_spectrum(
                D, epsilon
            )

            comp_iter = Rlambda[self.nbmDMAP] / Rlambda[self.nbmDMAP - 1]
            Rcomp[it] = comp_iter

            if self.verbose:
                print(
                    f"iter={it:2d}, epsilon={epsilon:12.5e}, "
                    f"lambda[k+1]/lambda[k]={comp_iter:12.5e}"
                )

            if comp_iter <= self.comp_ref:
                self.epsilonDIFF = epsilon
                self.iter_conv = it
                self.Rd = Rd
                self.Rdm1s2 = Rdm1s2
                found = True
                break

            k = self.nbmDMAP - 1

            lam_k = Rlambda[k]
            lam_k1 = Rlambda[k + 1]

            if self.verbose:
                print(
                    f"iter={it:2d}, eps={epsilon:12.5e} | "
                    f"λ[{k}]={lam_k:12.5e}, λ[{k + 1}]={lam_k1:12.5e}, "
                    f"ratio={lam_k1 / lam_k:12.5e}"
                )
        self.Rcomp = Rcomp

        if not found:
            raise ValueError(
                "STOP: log-scale search finished without finding epsilon satisfying the criterion."
            )

        _, Rd, Rdm1s2, Rlambda, MatRphi = self._build_kernel_and_spectrum(
            D, self.epsilonDIFF
        )

        self.Rd = Rd
        self.Rdm1s2 = Rdm1s2
        self.Rlambda = Rlambda[: self.mDP]
        self.MatRphi = MatRphi[:, : self.mDP]

        MatRg_mDP = self.MatRphi * Rdm1s2[:, None]
        self.MatRg = MatRg_mDP[:, : self.nbmDMAP]

        # a = g (g^T g)^{-1}
        self.MatRa = np.linalg.solve(self.MatRg.T @ self.MatRg, self.MatRg.T).T

        if self.verbose:
            print("\nSelected diffusion-maps parameters:")
            print(f"median squared distance = {self.epsilon_median:14.7e}")
            print(f"epsilonDIFF            = {self.epsilonDIFF:14.7e}")
            print(f"iter_conv              = {self.iter_conv:4d}")

        return self


def _solve_single_worker(args):
    """
    Worker for one Monte Carlo trajectory.
    Must live at top level for multiprocessing pickling.
    """
    (
        ell,
        nu,
        n_d,
        MatReta_d,
        nbMC,
        M0transient,
        f0,
        coeffDeltar,
        mode,
        MatRg,
        MatRa,
        verbose,
        ArrayGauss_ell,
        ArrayWiennerM0transient_ell,
    ) = args

    solver = ISDESolver(
        nu=nu,
        n_d=n_d,
        MatReta_d=MatReta_d,
        nbMC=nbMC,
        M0transient=M0transient,
        f0=f0,
        coeffDeltar=coeffDeltar,
        mode=mode,
        MatRg=MatRg,
        MatRa=MatRa,
        random_state=None,
        verbose=False if verbose is None else verbose,
    )

    Y = solver.solve_single(
        ArrayWiennerM0transient_ell=ArrayWiennerM0transient_ell,
        MatRGauss_ell=ArrayGauss_ell,
    )
    return ell, Y


class ISDESolver:
    def __init__(
        self,
        nu,
        n_d,
        MatReta_d,
        nbMC=20,
        M0transient=50,
        f0=4.0,
        coeffDeltar=1.0,
        mode="reduced",
        MatRg=None,
        MatRa=None,
        random_state=None,
        verbose=True,
    ):
        """
        ISDE solver for PLoM with two modes:
        - mode='reduced': solve reduced-order ISDE in Z-space
        - mode='full'   : solve full-order ISDE directly in H-space
        """
        self.nu = nu
        self.n_d = n_d
        self.MatReta_d = MatReta_d

        self.nbMC = nbMC
        self.M0transient = M0transient
        self.f0 = f0
        self.coeffDeltar = coeffDeltar
        self.mode = mode.lower()
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

        if self.mode not in ["reduced", "full"]:
            raise ValueError("mode must be either 'reduced' or 'full'")

        self.MatRg = MatRg
        self.MatRa = MatRa

        if self.mode == "reduced":
            if MatRg is None or MatRa is None:
                raise ValueError("MatRg and MatRa are required for mode='reduced'")
            self.nbasis = MatRg.shape[1]
        else:
            self.nbasis = n_d

        self.s = (4.0 / ((nu + 2) * n_d)) ** (1.0 / (nu + 4))
        # self.s *= 0.4
        self.s2 = self.s**2
        self.shss = 1.0 / np.sqrt(self.s2 + (n_d - 1) / n_d)
        self.sh = self.s * self.shss

        self.Deltar = 2.0 * np.pi * self.sh / self.coeffDeltar
        self.M0estim = (
            2.0 * np.log(100.0) * self.coeffDeltar / (np.pi * self.f0 * self.sh)
        )

        self.ArrayGauss = None
        self.ArrayWiennerM0transient = None

        self.ArrayY_ar = None  # generic state trajectories: Z if reduced, H if full
        self.ArrayH_ar = None  # reconstructed / generated H
        self.MatReta_ar = None  # reshaped output

    # ------------------------------------------------------------------
    # Random inputs
    # ------------------------------------------------------------------
    def generate_random_inputs(self):
        self.ArrayGauss = np.random.randn(self.nu, self.n_d, self.nbMC)
        self.ArrayWiennerM0transient = np.random.randn(
            self.nu, self.n_d, self.M0transient, self.nbMC
        )
        return self

    # ------------------------------------------------------------------
    # Drift in full-order coordinates
    # ------------------------------------------------------------------
    def _Lrond_full(self, MatRH):
        sh2 = self.sh * self.sh
        sh2m1 = 1.0 / sh2
        co1 = 1.0 / (2.0 * sh2)
        co2 = self.shss / (sh2 * self.n_d)
        X = self.shss * self.MatReta_d
        Y = MatRH
        X2 = np.sum(X * X, axis=0)[:, None]
        Y2 = np.sum(Y * Y, axis=0)[None, :]
        G = X.T @ Y

        dist2 = X2 + Y2 - 2.0 * G
        dist2 = np.maximum(dist2, 0.0)
        Rexpo = co1 * dist2
        expo_min = np.min(Rexpo, axis=0, keepdims=True)
        RS = np.exp(-(Rexpo - expo_min))

        q = np.sum(RS, axis=0) / self.n_d

        # Rgraduqp[:, ell] = MatRetaco2 @ RS[:, ell]
        MatRetaco2 = co2 * self.MatReta_d
        Rgraduqp = MatRetaco2 @ RS
        MatRL = -sh2m1 * Y + Rgraduqp / q[None, :]

        return MatRL

    # ------------------------------------------------------------------
    # Drift in reduced-order coordinates
    # ------------------------------------------------------------------
    def _Lrond_reduced(self, MatRZ):
        """
        Reduced drift: MatRZ shape = (nu, nbasis)
        """
        MatRU = MatRZ @ self.MatRg.T
        MatRL = self._Lrond_full(MatRU)
        MatRLrond = MatRL @ self.MatRa
        return MatRLrond

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    def _initial_state_and_velocity(self, MatRGauss_ell):
        """
        Returns initial state and velocity depending on mode.
        """
        if self.mode == "reduced":
            MatY0 = self.MatReta_d @ self.MatRa
            MatV0 = MatRGauss_ell @ self.MatRa
        else:
            MatY0 = self.MatReta_d.copy()
            MatV0 = MatRGauss_ell.copy()
        return MatY0, MatV0

    # ------------------------------------------------------------------
    # Noise term
    # ------------------------------------------------------------------
    def _noise_term(self, ArrayWiennerM0transient_ell, k):
        """
        Returns noise term in the appropriate coordinates.
        """
        if self.mode == "reduced":
            return ArrayWiennerM0transient_ell[:, :, k] @ self.MatRa
        else:
            return ArrayWiennerM0transient_ell[:, :, k]

    # ------------------------------------------------------------------
    # Drift dispatcher
    # ------------------------------------------------------------------
    def _drift(self, MatY):
        if self.mode == "reduced":
            return self._Lrond_reduced(MatY)
        return self._Lrond_full(MatY)

    # ------------------------------------------------------------------
    # Single Verlet solve
    # ------------------------------------------------------------------
    def solve_single(self, ArrayWiennerM0transient_ell, MatRGauss_ell):
        """
        Solve one trajectory.

        Output shape:
        - reduced mode: (nu, nbasis)
        - full mode   : (nu, n_d)
        """
        b = self.f0 * self.Deltar / 4.0
        a0 = self.Deltar / 2.0
        a1 = (1.0 - b) / (1.0 + b)
        a2 = self.Deltar / (1.0 + b)
        a3 = np.sqrt(self.f0 * self.Deltar) / (1.0 + b)

        MatY, MatV = self._initial_state_and_velocity(MatRGauss_ell)

        for k in range(self.M0transient):
            MatY_half = MatY + a0 * MatV
            MatL_half = self._drift(MatY_half)
            MatW = self._noise_term(ArrayWiennerM0transient_ell, k)

            MatVp1 = a1 * MatV + a2 * MatL_half + a3 * MatW
            MatYp1 = MatY_half + a0 * MatVp1

            MatY = MatYp1
            MatV = MatVp1

        return MatY

    # ------------------------------------------------------------------
    # Batch solve
    # ------------------------------------------------------------------
    def solve_batch(self, n_jobs=1):
        """
        Solve all nbMC trajectories.

        Parameters
        ----------
        n_jobs : int
            Number of processes.
            - n_jobs=1  -> serial
            - n_jobs=-1 -> use all available CPUs
            - n_jobs>1  -> multiprocessing
        """
        if self.ArrayGauss is None or self.ArrayWiennerM0transient is None:
            self.generate_random_inputs()

        self.ArrayY_ar = np.zeros((self.nu, self.nbasis, self.nbMC))

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        if n_jobs is None or n_jobs < 1:
            n_jobs = 1

        # Serial fallback
        if n_jobs == 1:
            for ell in range(self.nbMC):
                MatRGauss_ell = self.ArrayGauss[:, :, ell]
                ArrayWiennerM0transient_ell = self.ArrayWiennerM0transient[:, :, :, ell]

                self.ArrayY_ar[:, :, ell] = self.solve_single(
                    ArrayWiennerM0transient_ell,
                    MatRGauss_ell,
                )
            return self

        # Parallel execution across Monte Carlo trajectories
        args_list = []
        for ell in range(self.nbMC):
            args_list.append(
                (
                    ell,
                    self.nu,
                    self.n_d,
                    self.MatReta_d,
                    self.nbMC,
                    self.M0transient,
                    self.f0,
                    self.coeffDeltar,
                    self.mode,
                    self.MatRg,
                    self.MatRa,
                    False,
                    self.ArrayGauss[:, :, ell],
                    self.ArrayWiennerM0transient[:, :, :, ell],
                )
            )

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for ell, Y in executor.map(_solve_single_worker, args_list):
                self.ArrayY_ar[:, :, ell] = Y

        return self

    # ------------------------------------------------------------------
    # Reconstruct H
    # ------------------------------------------------------------------
    def reconstruct_H(self):
        if self.ArrayY_ar is None:
            raise ValueError("Run solve_batch() first.")

        self.ArrayH_ar = np.zeros((self.nu, self.n_d, self.nbMC))

        if self.mode == "reduced":
            for ell in range(self.nbMC):
                self.ArrayH_ar[:, :, ell] = self.ArrayY_ar[:, :, ell] @ self.MatRg.T
        else:
            self.ArrayH_ar = self.ArrayY_ar.copy()

        return self

    # ------------------------------------------------------------------
    # Reshape output
    # ------------------------------------------------------------------
    def reshape_output(self):
        if self.ArrayH_ar is None:
            raise ValueError("Run reconstruct_H() first.")

        n_ar = self.n_d * self.nbMC
        self.MatReta_ar = np.reshape(self.ArrayH_ar, (self.nu, n_ar))
        return self

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def solve(self, n_jobs=1):
        self.generate_random_inputs()
        self.solve_batch(n_jobs=n_jobs)
        self.reconstruct_H()
        self.reshape_output()

        if self.verbose:
            print(f"ISDE mode        : {self.mode}")
            print(f"nu               : {self.nu}")
            print(f"n_d              : {self.n_d}")
            print(f"nbMC             : {self.nbMC}")
            print(f"M0transient      : {self.M0transient}")
            print(f"Deltar           : {self.Deltar:14.7e}")
            print(f"M0estim          : {self.M0estim:14.7e}")
            print(f"n_jobs           : {n_jobs}")
            print(f"Output shape     : {self.MatReta_ar.shape}")

        return self.MatReta_ar, self.ArrayY_ar


class ConditionalSecondOrderMoment:
    """
    Estimate conditional mean and conditional second-order moment of QQ given WW = ww0
    using a Gaussian kernel estimator.
    """

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

        self.mq = None
        self.mw = None
        self.N = None

        self.mean_qq = None
        self.std_qq = None
        self.mean_ww = None
        self.std_ww = None

        self.qq_tilde = None
        self.ww_tilde = None
        self.sx = None
        self.cox = None

        self.is_fitted = False

    def fit(self, MatRqq, MatRww):
        """
        Fit the estimator from training samples.

        Parameters
        ----------
        MatRqq : ndarray of shape (mq, N)
            Samples of the quantity of interest QQ.
        MatRww : ndarray of shape (mw, N)
            Samples of the control variable WW.
        """
        MatRqq = np.asarray(MatRqq, dtype=float)
        MatRww = np.asarray(MatRww, dtype=float)

        if MatRqq.ndim != 2:
            raise ValueError("MatRqq must have shape (mq, N)")
        if MatRww.ndim != 2:
            raise ValueError("MatRww must have shape (mw, N)")
        if MatRqq.shape[1] != MatRww.shape[1]:
            raise ValueError("MatRqq and MatRww must have the same number of samples N")

        self.mq, self.N = MatRqq.shape
        self.mw, Nw = MatRww.shape

        if self.N < 2:
            raise ValueError("At least two samples are required")

        nx = self.mq + self.mw

        if self.bandwidth is None:
            self.sx = (4.0 / (self.N * (2.0 + nx))) ** (1.0 / (nx + 4.0))
        else:
            self.sx = float(self.bandwidth)
            if self.sx <= 0:
                raise ValueError("bandwidth must be positive")

        self.cox = 1.0 / (2.0 * self.sx * self.sx)

        self.mean_qq = np.mean(MatRqq, axis=1)
        self.std_qq = np.std(MatRqq, axis=1, ddof=1)
        self.mean_ww = np.mean(MatRww, axis=1)
        self.std_ww = np.std(MatRww, axis=1, ddof=1)

        self.std_qq[self.std_qq == 0] = 1.0
        self.std_ww[self.std_ww == 0] = 1.0

        self.qq_tilde = (MatRqq - self.mean_qq[:, None]) / self.std_qq[:, None]
        self.ww_tilde = (MatRww - self.mean_ww[:, None]) / self.std_ww[:, None]

        self.is_fitted = True
        return self

    def _compute_weights(self, Rww0):
        """
        Compute Gaussian kernel weights for conditioning on WW = ww0.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        Rww0 = np.asarray(Rww0, dtype=float)

        if Rww0.shape != (self.mw,):
            raise ValueError(f"Rww0 must have shape ({self.mw},)")

        ww0_tilde = (Rww0 - self.mean_ww) / self.std_ww
        diff = self.ww_tilde - ww0_tilde[:, None]
        weights = np.exp(-self.cox * np.sum(diff**2, axis=0))
        return weights

    def predict(self, Rww0):
        """
        Compute conditional mean and conditional second-order moment at WW = ww0.

        Parameters
        ----------
        Rww0 : ndarray of shape (mw,)
            Conditioning value.

        Returns
        -------
        REqq_ww0 : ndarray of shape (mq,)
            Conditional mean E[QQ | WW = ww0].
        REqq2_ww0 : ndarray of shape (mq,)
            Conditional second-order moment E[QQ^2 | WW = ww0].
        """
        weights = self._compute_weights(Rww0)
        den = np.sum(weights)

        if den <= 1e-15:
            raise ValueError(
                "Sum of kernel weights is too small. "
                "The conditioning point may be too far from the data or the bandwidth too small."
            )

        num = self.qq_tilde @ weights
        num2 = (self.qq_tilde**2) @ weights

        temp = self.std_qq * num / den
        REqq_ww0 = self.mean_qq + temp
        REqq2_ww0 = (
            self.mean_qq**2
            + 2.0 * self.mean_qq * temp
            + (self.std_qq**2) * (self.sx**2 + num2 / den)
        )

        return REqq_ww0, REqq2_ww0

    def predict_variance(self, Rww0):
        """
        Compute conditional variance Var(QQ | WW = ww0).

        Parameters
        ----------
        Rww0 : ndarray of shape (mw,)
            Conditioning value.

        Returns
        -------
        var_qq_ww0 : ndarray of shape (mq,)
            Conditional variance.
        """
        mean, second_moment = self.predict(Rww0)
        var = second_moment - mean**2
        return np.maximum(var, 0.0)


class ConditionalSampler:
    """
    Kernel-based conditional sampling class.

    Supports:
      - conditional weights p_i(w0)
      - conditional expectation E[R | W=w0]
      - conditional expectation E[Q | W=w0]
      - conditional covariance Cov(Q | W=w0)
      - resampling-based conditional samples from p(R | W=w0)
      - resampling-based conditional samples from p(Q | W=w0)
      - local Gaussian approximation for Q | W=w0
      - construction of X = [Q; W] samples

    Conventions
    -----------
    W : array of shape (N, d_w)
        Conditioning/input variables stored row-wise.

    R : array of shape (N, d_r)
        Output/response variables stored row-wise.

    Q : array of shape (d_q, N), optional
        Alternative output representation stored column-wise.
        If not given but R is provided, we set Q = R.T.

    Notes
    -----
    - If you only care about R|W, you can just fit(W, R).
    - If you want the Q-based methods too, they will work automatically
      with Q = R.T unless you explicitly pass Q.
    """

    def __init__(self, W=None, R=None, Q=None, jitter_std=0.0, random_state=None):
        self.W = None
        self.R = None
        self.Q = None

        self.n_samples = None
        self.d_w = None
        self.d_r = None
        self.n_q = None
        self.n_x = None

        self.q_idx = None
        self.w_idx = None

        self.jitter_std = float(jitter_std)
        self.rng = np.random.default_rng(random_state)

        if W is not None:
            self.fit(W=W, R=R, Q=Q)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, W, R=None, Q=None):
        """
        Fit the conditional sampler.

        Parameters
        ----------
        W : ndarray, shape (N, d_w)
            Conditioning variables.
        R : ndarray, shape (N, d_r), optional
            Responses stored row-wise.
        Q : ndarray, shape (d_q, N), optional
            Responses stored column-wise.

        Returns
        -------
        self
        """
        W = np.asarray(W, dtype=float)
        if W.ndim != 2:
            raise ValueError("W must have shape (N, d_w).")

        self.W = W
        self.n_samples, self.d_w = W.shape

        if R is not None:
            R = np.asarray(R, dtype=float)
            if R.ndim != 2:
                raise ValueError("R must have shape (N, d_r).")
            if R.shape[0] != self.n_samples:
                raise ValueError("R must have the same number of rows as W.")
            self.R = R
            self.d_r = R.shape[1]
        else:
            self.R = None
            self.d_r = None

        if Q is not None:
            Q = np.asarray(Q, dtype=float)
            if Q.ndim != 2:
                raise ValueError("Q must have shape (d_q, N).")
            if Q.shape[1] != self.n_samples:
                raise ValueError("Q must have the same number of samples as W.")
            self.Q = Q
            self.n_q = Q.shape[0]
        elif self.R is not None:
            self.Q = self.R.T
            self.n_q = self.Q.shape[0]
        else:
            self.Q = None
            self.n_q = None

        if self.Q is not None:
            self.q_idx = np.arange(self.n_q)
            self.w_idx = np.arange(self.n_q, self.n_q + self.d_w)
            self.n_x = self.n_q + self.d_w
        else:
            self.q_idx = None
            self.w_idx = None
            self.n_x = None

        return self

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------
    def silverman_bandwidth(self, W=None):
        """
        Silverman-type scalar bandwidth based on W.
        """
        if W is None:
            if self.W is None:
                raise ValueError("No W available. Fit the class first or pass W.")
            W = self.W

        W = np.asarray(W, dtype=float)
        N, d = W.shape
        std = np.std(W, axis=0, ddof=1)
        std_mean = np.mean(std)
        return std_mean * (4.0 / (N * (2.0 + d))) ** (1.0 / (d + 4.0))

    # ------------------------------------------------------------------
    # Kernel weights
    # ------------------------------------------------------------------
    def kernel_weights(self, w0, bandwidth):
        """
        Compute normalized Gaussian kernel weights for conditioning point w0.

        Parameters
        ----------
        w0 : ndarray, shape (d_w,)
        bandwidth : float or ndarray of shape (d_w,)

        Returns
        -------
        weights : ndarray, shape (N,)
        """
        if self.W is None:
            raise ValueError("Fit the sampler before calling kernel_weights.")

        w0 = np.asarray(w0, dtype=float).reshape(-1)
        if w0.shape[0] != self.d_w:
            raise ValueError(f"w0 must have length {self.d_w}.")

        if np.isscalar(bandwidth):
            bw = np.full(self.d_w, float(bandwidth))
        else:
            bw = np.asarray(bandwidth, dtype=float).reshape(-1)
            if bw.shape[0] != self.d_w:
                raise ValueError(f"bandwidth must have length {self.d_w}.")

        if np.any(bw <= 0):
            raise ValueError("All bandwidth values must be positive.")

        diff = (self.W - w0[None, :]) / bw[None, :]
        dist2 = np.sum(diff**2, axis=1)
        weights = np.exp(-0.5 * dist2)

        s = np.sum(weights)
        if s < 1e-14:
            raise ValueError(
                "All kernel weights are ~0. Increase bandwidth or check w0."
            )

        return weights / s

    # backward-compatible alias
    def _kernel_weights(self, w0, bandwidth):
        return self.kernel_weights(w0, bandwidth)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def effective_sample_size(self, w0, bandwidth):
        """
        Effective sample size of the conditional weights.
        """
        p = self.kernel_weights(w0, bandwidth)
        return 1.0 / np.sum(p**2)

    # ------------------------------------------------------------------
    # Conditional expectations
    # ------------------------------------------------------------------
    def conditional_expectation(self, w0, bandwidth):
        """
        Estimate E[R | W=w0].
        Returns shape (d_r,)
        """
        if self.R is None:
            raise ValueError("R is not available. Fit with R first.")
        p = self.kernel_weights(w0, bandwidth)
        return np.sum(p[:, None] * self.R, axis=0)

    def conditional_mean_q(self, w0, bandwidth):
        """
        Estimate E[Q | W=w0].
        Returns shape (d_q,)
        """
        if self.Q is None:
            raise ValueError("Q is not available. Fit with Q or R first.")
        p = self.kernel_weights(w0, bandwidth)
        return self.Q @ p

    # ------------------------------------------------------------------
    # Conditional covariance
    # ------------------------------------------------------------------
    def conditional_cov_q(self, w0, bandwidth, regularization=1e-10):
        """
        Estimate Cov(Q | W=w0).
        Returns shape (d_q, d_q)
        """
        if self.Q is None:
            raise ValueError("Q is not available. Fit with Q or R first.")

        p = self.kernel_weights(w0, bandwidth)
        mu = self.Q @ p
        Qc = self.Q - mu[:, None]
        C = (Qc * p[None, :]) @ Qc.T
        C += regularization * np.eye(self.n_q)
        return C

    # ------------------------------------------------------------------
    # Conditional samples: R | W=w0
    # ------------------------------------------------------------------
    def conditional_samples(
        self, w0, bandwidth, n_samples=200, random_state=None, return_info=False
    ):
        """
        Sample from an empirical approximation of p(R | W=w0).

        Returns
        -------
        samples : ndarray, shape (n_samples, d_r)
        """
        if self.R is None:
            raise ValueError("R is not available. Fit with R first.")

        rng = self.rng if random_state is None else np.random.default_rng(random_state)
        p = self.kernel_weights(w0, bandwidth)
        idx = rng.choice(self.n_samples, size=n_samples, replace=True, p=p)
        samples = self.R[idx]

        if return_info:
            return samples, idx, p
        return samples

    # ------------------------------------------------------------------
    # Conditional samples: Q | W=w0
    # ------------------------------------------------------------------
    def sample_q_given_w(
        self,
        w0,
        bandwidth,
        n_draws=100,
        jitter_std=None,
        return_indices=False,
        random_state=None,
    ):
        """
        Sample from an empirical approximation of p(Q | W=w0).

        Returns
        -------
        Qs : ndarray, shape (d_q, n_draws)
        """
        if self.Q is None:
            raise ValueError("Q is not available. Fit with Q or R first.")

        rng = self.rng if random_state is None else np.random.default_rng(random_state)
        p = self.kernel_weights(w0, bandwidth)
        idx = rng.choice(self.n_samples, size=n_draws, replace=True, p=p)
        Qs = self.Q[:, idx].copy()

        js = self.jitter_std if jitter_std is None else float(jitter_std)
        if js > 0:
            Qs += js * rng.standard_normal(Qs.shape)

        if return_indices:
            return Qs, idx
        return Qs

    # ------------------------------------------------------------------
    # Conditional samples: X = [Q; W=w0]
    # ------------------------------------------------------------------
    def sample_x_given_w(
        self,
        w0,
        bandwidth,
        n_draws=100,
        jitter_std=None,
        return_indices=False,
        random_state=None,
    ):
        """
        Sample from an approximation of p(X | W=w0), where X = [Q; W].

        Returns
        -------
        Xs : ndarray, shape (n_x, n_draws)
        """
        if self.Q is None:
            raise ValueError("Q is not available. Fit with Q or R first.")

        w0 = np.asarray(w0, dtype=float).reshape(-1)
        if w0.shape[0] != self.d_w:
            raise ValueError(f"w0 must have length {self.d_w}.")

        Qs, idx = self.sample_q_given_w(
            w0=w0,
            bandwidth=bandwidth,
            n_draws=n_draws,
            jitter_std=jitter_std,
            return_indices=True,
            random_state=random_state,
        )

        Xs = np.zeros((self.n_x, n_draws))
        Xs[self.q_idx, :] = Qs
        Xs[self.w_idx, :] = w0[:, None]

        if return_indices:
            return Xs, idx
        return Xs

    # ------------------------------------------------------------------
    # Local Gaussian approximation
    # ------------------------------------------------------------------
    def sample_q_given_w_local_gaussian(
        self, w0, bandwidth, n_draws=100, regularization=1e-8, random_state=None
    ):
        """
        Conditional Gaussian approximation:
        Q | W=w0 ~ N(mu(w0), Cov(w0))
        """
        if self.Q is None:
            raise ValueError("Q is not available. Fit with Q or R first.")

        rng = self.rng if random_state is None else np.random.default_rng(random_state)
        mu = self.conditional_mean_q(w0, bandwidth)
        C = self.conditional_cov_q(w0, bandwidth, regularization=regularization)
        return rng.multivariate_normal(mean=mu, cov=C, size=n_draws).T
