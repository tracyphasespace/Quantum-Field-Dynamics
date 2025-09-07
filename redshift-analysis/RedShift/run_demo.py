import numpy as np, pandas as pd, argparse, os
from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.kernels import te_correlation_phase
from qfd_cmb.projector import project_limber
from qfd_cmb.figures import plot_TT, plot_EE, plot_TE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--lmin", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=2500)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Planck-anchored numbers
    lA = 301.0
    rpsi = 147.0 # Mpc
    chi_star = lA * rpsi / np.pi
    sigma_chi = 250.0

    # Prepare window in χ
    chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 501)
    Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)

    # Multipoles
    ells = np.arange(args.lmin, args.lmax+1)

    # Power spectrum and projection
    Pk = lambda k: oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
    Ctt = project_limber(ells, Pk, Wchi, chi_grid)
    Cee = 0.25 * Ctt  # placeholder; use LOS with Mueller for production
    rho = np.array([te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star) for l in ells])
    Cte = rho * np.sqrt(Ctt*Cee)

    df = pd.DataFrame({"ell": ells, "C_TT": Ctt, "C_TE": Cte, "C_EE": Cee})
    csv = os.path.join(args.outdir, "qfd_demo_spectra.csv")
    df.to_csv(csv, index=False)

    plot_TT(ells, Ctt, os.path.join(args.outdir, "TT.png"))
    plot_EE(ells, Cee, os.path.join(args.outdir, "EE.png"))
    plot_TE(ells, Cte, os.path.join(args.outdir, "TE.png"))
    print(f"Wrote {csv} and PNGs in {args.outdir}")

if __name__ == "__main__":
    main()
