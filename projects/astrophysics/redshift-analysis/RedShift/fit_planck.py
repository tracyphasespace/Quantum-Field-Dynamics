import numpy as np, pandas as pd, argparse, json, os
import emcee
from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.projector import project_limber

def load_data(path):
    df = pd.read_csv(path)
    return df

def model_spectra(ells, params, chi_star, sigma_chi):
    ns, rpsi, Aosc, sigma_osc, fE = params
    chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 501)
    Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
    Pk = lambda k: oscillatory_psik(k, ns=ns, rpsi=rpsi, Aosc=Aosc, sigma_osc=sigma_osc)
    Ctt = project_limber(ells, Pk, Wchi, chi_grid)
    Cee = fE * Ctt
    Cte = 0.0*Ctt  # placeholder; extend with TE correlation model
    return Ctt, Cte, Cee

def lnlike(theta, data_df, chi_star, sigma_chi, which="TT"):
    ells = data_df["ell"].values
    Ctt, Cte, Cee = model_spectra(ells, theta, chi_star, sigma_chi)
    if which=="TT":
        model = Ctt
        data = data_df["C_TT"].values
        err = data_df["error_TT"].values if "error_TT" in data_df else np.ones_like(data)*np.std(data)*0.05
    elif which=="EE":
        model = Cee
        data = data_df["C_EE"].values
        err = data_df["error_EE"].values if "error_EE" in data_df else np.ones_like(data)*np.std(data)*0.05
    else:
        model = Cte
        data = data_df["C_TE"].values
        err = data_df["error_TE"].values if "error_TE" in data_df else np.ones_like(data)*np.std(data)*0.05
    return -0.5*np.sum(((data - model)/err)**2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with ell and C_* columns")
    ap.add_argument("--which", default="TT", choices=["TT","TE","EE"])
    ap.add_argument("--chi_star", type=float, default=14065.0)
    ap.add_argument("--sigma_chi", type=float, default=250.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--walkers", type=int, default=16)
    ap.add_argument("--out", default="fit_results.json")
    args = ap.parse_args()

    df = load_data(args.data)
    ells = df["ell"].values
    ndim = 5
    p0 = np.array([0.96, 147.0, 0.5, 0.03, 0.25])
    p0s = p0 + 1e-3*np.random.randn(args.walkers, ndim)

    def lnprob(th):
        ns, rpsi, Aosc, sigma_osc, fE = th
        if not (0.9<ns<1.05 and 80<rpsi<240 and 0.0<=Aosc<=1.0 and 0.001<sigma_osc<0.2 and 0.01<fE<1.0):
            return -np.inf
        return lnlike(th, df, args.chi_star, args.sigma_chi, which=args.which)

    sampler = emcee.EnsembleSampler(args.walkers, ndim, lnprob)
    sampler.run_mcmc(p0s, args.steps, progress=False)

    chain = sampler.get_chain(discard=args.steps//2, flat=True)
    best = chain[np.argmax([lnprob(th) for th in chain])]

    out = {"best_params": best.tolist(), "mean_params": chain.mean(axis=0).tolist()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved fit:", args.out)

if __name__ == "__main__":
    main()
