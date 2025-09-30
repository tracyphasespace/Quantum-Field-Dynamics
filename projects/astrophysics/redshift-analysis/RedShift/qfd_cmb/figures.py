import numpy as np
import matplotlib.pyplot as plt

def plot_TT(ells, Ctt, path):
    plt.figure()
    plt.loglog(ells, ells*(ells+1)*Ctt)
    plt.xlabel(r"$\ell$"); plt.ylabel(r"$\ell(\ell+1) C_\ell^{TT}$")
    plt.title("TT (monochrome)")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def plot_EE(ells, Cee, path):
    plt.figure()
    plt.loglog(ells, ells*(ells+1)*Cee)
    plt.xlabel(r"$\ell$"); plt.ylabel(r"$\ell(\ell+1) C_\ell^{EE}$")
    plt.title("EE (monochrome)")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def plot_TE(ells, Cte, path):
    plt.figure()
    sign = np.sign(Cte + 1e-30)
    plt.semilogx(ells, sign * ells*(ells+1)*Cte)
    plt.axhline(0, lw=0.8)
    plt.xlabel(r"$\ell$"); plt.ylabel(r"sign×$\ell(\ell+1) C_\ell^{TE}$")
    plt.title("TE (monochrome)")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
