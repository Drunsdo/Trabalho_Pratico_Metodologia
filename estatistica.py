import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

cel = pd.read_csv("dados/celulas.csv")
POLS = ["RR", "WF", "PF"]
KPI = {"eMBB": "thr", "MTC": "pkts", "URLLC": "buf"}
PARES = [("RR", "WF"), ("RR", "PF"), ("WF", "PF")]

def blocos(slice_, col):
    """Tabela 28 x 3: uma linha por quota (bloco), uma coluna por política."""
    return cel[cel.slice == slice_].pivot(index="tr", columns="pol", values=col)[POLS]

def hodges_lehmann(d):
    d = np.asarray(d)
    i, j = np.triu_indices(len(d))
    return np.median((d[i] + d[j]) / 2)

def bootstrap_hl(d, B=10_000, semente=20260927):
    d = np.asarray(d)
    idx = np.random.default_rng(semente).integers(0, len(d), size=(B, len(d)))
    return np.array([hodges_lehmann(d[k]) for k in idx])

# 1) Medianas descritivas
print("1) MEDIANAS NOS 28 BLOCOS")
for sl, col in KPI.items():
    print(f"   {sl:5s}", blocos(sl, col).median().round(3).to_dict())

# 2) Shapiro-Wilk nas diferenças pareadas: justifica o não paramétrico
print("\n2) SHAPIRO-WILK (p < 0,05 = não normal)")
rejeita = 0
for sl, col in KPI.items():
    w = blocos(sl, col)
    for a, b in PARES:
        p = stats.shapiro(w[b] - w[a]).pvalue
        rejeita += p < 0.05
        print(f"   {sl:5s} {b}-{a}: p = {p:.4f}")
print(f"   -> rejeita a normalidade em {rejeita} de 9")

# 3) Friedman por slice, com Holm entre os 3 slices
print("\n3) FRIEDMAN")
fr = []
for sl, col in KPI.items():
    w = blocos(sl, col)
    n, k = w.shape
    chi2, p = stats.friedmanchisquare(w.RR, w.WF, w.PF)
    fr.append({"slice": sl, "chi2": chi2, "p": p, "W": chi2 / (n * (k - 1)),
               "posto_medio": w.rank(axis=1).mean().round(2).to_dict()})
fr = pd.DataFrame(fr)
fr["p_holm"] = multipletests(fr.p, method="holm")[1]
print(fr.round(4).to_string(index=False))

# 4) Wilcoxon pareado + Holm nos 9 contrastes; efeito HL com IC; TOST
print("\n4) CONTRASTES (efeito em % da mediana do RR)")
res = []
for sl, col in KPI.items():
    w = blocos(sl, col)
    base = w.RR.median()
    for a, b in PARES:
        d = (w[b] - w[a]).values
        boot = bootstrap_hl(d)
        ic95 = 100 * np.quantile(boot, [0.025, 0.975]) / base
        ic90 = 100 * np.quantile(boot, [0.05, 0.95]) / base
        res.append({"slice": sl, "par": f"{b}-{a}",
                    "p": stats.wilcoxon(w[b], w[a]).pvalue,
                    "efeito_%": 100 * hodges_lehmann(d) / base,
                    "IC95_%": ic95.round(1), "IC90_%": ic90.round(1),
                    "equivalente": bool(ic90[0] > -5 and ic90[1] < 5)})
res = pd.DataFrame(res)
res["p_holm"] = multipletests(res.p, method="holm")[1]
print(res[["slice", "par", "p", "p_holm", "efeito_%", "IC95_%", "IC90_%", "equivalente"]]
      .round(4).to_string(index=False))

# 5) Poder: menor efeito detectável com n = 28 (Holm no pior caso, poder 0,8)
print("\n5) EFEITO MÍNIMO DETECTÁVEL")
dz = TTestPower().solve_power(nobs=28 * 0.864, alpha=0.05 / 9, power=0.8)
for sl, col in KPI.items():
    w = blocos(sl, col)
    for a, b in PARES:
        mde = 100 * dz * (w[b] - w[a]).std() / w.RR.median()
        print(f"   {sl:5s} {b}-{a}: ~{mde:.0f}% da mediana do RR")