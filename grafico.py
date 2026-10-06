import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

cel = pd.read_csv("dados/celulas.csv")
POLS = ["RR", "WF", "PF"]

def blocos(slice_, col):
    return cel[cel.slice == slice_].pivot(index="tr", columns="pol", values=col)[POLS]

def hodges_lehmann(d):
    d = np.asarray(d)
    i, j = np.triu_indices(len(d))
    return np.median((d[i] + d[j]) / 2)

def bootstrap_hl(d, B=10_000, semente=20260927):
    d = np.asarray(d)
    idx = np.random.default_rng(semente).integers(0, len(d), size=(B, len(d)))
    return np.array([hodges_lehmann(d[k]) for k in idx])

# p ajustado (Holm) dos 9 contrastes: define marcador cheio ou vazio
KPI = {"eMBB": "thr", "MTC": "pkts", "URLLC": "buf"}
PARES = [("RR", "WF"), ("RR", "PF"), ("WF", "PF")]
testes = []
for sl, col in KPI.items():
    w = blocos(sl, col)
    for a, b in PARES:
        testes.append({"slice": sl, "par": f"{b} vs {a}",
                       "p": stats.wilcoxon(w[b], w[a]).pvalue})
testes = pd.DataFrame(testes)
testes["p_holm"] = multipletests(testes.p, method="holm")[1]

# uma linha do gráfico por métrica; "expl" = exploratória (vazão entregue)
LINHAS = [("eMBB · vazão", "eMBB", "thr", "prim"),
          ("MTC · pacotes", "MTC", "pkts", "prim"),
          ("MTC · vazão", "MTC", "thr", "expl"),
          ("URLLC · buffer ↓", "URLLC", "buf", "prim"),
          ("URLLC · vazão", "URLLC", "thr", "expl")]
tab = []
for rotulo, sl, col, tipo in LINHAS:
    w = blocos(sl, col)
    base = w["RR"].median()
    for b in ["WF", "PF"]:
        d = (w[b] - w["RR"]).values
        boot = bootstrap_hl(d)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        p = np.nan
        if tipo == "prim":
            p = testes[(testes.slice == sl) & (testes.par == f"{b} vs RR")].p_holm.iloc[0]
        tab.append({"rotulo": rotulo, "par": f"{b} vs RR", "tipo": tipo,
                    "m": 1 + hodges_lehmann(d) / base,      # razão: 1 = igual ao RR
                    "lo": 1 + lo / base, "hi": 1 + hi / base, "p": p})
tab = pd.DataFrame(tab)
print(tab.round(3))

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.axvspan(0.95, 1.05, color="#DCE6F8")            # faixa ±5% (SESOI)
ax.axvline(1.0, color="gray", ls="--", lw=0.8)     # efeito zero
for i, (rotulo, *_) in enumerate(LINHAS):
    y = -i
    for _, r in tab[tab.rotulo == rotulo].iterrows():
        dy, marcador = (0.15, "s") if r.par == "WF vs RR" else (-0.15, "o")
        cor = "#2254C5" if r.tipo == "prim" else "#94A3B8"
        ax.plot([r.lo, r.hi], [y + dy, y + dy], color=cor)                       # barra = IC 95%
        cheio = pd.notna(r.p) and r.p < 0.05
        ax.plot(r.m, y + dy, marcador, color=cor, mfc=cor if cheio else "white")  # ponto = HL
ax.set_xscale("log")
ax.set_xticks([0.5, 0.75, 1, 1.5, 2])
ax.set_xticklabels(["−50%", "−25%", "0", "+50%", "+100%"])
ax.minorticks_off()
ax.set_yticks([-i for i in range(len(LINHAS))])
ax.set_yticklabels([l[0] for l in LINHAS])
ax.set_xlabel("efeito relativo ao RR (Hodges–Lehmann, IC 95%)")
plt.tight_layout()
plt.savefig("forest.pdf", bbox_inches="tight")            
plt.savefig("forest.png", dpi=300, bbox_inches="tight")   
print("Exportado: forest.pdf e forest.png")