import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------- 1) Estilo: fonte, tamanho, sem bordas ----------
FONTE = "Helvetica"   # relatório SBC; para slides use "Helvetica"
AZUL, CINZA, FAIXA = "#2254C5", "#94A3B8", "#DCE6F8"
plt.rcParams.update({"font.family": FONTE, "font.size": 9,
                     "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False})

# ---------- 2) Mesmos cálculos do passo 6 (agora com o IC 90% do TOST) ----------
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

LINHAS = [("eMBB · vazão", "eMBB", "thr", "prim"),
          ("MTC · pacotes/250 ms", "MTC", "pkts", "prim"),
          ("MTC · vazão entregue", "MTC", "thr", "expl"),
          ("URLLC · buffer (↓)", "URLLC", "buf", "prim"),
          ("URLLC · vazão entregue", "URLLC", "thr", "expl")]
tab = []
for rotulo, sl, col, tipo in LINHAS:
    w = blocos(sl, col)
    base = w["RR"].median()
    for b in ["WF", "PF"]:
        d = (w[b] - w["RR"]).values
        boot = bootstrap_hl(d)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        lo90, hi90 = np.quantile(boot, [0.05, 0.95])
        p = np.nan
        if tipo == "prim":
            p = testes[(testes.slice == sl) & (testes.par == f"{b} vs RR")].p_holm.iloc[0]
        tab.append({"rotulo": rotulo, "par": f"{b} vs RR", "tipo": tipo,
                    "m": 1 + hodges_lehmann(d) / base,
                    "lo": 1 + lo / base, "hi": 1 + hi / base,
                    "lo90": 1 + lo90 / base, "hi90": 1 + hi90 / base, "p": p})
tab = pd.DataFrame(tab)

def fmt_p(p):
    return "<0,001" if p < 0.001 else f"{p:.2f}".replace(".", ",")

# ---------- 3) Desenho ----------
fig, ax = plt.subplots(figsize=(5.1, 2.6))          
ax.axvspan(0.95, 1.05, color=FAIXA, lw=0, zorder=1)
ax.axvline(1.0, color="black", lw=0.7, ls="--", zorder=2)

for i, (rotulo, sl, col, tipo) in enumerate(LINHAS):
    y = -i
    cor = AZUL if tipo == "prim" else CINZA
    linha = tab[tab.rotulo == rotulo]
    for _, r in linha.iterrows():
        dy, marcador = (0.16, "s") if r.par == "WF vs RR" else (-0.16, "o")
        cheio = pd.notna(r.p) and r.p < 0.05
        ax.plot([r.lo, r.hi], [y + dy, y + dy], color=cor, lw=1.5,
                solid_capstyle="round", zorder=3)
        ax.plot(r.m, y + dy, marcador, ms=5, color=cor,
                mfc=cor if cheio else "white", mew=1.2, zorder=4)
    # texto à direita: p ajustado (primárias) ou equivalência TOST (exploratórias)
    wf, pf = linha.iloc[0], linha.iloc[1]
    if tipo == "prim":
        txt = f"$p_H$: WF {fmt_p(wf.p)} · PF {fmt_p(pf.p)}"
    elif all((q.lo90 > 0.95) and (q.hi90 < 1.05) for q in (wf, pf)):
        txt = "equivalentes (TOST ±5%)"
    else:
        txt = ""
    ax.text(2.75, y, txt, va="center", ha="left", fontsize=8, color=cor)

# ---------- 4) Eixos ----------
ax.set_xscale("log")
ax.minorticks_off()
ax.set_xticks([0.5, 0.75, 1, 1.25, 2])
ax.set_xticklabels(["−50%", "−25%", "0", "+25%", "+100%"])
ax.tick_params(axis="x", labelsize=8)
ax.set_xlim(0.42, 2.7)
ax.set_ylim(-4.55, 0.5)
ax.set_yticks([-i for i in range(len(LINHAS))])
ax.set_yticklabels([l[0] for l in LINHAS])
for t, l in zip(ax.get_yticklabels(), LINHAS):          
    t.set_color(AZUL if l[3] == "prim" else CINZA)
    t.set_fontweight("bold" if l[3] == "prim" else "normal")
ax.tick_params(axis="y", length=0)
ax.set_xlabel("efeito relativo ao RR (Hodges–Lehmann ÷ mediana RR), IC 95% bootstrap, escala log",
              fontsize=8)

# ---------- 5) Legenda acima do gráfico ----------
legenda = [Line2D([], [], marker="s", color=AZUL, mfc="white", lw=1.4, ms=5, label="WF vs RR"),
           Line2D([], [], marker="o", color=AZUL, mfc="white", lw=1.4, ms=5, label="PF vs RR"),
           Line2D([], [], marker="o", color=AZUL, mfc=AZUL, lw=0, ms=5,
                  label="cheio: $p_{Holm}<0{,}05$"),
           Line2D([], [], marker="s", color=CINZA, mfc="white", lw=1.4, ms=4.5,
                  label="cinza: exploratório"),
           Patch(color=FAIXA, label="SESOI ±5%")]
ax.legend(handles=legenda, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
          frameon=False, fontsize=7.5, columnspacing=1.0, handletextpad=0.35, handlelength=1.5)

# ---------- 6) Exportar ----------
plt.tight_layout(pad=0.2)
plt.savefig("forest_relatorio.pdf", bbox_inches="tight", pad_inches=0.02)
plt.savefig("forest_relatorio.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("Exportado: forest_relatorio.pdf e forest_relatorio.png")