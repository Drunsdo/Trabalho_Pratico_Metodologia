import glob, re, time
import numpy as np
import pandas as pd

RAIZ = "coloran/rome_static_medium"
COLUNAS = {"Timestamp": "ts", "IMSI": "imsi", "slice_id": "slice_id",
           "dl_buffer [bytes]": "buf", "tx_brate downlink [Mbps]": "thr",
           "tx_pkts downlink": "pkts", "dl_cqi": "cqi", "sum_requested_prbs": "req_prb"}

linhas, t0 = [], time.time()
pastas_bs = sorted(glob.glob(f"{RAIZ}/sched*/tr*/exp*/bs*"))
print("BS-corridas encontradas:", len(pastas_bs))

for k, pasta in enumerate(pastas_bs):
    sched, tr, exp, bs = map(int, re.search(r"sched(\d)/tr(\d+)/exp(\d+)/bs(\d)", pasta).groups())

    # 1) ler os 6 UEs desta BS nesta corrida
    ues = []
    for arq in sorted(glob.glob(f"{pasta}/slices_bs*/*_metrics.csv")):
        d = pd.read_csv(arq)
        d.columns = [c.strip() for c in d.columns]
        d = d[list(COLUNAS)].rename(columns=COLUNAS)
        if len(d):
            ues.append(d)
    if not ues:
        continue

    # 2) janela de regime: do 1o ao último instante com tráfego na BS, menos 30 s em cada ponta
    ativos = [u.ts[u.pkts > 0] for u in ues if (u.pkts > 0).any()]
    if not ativos:
        continue
    t_on = min(a.min() for a in ativos) / 1000 + 30
    t_off = max(a.max() for a in ativos) / 1000 - 30

    # 3) uma linha por UE: média de cada métrica dentro da janela
    for u in ues:
        j = u[(u.ts / 1000 >= t_on) & (u.ts / 1000 <= t_off)]
        linhas.append({"sched": sched, "tr": tr, "exp": exp, "bs": bs,
                       "ue": int(u.imsi.iloc[0]) % 1000,
                       "slice_id": int(u.slice_id.mode().iloc[0]),
                       "win_s": t_off - t_on, "t_start": u.ts.min() / 1000,
                       "thr": j.thr.mean(), "pkts": j.pkts.mean(), "buf": j.buf.mean(),
                       "cqi": j.cqi.mean(), "req_prb": j.req_prb.mean()})
    if k % 500 == 0:
        print(f"  {k}/{len(pastas_bs)}  ({time.time() - t0:.0f} s)")

ue = pd.DataFrame(linhas)
ue.to_csv("dados/ue_runs.csv", index=False)
print("Linhas UE-corrida:", len(ue))