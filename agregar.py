import pandas as pd

ue = pd.read_csv("dados/ue_runs.csv")
ue["pol"] = ue.sched.map({0: "RR", 1: "WF", 2: "PF"})
ue["slice"] = ue.slice_id.map({0: "eMBB", 1: "MTC", 2: "URLLC"})

# Slice de cada UE segundo o README do ColO-RAN (o desenho do experimento)
README = {0: [3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48],
          1: [4, 7, 11, 14, 18, 21, 25, 28, 32, 35, 39, 42, 46, 49],
          2: [2, 5, 9, 12, 16, 19, 23, 26, 30, 33, 37, 40, 44, 47]}
ue["slice_desenho"] = ue.ue.map({u: s for s, us in README.items() for u in us})
print("UE-corridas (arquivos):", ue.groupby("pol").size().to_dict())

# E1: corrida curta ou abortada (janela de regime < 120 s)
ue = ue[ue.win_s >= 120]
print("após E1 (janela < 120 s):", ue.groupby("pol").size().to_dict())

# E2: UE sem tráfego ofertado (fonte de tráfego falhou)
sem_trafego = (ue.pkts == 0) & (ue.buf == 0) & (ue.req_prb == 0)
ue = ue[~sem_trafego]
print("após E2 (sem tráfego):", ue.groupby("pol").size().to_dict())

# E3: slice fora do desenho (precisa ter exatamente os 2 UEs certos)
ue = ue.assign(ok=ue.slice_id == ue.slice_desenho)
grupo = ["sched", "tr", "exp", "bs", "slice_id"]
ue = ue[ue.groupby(grupo).ok.transform("all") & (ue.groupby(grupo).ue.transform("size") == 2)]
print("após E3 (fora do desenho):", ue.groupby("pol").size().to_dict())

# Médias em cascata: UE -> slice-BS -> corrida (média das 7 BSs) -> bloco (média das corridas)
m = ["thr", "pkts", "buf", "cqi"]
slice_bs = ue.groupby(["sched", "tr", "exp", "bs", "pol", "slice"])[m].mean().reset_index()
corrida  = slice_bs.groupby(["sched", "tr", "exp", "pol", "slice"])[m].mean().reset_index()
celulas  = corrida.groupby(["slice", "tr", "pol"])[m].mean().reset_index()

print("corridas válidas:", corrida[["sched", "tr", "exp"]].drop_duplicates().shape[0])
print("blocos por slice x política:\n", celulas.groupby(["slice", "pol"]).size().unstack())
celulas.to_csv("dados/celulas.csv", index=False)