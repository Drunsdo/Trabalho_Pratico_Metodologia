# Como rodar

## 1. Instalar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy statsmodels matplotlib
```

## 2. Baixar os dados

```bash
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/wineslab/colosseum-oran-coloran-dataset.git coloran
cd coloran
git sparse-checkout init --no-cone
printf '/README.md\n/rome_static_medium/*/tr*/*/*/*/*_metrics.csv\n' > .git/info/sparse-checkout
git checkout master
cd ..
```

## 3. Rodar

```bash
mkdir -p dados
python extrair.py
python agregar.py
python estatistica.py
python grafico.py
python acabamento.py
```

| Script | Gera |
|---|---|
| `extrair.py` | `dados/ue_runs.csv` |
| `agregar.py` | `dados/celulas.csv` |
| `estatistica.py` | resultados dos testes no terminal |
| `grafico.py` | `forest.pdf` e `forest.png` |
| `acabamento.py` | `forest_relatorio.pdf` e `forest_relatorio.png` |