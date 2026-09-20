# PR report (IEEE conference format)

- `main.tex`: report source (IEEEtran, `IEEEtran.cls` copied from the IEEE conference template).
- `main.pdf`: compiled report.
- `data/`: metric files copied from the experiment outputs (`results/metrics/`), the only inputs of the figures and tables. `data/headfp1/` holds the reported model (head + last propagation layer); the files directly in `data/` for the fine-tuned model are the head-only ablation.
- `data/decoder/`: the third rung of the capacity ladder (head + fp1-fp3, 743,041 parameters): comparison tables, per-affordance held-out IoU, training history and configuration.
- `data/baselines/`: the query-independent floors (majority class and random class) with per-class IoU.
- `figures/raw/`: before/after renderings written by `python -m src.visualization.qualitative`.
- `make_figures.py`: builds `figures/fig_results.pdf`, `fig_perclass.pdf`, `fig_qual_a.png`, `fig_qual_b.png` from `data/` and `figures/raw/`.

Rebuild:

```bash
cd docs/report-pr
python make_figures.py
pdflatex main.tex && pdflatex main.tex
```

Table values in `main.tex` were typed from `data/` (the result tables) and from the training record (`data/finetune_history.csv`).
