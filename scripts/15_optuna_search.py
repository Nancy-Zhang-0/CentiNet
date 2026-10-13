"""Optuna hyperparameter search on the winning architecture.

Each trial launches scripts/03_train.py as a subprocess with sampled overrides
and reads back the best validation CL_WC MAE (best_summary.json). Trials use a
shortened 40-epoch cosine schedule (patience 10) for screening; re-train the
top configurations with the full schedule afterwards.

Leakage rules: the objective is VAL MAE only; the test split is never touched.
Study state persists in results/optuna_study.db (resumable). Trial run dirs go
under runs_optuna/ so the main runs/ stays clean.

Usage:
  python scripts/15_optuna_search.py --n_trials 24
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROJECT_ROOT  # noqa: E402

PY = sys.executable
TRAIN = str(PROJECT_ROOT / "scripts" / "03_train.py")


def objective(trial: optuna.Trial, args) -> float:
    lr = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
    wd = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    drop_path = trial.suggest_float("drop_path", 0.0, 0.2)
    norm = trial.suggest_categorical("norm", ["bn", "gn"])
    pool = trial.suggest_categorical("pool", ["cls", "cls+mean"])
    noise_std = trial.suggest_float("noise_std", 0.01, 0.05)
    rotate_deg = trial.suggest_float("rotate_deg", 10.0, 20.0)

    overrides = [
        f"train.model={args.model}", f"train.seed={args.seed}",
        f"train.lr={lr}", f"train.weight_decay={wd}", f"train.dropout={dropout}",
        f"train.model_kwargs.drop_path={drop_path}",
        f"train.model_kwargs.norm={norm}", f"train.model_kwargs.pool={pool}",
        f"augment.noise_std={noise_std}", f"augment.rotate_deg={rotate_deg}",
        f"train.epochs={args.epochs}", f"train.patience={args.patience}",
        f"train.num_workers={args.num_workers}",
        f"paths.runs_dir={args.runs_dir}",
    ]
    cmd = [PY, TRAIN] + [x for o in overrides for x in ("--set", o)]
    print(f"[trial {trial.number}] {json.dumps(trial.params)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=PROJECT_ROOT, timeout=3600 * 4)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
        raise optuna.TrialPruned(f"training failed (rc={proc.returncode})")

    # the run dir is printed by 03_train; recover it from stdout
    run_dir = None
    for line in proc.stdout.splitlines():
        if line.startswith("run dir: "):
            run_dir = Path(line.split("run dir: ")[1].split(" (")[0])
            break
    if run_dir is None:
        raise optuna.TrialPruned("run dir not found in trainer output")
    best = json.loads((run_dir / "best_summary.json").read_text())
    mae = float(best["best_val_mae_CL_WC"])
    trial.set_user_attr("run_dir", str(run_dir))
    print(f"[trial {trial.number}] val CL_WC MAE = {mae:.3f}", flush=True)
    return mae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet_transformer_hybrid")
    ap.add_argument("--n_trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--runs_dir", default="runs_optuna")
    ap.add_argument("--study_name", default="deep-amyloid-hpo")
    args = ap.parse_args()

    (PROJECT_ROOT / args.runs_dir).mkdir(exist_ok=True)
    storage = f"sqlite:///{PROJECT_ROOT / 'results' / 'optuna_study.db'}"
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials,
                   gc_after_trial=True)

    print("\n=== best trial ===")
    print(f"val CL_WC MAE {study.best_value:.3f}")
    print(json.dumps(study.best_params, indent=2))
    df = study.trials_dataframe()
    df.to_csv(PROJECT_ROOT / "results" / "optuna_trials.csv", index=False)
    print(f"trials table -> results/optuna_trials.csv")


if __name__ == "__main__":
    main()
