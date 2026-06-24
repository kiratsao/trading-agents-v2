import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.verify_engine import run_backtest, _load_data

def main():
    df = _load_data()
    
    fast_ranges = [20, 30, 40]
    slow_ranges = [80, 100, 120]
    mult_ranges = [1.5, 2.0, 2.5]
    
    results = []
    
    for fast in fast_ranges:
        for slow in slow_ranges:
            for mult in mult_ranges:
                print(f"Testing: Fast={fast}, Slow={slow}, Mult={mult} ...")
                ec, trades, metrics = run_backtest(
                    df,
                    ema_fast=fast,
                    ema_slow=slow,
                    atr_stop_mult=mult,
                )
                if metrics:
                    results.append({
                        "fast": fast,
                        "slow": slow,
                        "mult": mult,
                        "CAGR": metrics["CAGR_%"],
                        "MDD": metrics["MDD_%"],
                        "Sharpe": metrics["Sharpe"],
                        "Calmar": metrics["Calmar"],
                    })
                    
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("CAGR", ascending=False)
    print("\nTop Results:")
    print(res_df.head(10).to_string())
    
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    res_df.to_csv(out_dir / "optimization_results.csv", index=False)
    print(f"\nSaved results to results/optimization_results.csv")

if __name__ == "__main__":
    main()
