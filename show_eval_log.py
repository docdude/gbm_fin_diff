#!/usr/bin/env python3
"""Pretty-print eval_log.jsonl as a comparison table.

Usage:
    python show_eval_log.py                # show key ratios
    python show_eval_log.py --full         # show all fields
    python show_eval_log.py --csv          # dump as CSV
"""
import json, os, sys, argparse

LOG_PATH = os.path.join(os.path.dirname(__file__), "eval_log.jsonl")

KEY_COLS = [
    ("checkpoint",          "checkpoint",              "s",   30),
    ("sampler",             "sampler",                 "s",   22),
    ("eps",                 "eps",                     "g",   10),
    ("ret_std_ratio",       "return_std_ratio",        ".2f",  9),
    ("AC1_gen",             "return_ac1_gen",          "+.3f", 9),
    ("QV_ratio",            "quadratic_var_ratio",     ".2f",  9),
    ("MaxDD_ratio",         "max_drawdown_ratio",      ".2f", 10),
    ("VoV_ratio",           "vol_of_vol_ratio",        ".2f",  9),
    ("a_gen",               "alpha_gen",               ".2f",  7),
    ("a_real",              "alpha_real",              ".2f",  7),
    ("b_gen",               "beta_gen",                ".3f",  7),
    ("b_real",              "beta_real",               ".3f",  7),
    ("Lev_L1",              "leverage_L1_gen",         "+.3f", 8),
    ("TP_dens_ratio",       "turning_pt_density_ratio",".2f", 12),
]


def load_records():
    if not os.path.exists(LOG_PATH):
        print(f"No log file found at {LOG_PATH}")
        sys.exit(1)
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_table(records):
    # Header
    header = "  ".join(f"{h:<{w}}" for h, _, _, w in KEY_COLS)
    print(header)
    print("-" * len(header))
    for rec in records:
        parts = []
        for _, key, fmt, w in KEY_COLS:
            val = rec.get(key)
            if val is None:
                cell = "N/A"
            elif fmt == "s":
                # Shorten checkpoint path
                cell = str(val)
                if len(cell) > w:
                    cell = "..." + cell[-(w-3):]
            else:
                cell = f"{val:{fmt}}"
            parts.append(f"{cell:<{w}}")
        print("  ".join(parts))


def print_csv(records):
    if not records:
        return
    keys = list(records[0].keys())
    print(",".join(keys))
    for rec in records:
        print(",".join(str(rec.get(k, "")) for k in keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Show all fields as JSON")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    records = load_records()
    print(f"Loaded {len(records)} eval records from {LOG_PATH}\n")

    if args.csv:
        print_csv(records)
    elif args.full:
        for i, rec in enumerate(records):
            print(f"--- Run {i+1} ---")
            print(json.dumps(rec, indent=2))
    else:
        print_table(records)


if __name__ == "__main__":
    main()
