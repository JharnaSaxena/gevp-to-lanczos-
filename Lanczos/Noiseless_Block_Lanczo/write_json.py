#!/usr/bin/env python3
"""Dump every Ritz value to output/ritz_values.json"""
import json
import os
from mpmath import mp
from lanczos_core import (build_mock, extract_all, OUTPUT_DIR,)
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    C, E, _Zt, _Z = build_mock(32)
    histories = {
        "block":      extract_all(C, 8, 2),
        "scalar_C00": extract_all([mp.matrix([[C[t][0,0]]]) for t in range(32)], 16, 1),
        "scalar_C11": extract_all([mp.matrix([[C[t][1,1]]]) for t in range(32)], 16, 1),
    }
    data = {}
    for name, hist in histories.items():
        data[name] = [
            {"m": e["m"], "t": e["t"],
             "energies": [str(mp.re(v)) for v in e["energies"]],
             "lambdas":  [str(mp.re(v)) for v in e["lambdas"]]}
            for e in hist
        ]
    path = f"{OUTPUT_DIR}/ritz_values.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[ritz_values.json]  written  →  {path}")

if __name__ == "__main__":
    main()
