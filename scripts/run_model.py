"""Profile one model and print even-vs-balanced comparisons."""
import argparse
import json

from pp_balance.profiler import profile_model
from pp_balance.report import compare
from pp_balance.validate import check_reconstruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pp-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = profile_model(args.model, args.prompt_tokens,
                        args.decode_steps, args.repeats)
    check_reconstruction(sum(res.prefill_ms), res.model_fwd_ms)

    rows = []
    for regime, costs in (("prefill", res.prefill_ms),
                          ("decode", res.decode_ms)):
        for pp in args.pp_sizes:
            row = compare(costs, [0] * len(costs), pp, float("inf"))
            rows.append({"regime": regime, **row.__dict__})
            print(f"{regime:8s} pp={pp}  even={row.even_max_cost:8.3f}ms  "
                  f"balanced={row.balanced_max_cost:8.3f}ms  "
                  f"reduction={row.reduction_pct:5.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "ctx": args.prompt_tokens,
                       "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
