"""End-to-end CLI for the GOES-19 frame interpolation pipeline.

Stages:
    download   discover + download 30-min-gap triplets from NOAA S3
    explore    run one triplet through the whole pipeline, saving diagnostics
    train      self-supervised training (U-Net + Z-map), saves checkpoints
    eval       benchmark vs linear interpolation, report + GIF + metrics
    all        download -> explore -> train -> eval
"""

import argparse
import json

from goes_interp import config, demo, setup_splatting, utils


def _banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def get_data(args):
    """Ensure GOES-19 triplet data is available locally. Returns (train, val)."""
    from goes_interp.data.download import ensure_data

    return ensure_data(target=args.triplets, hours_back=args.hours)


def get_models(device):
    """Instantiate RAFT, Z-map and U-Net (with softmax-splatting setup)."""
    setup_splatting.ensure()
    from goes_interp.models import build_models

    return build_models(device=device)


def run_download(args):
    _banner("STAGE 1/4 - DATA: GOES-19 Channel 13 triplets (30-min gaps)")
    return get_data(args)


def run_explore(args, data=None, models=None):
    _banner("STAGE 2/4 - EXPLORE: pipeline diagnostics on one training triplet")
    train_triplets = (data or get_data(args))[0]
    device = args.device or utils.get_device()
    raft, zmap, _ = models or get_models(device)
    demo.explore_triplet(train_triplets[0], raft, zmap, device)


def run_train(args, data=None, models=None):
    _banner("STAGE 3/4 - TRAIN: self-supervised on GOES-19")
    train_triplets = (data or get_data(args))[0]
    device = args.device or utils.get_device()
    raft, zmap, unet = models or get_models(device)
    if not train_triplets:
        print("No training triplets; skipping train stage.")
        return
    from goes_interp.train import train

    train(raft, zmap, unet, train_triplets, device,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


def run_eval(args, data=None, models=None):
    _banner("STAGE 4/4 - EVALUATE: model vs linear interpolation")
    train_triplets, val_triplets = data or get_data(args)
    device = args.device or utils.get_device()
    raft, zmap, unet = models or get_models(device)

    from goes_interp import visualize
    from goes_interp.evaluate import evaluate, print_metrics_table
    from goes_interp.inference import load_checkpoints

    load_checkpoints(zmap, unet, device)
    results = evaluate(val_triplets, raft, zmap, unet, device)
    print_metrics_table(results)
    visualize.plot_benchmark(results)

    best = results["best"]
    if best.get("pred") is not None:
        visualize.plot_comparison(
            best["n0"], best["linear"], best["pred"], best["gt"], best["n60"],
            ssim_lin=results["summary"]["lin_ssim"]["mean"],
            ssim_pred=results["summary"]["mod_ssim"]["mean"],
        )
        demo.generate_gif(best["n0"], best["n60"], raft, zmap, unet, device)

    _write_artifacts(results, n_train=len(train_triplets))


def _write_artifacts(results, n_train):
    from goes_interp import visualize

    summary = results["summary"]
    payload = {
        "n_train": n_train,
        "n_val": results["n_val"],
        "metrics_mean": {k: v["mean"] for k, v in summary.items()},
        "metrics_std": {k: v["std"] for k, v in summary.items()},
        "best_ssim": results["best"].get("ssim"),
        "worst_ssim": results["worst"].get("ssim"),
    }
    results_path = config.OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics: {results_path}")

    report_path = config.OUTPUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(visualize.summary_box(results, n_train, results["n_val"]) + "\n")
    print(f"Saved report: {report_path}")


def run_all(args):
    data = get_data(args)
    models = get_models(args.device or utils.get_device())
    run_explore(args, data=data, models=models)
    run_train(args, data=data, models=models)
    run_eval(args, data=data, models=models)


def build_parser():
    parser = argparse.ArgumentParser(
        description="GOES-19 satellite frame interpolation (RAFT + softmax splatting + U-Net)."
    )
    parser.add_argument("--stage", default="all",
                        choices=["download", "explore", "train", "eval", "all"],
                        help="pipeline stage to run (default: all)")
    parser.add_argument("--triplets", type=int, default=config.TARGET_TRIPLETS,
                        help="target number of triplets to collect")
    parser.add_argument("--hours", type=int, default=config.HOURS_BACK,
                        help="how far back (hours) to search the S3 bucket")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", default=None,
                        help="force a torch device, e.g. cuda or cpu")
    return parser


def main():
    args = build_parser().parse_args()
    config.ensure_dirs()
    utils.describe_device()
    utils.set_seed()
    print(f"Interpolation setup: 30-min gaps | target triplets: {args.triplets}")

    stages = {
        "download": run_download,
        "explore": run_explore,
        "train": run_train,
        "eval": run_eval,
        "all": run_all,
    }
    stages[args.stage](args)
    print("\nDone.")


if __name__ == "__main__":
    main()
