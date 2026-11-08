"""``python -m leansteer.cli <stage> -c configs/<cfg>.yaml [--models m ...] [--limit N]``

stages: data | download | collect | probe | steer | report | all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import yaml

from leansteer import config  # noqa: F401
from leansteer.experiment import load_config

STAGES = ("data", "download", "collect", "collect-tf", "probe", "probe-tf", "steer", "prove-gen", "prove-verify",
          "controls", "uncertainty", "abstain", "modernbert-tf", "transfer", "ext-collect", "ext-generate", "ext-prove-gen", "ext-prove-verify", "ext-eval", "steer-variant", "prove-variant-gen", "prove-variant-verify", "collect-heads", "probe-heads", "steer-heads", "prove-heads-gen", "prove-heads-verify", "baselines", "modernbert", "relabel-select", "relabel-gen", "relabel-verify", "relabel-eval", "report", "all")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--anchors", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only N step variants (smoke tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--variant", default=None, help="steering variant name (steer-variant / prove-variant-*)")
    ap.add_argument("--ext", default=None, help="external dataset name (ext-* stages): prm800k | minif2f")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, dotted keys, YAML values (e.g. steer.batch_size=8)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    if args.models:
        cfg["models"] = args.models
    if args.tasks:
        cfg["tasks"] = args.tasks
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.device:
        cfg["device"] = args.device
    if args.run_name:
        cfg["run_name"] = args.run_name
    for item in args.set:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    stages = ["collect", "probe", "steer", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "data":
            from leansteer.data.faithform import load_af, task_summary
            df = load_af(models=cfg["models"])
            print(task_summary(df, cfg["balance"], cfg["analysis"]["seed"]).to_string(index=False))
        elif stage == "download":
            from leansteer.models.download import download_models
            download_models(cfg["models"])
        elif stage == "collect":
            from leansteer.run.collect import run_collect
            for m in cfg["models"]:
                run_collect(cfg, m)
        elif stage == "probe":
            from leansteer.analysis.probes import run_probes
            for m in cfg["models"]:
                run_probes(cfg, m, tasks=args.tasks, anchors=args.anchors)
        elif stage == "steer":
            from leansteer.run.steer import run_steer
            for m in cfg["models"]:
                run_steer(cfg, m, tasks=args.tasks)
        elif stage == "collect-tf":
            from leansteer.run.collect import run_collect_tf
            for m in cfg["models"]:
                run_collect_tf(cfg, m)
        elif stage == "probe-tf":
            from leansteer.analysis.probes import run_probes
            for m in cfg["models"]:
                run_probes(cfg, m, tasks=args.tasks or cfg["tf"]["tasks"], anchors=args.anchors or cfg["tf"]["anchors"], store="acts_tf")
        elif stage == "controls":
            from leansteer.analysis.controls import run_controls
            for m in cfg["models"]:
                run_controls(cfg, m)
        elif stage == "steer-variant":
            from leansteer.run.variants import run_steer_variant
            for m in cfg["models"]:
                run_steer_variant(cfg, m, args.variant, tasks=args.tasks)
        elif stage == "prove-variant-gen":
            from leansteer.run.prove import run_prove_gen
            run_prove_gen(cfg, which=f"steer_{args.variant}")
        elif stage == "prove-variant-verify":
            from leansteer.run.prove import run_prove_verify
            print(run_prove_verify(cfg, which=f"steer_{args.variant}"))
        elif stage == "ext-collect":
            from leansteer.run.external import run_ext_collect
            for m in cfg["models"]:
                run_ext_collect(cfg, args.ext, m)
        elif stage == "ext-generate":
            from leansteer.run.external import run_ext_generate
            for m in cfg["models"]:
                run_ext_generate(cfg, args.ext, m)
        elif stage == "ext-prove-gen":
            from leansteer.run.prove import run_prove_gen
            run_prove_gen(cfg, which=f"ext_{args.ext}")
        elif stage == "ext-prove-verify":
            from leansteer.run.prove import run_prove_verify
            print(run_prove_verify(cfg, which=f"ext_{args.ext}"))
        elif stage == "ext-eval":
            from leansteer.analysis.external_eval import run_ext_eval
            print(json.dumps(run_ext_eval(cfg, args.ext), indent=1)[:3000])
        elif stage == "transfer":
            from leansteer.analysis.probes import cross_task_transfer
            from leansteer.data.faithform import TASKS, load_af
            from leansteer.experiment import Run, write_json
            from leansteer.run.collect import load_acts
            run = Run(cfg)
            for m in cfg["models"]:
                df = load_af(models=[m])
                for anchor in cfg["analysis"]["anchors"]:
                    X, row_of, index = load_acts(run, m, anchor)
                    have = [t for t in TASKS if (run.probe_dir(m, anchor, t) / "result.json").exists()]
                    write_json(run.model_dir(m) / "probes" / anchor / "transfer.json", cross_task_transfer(run, m, anchor, df, cfg, have, X, row_of, index))
        elif stage == "abstain":
            from leansteer.analysis.abstain import run_abstain
            for m in cfg["models"]:
                run_abstain(cfg, m)
        elif stage == "uncertainty":
            from leansteer.analysis.uncertainty import run_uncertainty
            for m in cfg["models"]:
                run_uncertainty(cfg, m, tasks=args.tasks)
        elif stage == "collect-heads":
            from leansteer.run.collect import run_collect_heads
            for m in cfg["models"]:
                run_collect_heads(cfg, m)
        elif stage == "probe-heads":
            from leansteer.analysis.heads import run_head_probes
            for m in cfg["models"]:
                run_head_probes(cfg, m, tasks=args.tasks, anchors=args.anchors)
        elif stage == "steer-heads":
            from leansteer.run.steer import run_steer_heads
            for m in cfg["models"]:
                run_steer_heads(cfg, m, tasks=args.tasks)
        elif stage == "prove-heads-gen":
            from leansteer.run.prove import run_prove_gen
            run_prove_gen(cfg, which="steer_heads")
        elif stage == "prove-heads-verify":
            from leansteer.run.prove import run_prove_verify
            print(run_prove_verify(cfg, which="steer_heads"))
        elif stage == "prove-gen":
            from leansteer.run.prove import run_prove_gen
            run_prove_gen(cfg)
        elif stage == "prove-verify":
            from leansteer.run.prove import run_prove_verify
            print(run_prove_verify(cfg))
        elif stage == "relabel-select":
            from leansteer.analysis.relabel import run_relabel_select
            print(run_relabel_select(cfg))
        elif stage == "relabel-gen":
            from leansteer.run.prove import run_prove_gen
            run_prove_gen(cfg, which="relabel")
        elif stage == "relabel-verify":
            from leansteer.run.prove import run_prove_verify
            print(run_prove_verify(cfg, which="relabel"))
        elif stage == "relabel-eval":
            from leansteer.analysis.relabel import run_relabel_eval
            print(run_relabel_eval(cfg))
        elif stage == "baselines":
            from leansteer.analysis.baselines import run_baselines
            for m in cfg["models"]:
                run_baselines(cfg, m)
        elif stage == "modernbert":
            from leansteer.analysis.modernbert import run_modernbert
            for m in cfg["models"]:
                run_modernbert(cfg, m, tasks=args.tasks, seeds=int(cfg["modernbert"].get("seeds", 1)))
        elif stage == "modernbert-tf":
            from leansteer.analysis.modernbert import run_modernbert
            for m in cfg["models"]:
                run_modernbert(cfg, m, tasks=args.tasks or cfg["tf"]["tasks"], inputs="step_statement",
                               seeds=int(cfg["modernbert"].get("seeds", 1)))
        elif stage == "report":
            from leansteer.analysis.report import run_report
            print(run_report(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
