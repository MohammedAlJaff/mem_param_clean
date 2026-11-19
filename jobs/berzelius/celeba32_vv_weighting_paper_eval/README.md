# CelebA32 v/v loss-weighting paper evaluation

This 18-task array evaluates the complete native-`v/v` loss-weighting study:

- training sizes `N={512,2048}`;
- training seeds `{0,1,2}`;
- loss weights `x_equivalent=(1-t)^2`, `unit=1`, and `eps_equivalent=t^2`;
- 72 saved checkpoints per run;
- 10,000 generated images per checkpoint.

Each task uses the same seed-2003 canonical evaluation-noise bank. The source
runs keep their original monitoring noise. Each task computes the complete
natural-image metric suite against the frozen clean CelebA test bank. It also
computes held-out losses on 512 images.

The launcher freezes the evaluator source, environment, source paths, all
checkpoint hashes, reference hashes, noise hashes, and encoder hashes before
submission. A worker invokes the evaluator twice. The second invocation must
reuse every expensive artifact. The final validator checks all 72 checkpoints,
finite metrics, bounded memorization fractions, sample shapes, readable plots,
paper-tier manifest fields, and cache reuse.

Preview the command without creating a frozen batch:

```bash
bash jobs/berzelius/celeba32_vv_weighting_paper_eval/submit.sh --dry-run
```

Submit all 18 tasks without a concurrency throttle:

```bash
bash jobs/berzelius/celeba32_vv_weighting_paper_eval/submit.sh
```

Restart one task from its first incomplete checkpoint:

```bash
bash jobs/berzelius/celeba32_vv_weighting_paper_eval/submit.sh \
  --resume-root outputs/evals/celeba32_vv_weighting_full_10000/TIMESTAMP \
  --tasks 3
```

The default request for each task is one thin A100-40GB GPU, eight CPUs,
64 GB memory, normal QoS, and 12 hours. The retained artifacts can use about
30 GB per run, or about 540 GB for the complete matrix. Do not remove feature
or sample artifacts until the paper figure contract is frozen.
