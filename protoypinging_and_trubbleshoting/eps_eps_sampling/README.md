# Epsilon sampling diagnostics

STANDING COLLABORATION RULE
Use plain English, short sentences, active verbs, precise terms, and a natural
voice. Help the researcher understand and own the work. For substantial changes,
show the relevant files, how the parts connect, and a small usage example with
expected outputs. Use real code or clearly labelled pseudocode. State what was
tested. Link to the implementation, settings, and results. Keep reusable examples
in the relevant docs and the layout in docs/repo_map.md. Involve the researcher
in meaningful choices without routine approval requests. Apply this rule to
future notes and documentation. Full guidance: root AGENTS.md, Collaboration
and Code Understanding.

This directory investigates the failed Flowers102 `eps/eps` sampling run.
The saved model has a low recorded loss, but its 50-step samples contain noise.

This result does not establish that `eps/eps` fails in general. Historical
CelebA32 grayscale runs produced good FID values under a different protocol.
See [`OPEN_PROBLEM.md`](OPEN_PROBLEM.md) for the unresolved comparison.

The investigation separates two questions:

1. Does a finer Heun grid repair the fixed learned vector field?
2. Does epsilon error become large velocity error near `t = 0.01`?

## Directory structure

```text
eps_eps_sampling/
  OPEN_PROBLEM.md
  scripts/
    replay_heun_resolution.py
    measure_eps_conditioning.py
  notebooks/
    eps_eps_heun_resolution.ipynb
```

## Solver-resolution test

Run the following command from the repository root:

```bash
PYTHONPATH=. conda run -n memcompgen python \
  protoypinging_and_trubbleshoting/eps_eps_sampling/scripts/replay_heun_resolution.py \
  --run-dir outputs/sanity/flowers102_runner_eps_eps_exact_64_n8_10k_20260820T191129Z \
  --steps 200 500 \
  --device mps
```

The script replays every saved checkpoint. It uses the saved data and the fixed
noise bank. It changes only `sampling.n_steps`.

The script creates these named evaluations:

```text
evaluations/heun_200_exact/
evaluations/heun_500_exact/
```

The script reuses a complete evaluation when its settings match. It rejects a
partial directory or a directory with different settings.

## Endpoint-conditioning test

Run the following command from the repository root:

```bash
PYTHONPATH=. conda run -n memcompgen python \
  protoypinging_and_trubbleshoting/eps_eps_sampling/scripts/measure_eps_conditioning.py \
  --run-dir outputs/sanity/flowers102_runner_eps_eps_exact_64_n8_10k_20260820T191129Z \
  --device mps
```

The script evaluates the final checkpoint on exact interpolation states. It
uses the saved training tensor and the saved noise bank. It records epsilon
error and the induced velocity error at fixed times.

The report is saved inside the run directory:

```text
diagnostics/eps_eps_conditioning_step_010000.json
```

This test does not inspect generated trajectory states. It measures the model
on controlled path states where the true epsilon and velocity are known.

## Interpretation

- Stable 200- and 500-step samples indicate solver convergence.
- Bad but stable 200- and 500-step samples indicate a learned-field problem.
- Large early induced velocity error indicates `1/t` amplification.
- Irregular changes across step counts indicate stiffness or off-path errors.

Do not change the division setting, time interval, output, noise bank, or model
during the solver-resolution test.

## Current result

The 200- and 500-step samples converged to the same failed result. Their sample
RMSE is `0.002376`, but both grids remain colored noise. The final mean `d1`
values are `268.53` and `268.38`.

At `t = 0.01`, epsilon RMSE is `0.0277`. The induced velocity RMSE is `2.7713`
because the conversion amplifies error by `100`.

Coarse Heun resolution is not the main failure cause for this saved model. The
historical CelebA32 result makes this a cross-protocol problem. Test the old
`0.05` sampling clamp first. Then test the same Flowers images at RGB64, RGB32,
and grayscale32. This checks whether the 12-fold input-dimension increase is
part of the failure. Reproduce one old CelebA32 run after these controls. Keep
the `v/v` plus `eps_equivalent` comparison as a later representation control.
