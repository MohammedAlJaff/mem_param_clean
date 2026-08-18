# Objective

This repository supports the paper:

```text
Equivalent in Principle, Unequal in Practice
```

The study asks whether algebraically related flow-matching targets become
different finite-capacity learning problems.

The controlled comparison holds these fixed:

- architecture
- data
- optimizer
- sampler
- training budget
- evaluation protocol

The target parameterizations are:

- `x`
- `eps`
- `v`

Primary measurements:

- `f_mem`
- nearest-neighbor ratio `r = d1 / d2`
- nearest-neighbor distances `d1` and `d2`
- checkpoint-level memorization trajectories
- quality-controlled memorization summaries
