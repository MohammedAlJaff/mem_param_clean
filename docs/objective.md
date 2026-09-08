# Objective

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
