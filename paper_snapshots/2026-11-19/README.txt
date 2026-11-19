PAPER WORKSPACE SNAPSHOT: 2026-11-19

This snapshot preserves the local paper files before new rebuttal work.
It is not a record of the submitted manuscript. Figure review states remain
as recorded. No notebook, plot, training, or evaluation was rerun.

Source code commit: 8c43c8662d7ade6468e8df5730f5bae5ad7096a1
Archive: paper_figures.zip
Contents: 342 files from outputs/paper_figures/.
Includes PDFs, SVGs, captions, Overleaf files, previews, references, and
existing build versions with their archived plotting source.
The original folder layout and file bytes are preserved.

manifest.json lists every archived path, byte size, and SHA-256 hash.
It also records the source code commit and the existing notebook hash.
SHA256SUMS checks the archive and manifest together.

From the repository root, verify the saved files:
  shasum -a 256 -c paper_snapshots/2026-11-19/SHA256SUMS

Extract into a new, empty directory:
  python -m zipfile -e paper_snapshots/2026-11-19/paper_figures.zip /tmp/mem_param_paper_snapshot_2026-11-19

Expected result: a paper_figures/ directory containing the preserved files.
Keep new experiment results under outputs/. Do not replace this snapshot.
See docs/local_plotting_handoff.md for the rebuttal branch and cluster setup.
