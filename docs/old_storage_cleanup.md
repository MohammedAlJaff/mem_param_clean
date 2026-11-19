# Old storage cleanup

Use `old_storage_cleanup` to audit historical job output before removing large
checkpoint files. The command is dry-run by default. It does not modify the old
repository.

## Retention policy

The audit keeps all configurations, manifests, metrics, figures, logs, training
histories, and reproducibility artifacts. For each completed run, it keeps the
first checkpoint, every tenth saved checkpoint, and the final checkpoint. It
combines checkpoint steps across resumed attempts before it selects files.

The audit also keeps checkpoints that the current documentation names. It keeps
W&B event files unless a run is cancelled or its remote sync is verified.

The scan does not include current clean outputs, historical evaluation tables,
the old dataset directory, or unrelated old source code. It includes the live
source files below `prototyping/intr_amb_experiments`, but excludes that
directory's `outputs` subtree. The audit saves those source files in
`INTR_AMB_SOURCE_SNAPSHOT.tar.gz` before it lists them for deletion. This
archive preserves tracked changes and untracked diagnostic scripts.

## Create a dry-run audit

Run this command from the clean repository root:

```bash
python -m mem_param_src.runners.old_storage_cleanup \
  --old-root ../old_mem_par \
  --clean-root . \
  --output-dir outputs/storage_audits/old_mem_par/TIMESTAMP
```

The command writes:

- `KEEP_MANIFEST.tsv` for files that remain;
- `DELETE_MANIFEST.tsv` for exact deletion candidates;
- `INTR_AMB_SOURCE_SNAPSHOT.tar.gz` for the exact obsolete intrinsic/ambient
  source files selected by the audit.
- `SUMMARY.json` for counts, hashes, hard-link accounting, and expected space.

Review all four files before deletion. The predicted space uses allocated
bytes and counts a hard-linked file only when every link is marked for removal.

## Apply one reviewed group

Applying a manifest is destructive. Use the exact old root as the confirmation
value. Apply one group at a time, then run `nscquota`.

```bash
python -m mem_param_src.runners.old_storage_cleanup \
  --old-root ../old_mem_par \
  --clean-root . \
  --output-dir outputs/storage_audits/old_mem_par/TIMESTAMP \
  --apply \
  --group capacity_checkpoints \
  --confirm-root ../old_mem_par
```

Before deletion, the command verifies the manifest hash and every selected
file's path, size, modification time, inode, and hard-link count. It unlinks
only listed regular files. It never removes directories recursively. Each
group gets separate progress and completion records in the audit directory.
For the `obsolete_intrinsic_ambient_source` group, it also verifies the source
archive hash before it removes any live source file.

The available groups are recorded in `SUMMARY.json`. Do not apply a group while
an active job reads from `old_mem_par`.
