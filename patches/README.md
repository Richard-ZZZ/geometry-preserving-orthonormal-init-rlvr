# Runtime Patches

This directory contains the `verl` runtime patch required by the DAPO 1.5B reproduction scripts.

## Patch File

- `verl_rollout_adapter_sync.patch`: adapter synchronization, vLLM rollout path resolution, and SVD-initialized adapter reference/KL handling.

## How To Apply

From a clean `verl` checkout at the expected base commit:

```bash
git apply /path/to/geometry-preserving-orthonormal-init-rlvr/patches/verl_rollout_adapter_sync.patch
```

Then install `verl` in editable mode according to `INSTALL.md`.

## Base Commit Used When Exporting

```text
f332fc814718b9ea7968f6d264211460d4e90fff
```

If the patch does not apply cleanly, use the file list in `docs/verl_vllm_rollout_notes.md` to port the changes manually.
