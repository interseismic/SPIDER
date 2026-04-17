# Troubleshooting

## `torch.compile` warning on Python 3.13+

If you see a warning that Dynamo/`torch.compile` is unsupported on Python 3.13+, either:

- disable compile in config (`inference.runtime.torch.compile_eikonet=false`), or
- run under a supported Python version.

## Shared-event PCG fallback warning

If logs show frequent fallback or "all groups fell back to diagonal":

- inspect fallback reason counters (`rows_cap`, `nodes_cap`, `tau_zero`),
- increase `shared_event_re.limits.max_rows` and/or `max_nodes`,
- verify `shared_event_re.model.tau_s` is positive,
- enable `fallback.abort_on_pcg_fallback=true` during debugging.

See {doc}`pcg-whitening-convergence` for the full checklist.

## Sampler instability at higher LR

If trajectories explode when freezing preconditioner:

- ensure adaptation phase is long enough before freeze,
- raise `inference.sampler.eps`,
- use gradient clipping (`inference.sampler.grad_clip_norm`),
- lower LR and retune gradually.
