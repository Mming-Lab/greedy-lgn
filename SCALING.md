# Scaling track (reference, off the main track)

The [README](README.md) and [RESULTS.md](RESULTS.md) main track holds everything to a **fixed budget — 500 gates/layer, single network** — because the game there is which *ideas* move the accuracy points. This file collects the **scaling levers**: wider layers (`--gates`) and ensembles of independent nets (`--ensemble`). They spend compute and inference-circuit area, and reliably buy accuracy — but they don't tell you which idea is any good, so they sit outside the arena. This track is parked; it gets revisited only when a fixed-budget winner deserves a one-off scale check.

For reference, where the scaling levers take the same pipeline:

| | plain greedy (start) | best with scaling | how |
|---|---|---|---|
| **digits** | 88.2% | **96.4%** | 2,000 gates + `--skip-input`, ×4 ensemble (majority vote) |
| **MNIST** | 74.3% | **90.9%** | 4,000 gates + `--skip-input`, ×4 ensemble (soft vote) |

End-to-end backprop at equal *training memory* averages 91.5% on digits — the scaled stack is above it, at the cost of more inference area.

Note: the residual/boosting readout (a **fixed-budget** idea, see [RESULTS.md](RESULTS.md#residualboosting-readout-accumulate-the-answer-and-the-depth-decay-vanishes)) reaches **90.9% MNIST as a single 500-gate net**, matching this whole scaling stack at a fraction of the area — and it keeps scaling with width (2,000 gates → 95.4%). So the scaling flagship is no longer the repo's best number; it is kept here as the reference it always was.

## Detailed logs (in RESULTS.md)

- [Memory-matched comparison: equal training memory, greedy wins](RESULTS.md#memory-matched-comparison-equal-training-memory-greedy-wins)
- [Ensemble voting: parallel circuits are the training-memory-free width lever](RESULTS.md#ensemble-voting-parallel-circuits-are-the-training-memory-free-width-lever)
- [MNIST scaling: width × ensembles push past 90%](RESULTS.md#mnist-scaling-width--ensembles-push-past-90)

Full run logs: [issue #3](https://github.com/Mming-Lab/greedy-lgn/issues/3) (memory-matched), [#8](https://github.com/Mming-Lab/greedy-lgn/issues/8) (ensemble), [#9](https://github.com/Mming-Lab/greedy-lgn/issues/9) (MNIST scaling).
