# Polyglot

Real-time cross-lingual voice agent (English, Spanish, Hindi, Tagalog) answering
airline disruption questions from an English policy corpus, with a measured
latency ablation and a deterministic replay evaluation harness.

See `SPEC.md` for the full technical specification and `CLAUDE.md` for the
project's working rules. `PROGRESS.md` tracks milestone-by-milestone status.

Results tables and reproduction instructions land here in M8, once there are
real measured numbers to report (see `CLAUDE.md`: no fabricated metrics).

## Development

```
make setup
make lint
make test
```
