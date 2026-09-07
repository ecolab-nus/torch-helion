# Documentation

Function and structure manual for torch-helion.

- [Atlas](atlas.html) — interactive visualisation of the framework with the
  real `llama_block` run: OpGraph explorer, OpIR kernel flow, generated
  kernels, cost model vs Loom (open in a browser).
- [Architecture](architecture.md) — pipeline stages, source layout, the
  Llama-block kernel decomposition.
- [OpGraph and capture](opgraph.md) — `torch.export` capture, the OpGraph
  data model and op vocabulary.
- [Optimizer](optimizer.md) — canonicalisation passes, partitioning,
  legality, planner, OpIR.
- [Cost models](cost_models.md) — hardware spec reader, analytic estimator,
  tile solver, Loom backend.
- [Codegen](codegen.md) — kernel templates and the empirically verified
  Loom kernel contract.
- [Results layout](results.md) — what `results/<run>/` contains.
- [Entry points](../cmd/README.md) — the three ways to run the stack.
- [Development environment](environment.md) — the Loom container image,
  repository layout, `install-docker.sh`, and how the shared Python
  environment is assembled.
