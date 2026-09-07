# Results layout

`compile_model()` writes every intermediate and final result to
`results/<run_name>/` (override with `CompileConfig(results_dir=...)` or
`TORCH_HELION_RESULTS`).

| file | stage | content |
| --- | --- | --- |
| `00_model.txt` | capture | model `repr` and the `CompileConfig` |
| `01_fx_graph.txt`, `01_fx_op_histogram.json` | capture | exported core-ATen graph and op counts |
| `02_opgraph_raw.{json,txt}` | capture | raw OpGraph (1:1 with ATen) |
| `03_opgraph_canonical.{json,txt}` | optimizer | canonical loom-style OpGraph |
| `03_pass_log.json` | optimizer | rewrites per pass and numerical drift vs. the raw graph |
| `04_hw_spec.json` | cost model | mesh, memories, registered hardware functions |
| `05_plan/candidates.json`, `05_plan/plan.md` | planner | every partition candidate with legality and cost; the chosen program |
| `06_opir.{json,txt}` | optimizer | the OpIR `Program` (kernel specs, tiles, costs) |
| `07_validation.json` | codegen | Helion frontend / Loom exploration status per kernel |
| `08_reference_check.json` | codegen | OpIR interpreter vs PyTorch model |
| `09_cost_report.{md,json}` | cost model | per-kernel estimate, kernel-to-kernel links |
| `10_loom_results.json` | Loom (optional) | Loom's optimal time and block sizes per kernel |
| `kernels/<k>.py` | codegen | generated Helion kernels (Loom CLI entry points) |
| `kernels/config_files/<k>.json`, `<k>.assigned.json` | codegen | Loom configs (solver / assigned tiles) |
| `kernels/README.md` | codegen | how to run each kernel through Loom |
| `params.pt` | codegen (optional) | transformed weights/tables consumed by the kernels |
| `loom/<k>/` | Loom (optional) | Loom's `IRs/` and `constraints/` outputs; `<k>_assigned/` when run with the planner's tiling |
| `loom/<k>.loom.log` | Loom (optional) | stdout and stderr of that kernel's Loom run |
| `timings.json` | pipeline | wall time per stage |
