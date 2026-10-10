# Repository closeout gate

Run this only after all authorized cloud experiments and evaluations finish.

## Required structure work

1. Inventory source, tests, configuration, evidence, legacy material, and generated files.
2. Move active code into a coherent package layout with clear training, reward,
   evaluation, cloud, and shared-data modules.
3. Separate legacy/audit evidence from the supported implementation without deleting
   provenance needed to interpret earlier results.
4. Consolidate duplicated configuration and document every supported entry point.
5. Update imports, Docker build context, cloud job paths, tests, and documentation.
6. Add a root project overview containing setup, architecture, reproduction commands,
   experiment results, costs, limitations, and artifact locations.
7. Run the complete relevant test suite and configuration validation after moves.

## Review gate

After restructuring, a separate high-capability review agent must examine:

- repository structure and maintainability;
- PPO/DPO/SFT algorithm correctness;
- data-split and contamination controls;
- cloud reproducibility and budget safety;
- checkpoint and artifact provenance;
- whether reported conclusions match the measured evidence;
- documentation completeness and stale or broken paths.

Resolve actionable findings before closing. The project is complete only when the
experiments, restructure, tests, documentation, and independent review all pass.
