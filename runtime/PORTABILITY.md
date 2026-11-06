# Portability changes

Frozen round1 and round2 implementations remain separate Python namespaces. Only modules transitively required by the existing reference entry, single-case ACT entry and station loader are included. Old batch-generation/remote orchestration entry points are not exposed. The original full package and deployment archives remain internal.

Paths are resolved relative to installed resources or an explicit material root. The confirmed sequence loader accepts a path relative to its own immutable reference file. The natural-rest binding check resolves relative paths against its storyboard file and still requires the same reviewed file. Asset lookup resolves the bundled asset suffix. No controller parameters or state vectors are rewritten. Source private paths are replaced by public provenance IDs; the original mapping is retained internally. Public-score files are E1 results for the same 52 original ACT exams. No extra experiment is part of the published benchmark.

Original data arrays, video streams, weights, action labels, train/validation memberships and scoring thresholds are unchanged. The directory names are four_tasks_646 and a2_close_coverage_300. Copy-on-write filesystem copies are independent deliverable files, not links to the originals. Metadata fields that describe historical labels or configuration versions are preserved even where their original text is not English.

The original ACT backbone constructor could fetch an ImageNet file before loading the complete saved checkpoint. The package disables only that redundant initialization download; it still loads the complete checkpoint and uses the original pre/post-processing statistics, prediction cadence and temporal aggregation.
