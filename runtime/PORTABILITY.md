# Packaging notes

The round 1 and round 2 ACT runs are kept as separate Python namespaces, so each reproduces exactly as published. The package includes only what the reference controller, the ACT inference entry, and the station loader need.

Paths resolve relative to the installed package or to the materials folder you pass in. Controller parameters, state vectors, data arrays, videos, weights, action labels, train and validation splits, and scoring thresholds are unchanged. Some metadata fields keep their original non-English text.

The original ACT backbone constructor downloads an ImageNet file before loading the saved checkpoint. The package skips that download only; it still loads the complete checkpoint with the original pre- and post-processing, prediction cadence, and temporal aggregation.
