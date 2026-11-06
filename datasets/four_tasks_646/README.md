# A1/A2 Opening and Closing — 646 Demonstrations

646 frozen simulated demonstrations; 521 training / 125 validation. License: CC BY 4.0; attribution: Rusty Valve contributors. The original camera streams, action/state arrays, labels and split memberships are unchanged. File paths and directory names have been made portable.

This repository includes the [manifest](manifest.json) and each `datasets/<task>/episode_map.json`. The complete dataset is stored separately. In that dataset, read a split at `datasets/<task>/train` or `datasets/<task>/validation` using LeRobot 0.6.1 with the PyAV backend. Three cameras are stored at 256×256; controls are 100 Hz and actual capture is 20 Hz.

See the [dataset card](../../docs/dataset_card.md#download-license-and-citation) for download status and the [runtime guide](../../runtime/README.md#data) for the full format. Historical source identifiers refer to the [source index](../../provenance/recorded_sources.json).
