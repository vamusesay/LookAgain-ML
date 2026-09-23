# Local inputs

Set `LOOKAGAIN_PRIVATE_INPUT_ROOT` to an authorized input directory and `MULTI_ENCODERS_ROOT` to the authorized research layout. Paths are resolved by the configuration loader. Never commit these directories.

Required manifest fields: `observation_id`, `image_path`, `outcome`, `group_id`, `split`, `image_sha256`, `reference_row`. One row per image; split is `train`, `validation`, or `test`. Reference features require a provenance mapping and matching row/hash digests.
