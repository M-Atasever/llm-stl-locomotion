# Prompts

The prompt sets are separated by formulation and ordered by pipeline stage:

1. symbolic STL specification generation;
2. integration of the specification into the reward/training code;
3. expert-trajectory robustness collection.

`gait_agnostic/` contains the prompts used for the shared-formula pipeline. `multi_gait/` contains the prompts used for the gait-conditioned walk/trot/bound pipeline.

The three numbered files directly inside each formulation are the existing GPT prompt set. The corresponding `Qwen/` subdirectory contains the three supplied Qwen prompt snapshots from `qwen.zip`, using the same numbering. The larger Qwen multi-gait prompts include the original attached code/test inputs; those are generation context, not alternative runtime implementations.

Qwen prompting/re-prompting scripts are not included. The complete archived workflow cannot be replayed from the supplied files: multi-gait runners require missing conversation state and original code inputs, and the combined prompt module also contains Phase 2 material. The gait-agnostic first-prompt runner is standalone, but runners are omitted consistently with the existing GPT layout. The supplied snapshots and generated specifications are retained; exact full conversation/replay provenance is `PLACEHOLDER`. No API requests are required to use the retained source.

The six Qwen snapshots are preserved byte-for-byte, including their original whitespace, so they can be compared directly with the supplied archive.
