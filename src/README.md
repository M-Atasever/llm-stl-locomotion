# Source layout

The final GPT pipelines are self-contained under:

- `GPT/gait_agnostic/`
- `GPT/multi_gait/`

The four Python modules directly in this directory reserve the shared-file locations from the repository template. They remain explicit `PLACEHOLDER` modules because the available final artifacts do not establish one shared Barkour environment, trainer, curriculum trainer, or evaluator that is valid for both formulations.

Keeping the two verified implementations separate avoids inventing a merged interface or silently changing either final training configuration.
