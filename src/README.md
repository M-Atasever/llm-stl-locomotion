# Source layout

Model-specific pipeline source is kept under:

- `GPT/gait_agnostic/`
- `GPT/multi_gait/`
- `Qwen/gait_agnostic/`
- `Qwen/multi_gait/`

The four Python modules directly in this directory reserve the shared-file locations from the repository template. They remain explicit `PLACEHOLDER` modules because the available artifacts do not establish one shared Barkour environment, trainer, curriculum trainer, or evaluator that is valid across models and formulations.

See each model's README for its entry points and limitations. In particular, the supplied Qwen multi-gait temporal configuration requires confirmation and is guarded against execution; no replacement setting has been guessed.
