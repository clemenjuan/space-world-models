# AGENTS.md

**NO TRASH FILES.** Keep the repo concise: add only necessary source, configs, tests, and canonical docs. Generated data, notebooks, coverage reports, caches, scratch experiments, and one-off artifacts stay ignored, removed after use or outside the repo.


## Operational Safety Guardrails

- Do not run SSH, SCP, rsync, remote-login, or host-reachability probes from this repo. Do not edit `~/.ssh/config` or retry SSH failures unless the user explicitly asks for that exact action.
- Do not start detached/background work (`tmux new-session -d`, `nohup`, trailing `&`, cron-like loops, or `while true`) unless the user explicitly approves the exact command, stop condition, log path, and kill/cleanup command.
- Long training, evaluation, live LLM, W&B, Ollama, or other networked runs require explicit approval with expected runtime, endpoint/network use, max concurrency, and output location. Default verification is local, bounded, and offline.
- Do not run `scripts/refresh_board.py` or similar rebuild commands in a loop. Use a one-shot rebuild only, then stop.
- If SSH, sandboxing, or local command execution starts failing, do not retry in a loop. Gather one local-only diagnostic pass, report the concrete error, and stop for user direction.
- Before experiment/server work, inspect current local processes once with a bounded command such as `pgrep -af 'uv|autops|python|tmux|nohup|ssh|scp|rsync'`. Do not kill or restart processes without explicit approval.
