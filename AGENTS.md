# Working on this starter

Read README.md first. Keep the command path short: bootstrap, self-test, train, pause, resume, generate.

This repository is the source of truth. Make changes in this checkout; generated ZIPs and extracted copies are distribution artifacts, not separate development sources.

- Keep bootstrap.py standard-library-only and usable without pip or venv installed.
- Preserve full optimizer/scheduler/RNG checkpoints and the resume compatibility checks.
- Keep the Qwen3 non-thinking template and answer-only supervision consistent.
- Support configuration through JSON; do not introduce a web service or training framework wrapper.
- Never put credentials, personal data, private hostnames/addresses or machine paths into public examples.
- Keep data/, runs/, models/, exports/, .venv/ and .bootstrap/ excluded from source packages and Git.
- When changing dependencies, regenerate uv.lock and test a clean CPU bootstrap. Do not update an environment holding unfinished runs.
- Run the unit tests and python3 lab.py self-test after changing training or bootstrap logic. CUDA changes still need a real GPU check; report that separately.
- Update README.md and configs when commands change. Keep setup prose concise.
- Keep setup, tests and training as manually invoked local commands. Do not add GitHub Actions workflows to this project.

- Keep source in this shared checkout. Install Python, virtual environments, caches, datasets and run outputs under the machine-local runtime selected by training_lab/paths.py. Never put heavy runtime files on the source NAS.
- Never clear global uv or Hugging Face caches for this project's cleanup. Other applications depend on shared caches. Inspect and clean only this project's dedicated runtime paths, preserving datasets and checkpoints unless their deletion is explicitly requested.
