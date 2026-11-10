# Working on this starter

Read README.md first. Keep the command path short: bootstrap, self-test, train, pause, resume, generate.

- Keep bootstrap.py standard-library-only and usable without pip or venv installed.
- Preserve full optimizer/scheduler/RNG checkpoints and the resume compatibility checks.
- Keep the Qwen3 non-thinking template and answer-only supervision consistent.
- Support configuration through JSON; do not introduce a web service or training framework wrapper.
- Never put credentials, personal data, private hostnames/addresses or machine paths into public examples.
- Keep data/, runs/, models/, exports/, .venv/ and .bootstrap/ excluded from source packages and Git.
- When changing dependencies, regenerate uv.lock and test a clean CPU bootstrap. Do not update an environment holding unfinished runs.
- Run the unit tests and python lab.py self-test after changing training or bootstrap logic. CUDA changes still need a real GPU check; report that separately.
- Update README.md and configs when commands change. Keep setup prose concise.
