# Model Training Starter

Go from a folder of Python files to a local Qwen3 fine-tuning environment. Train a small adapter, save a full checkpoint, stop the process, and resume later.

Direct **Transformers + PEFT + bitsandbytes**. No hosted training service or Unsloth.

## Start here

You need **Python 3.10+**, internet access and disk space. For 4B/8B training, install a working NVIDIA driver first. On Windows, Ubuntu WSL2 is the recommended route; the driver belongs on Windows.

From this folder:

```bash
python bootstrap.py
python lab.py self-test
python lab.py train configs/4b-smoke.json --pause-after 10
python lab.py resume runs/4b-smoke/checkpoint-10
```

Use `python3` if your machine has no `python` command. No environment activation is needed.

| Command | Result |
|---|---|
| `bootstrap.py` | Installs project-local uv, managed Python 3.12 and locked dependencies into `.venv`; checks computation |
| `self-test` | Proves pause/resume using a tiny random CPU model; no model download |
| `train … --pause-after 10` | Downloads the official 4B base, trains to step 10, saves and exits |
| `resume …/checkpoint-10` | Restores that experiment and finishes its original 20-step budget |

The 16 training and 4 validation examples are synthetic plumbing tests. The resulting adapter is not a useful specialist. Bootstrap does not download model weights or start training.

## Supported machines

| Machine | Bootstrap result |
|---|---|
| x86-64 Linux with glibc 2.28+, including Ubuntu WSL2 | CUDA training when a compatible NVIDIA driver is visible; otherwise CPU self-test |
| Native Windows x86-64 | Same CUDA/CPU selection; WSL2 is the recommended training route |
| Linux ARM64 with glibc 2.28+ or Apple Silicon macOS | CPU self-test with `--backend cpu`; this starter does not supply an ARM CUDA/MPS training recipe |
| Alpine/musl, Intel macOS, 32-bit systems | Not supported by this pinned environment |

The CUDA environment uses **PyTorch 2.8 / CUDA 12.6**. The driver must report CUDA 12.6 or newer, and the GPU must be supported by that PyTorch build. Bootstrap tests an NF4 forward/backward pass. It does not install drivers, change system packages, or silently run full-model training on CPU.

For a 3080 12 GB, start with 4B at 512 tokens. Treat 8B as a memory pilot. Allow roughly 40 GB free for the first exercise and more for both models and exports. File size alone does not establish VRAM fit.

```bash
python bootstrap.py --backend cpu
python lab.py doctor
```

Rerunning bootstrap checks the existing environment without updating it. Use a fresh checkout to change the backend or dependency lock while another run is unfinished. CPU mode is for code checks and the tiny self-test, not useful 4B/8B fine-tuning.

## Pause and resume

In a second terminal, from this folder:

```bash
python lab.py pause runs/4b-smoke
```

Wait for **PAUSED** and the training terminal's prompt. Then you can sleep, reboot or use the GPU elsewhere. Resume the checkpoint named in that output:

```bash
python lab.py resume runs/4b-smoke/checkpoint-10
```

`checkpoint-10` is the exact path from the planned first example. An unscheduled pause may produce another step number. Use the path that actually exists.

| Keep | Why |
|---|---|
| Full `checkpoint-N/` | Adapter, optimizer, scheduler, random state and training step |
| `run.json`, unchanged data and training code | Identifies the experiment and validates resume |
| Original environment and base cache | Preserves dependency versions and the pinned model revision |

The last two checkpoints are retained. Ctrl+C or a crash can lose work since the last complete save. `adapter/` is created on completion and is for inference/export; it cannot resume optimizer state. `--pause-after` uses absolute steps. Resume restores the original total, not an additional step budget.

## Compare the result

```bash
python lab.py generate runs/4b-smoke "Summarize a backup failure caused by a full disk." --base-only
python lab.py generate runs/4b-smoke "Summarize a backup failure caused by a full disk."
```

Run sequentially. Each process releases its GPU allocations when it exits. Use held-out prompts to judge a real experiment.

## Train your own task

1. Complete the 4B test.
2. Run `python lab.py train configs/8b-pilot.json` to test the larger model's memory use.
3. Put reviewed examples in `data/specialist/train.jsonl` and `data/specialist/valid.jsonl`. Keep an untouched test set separately.
4. Edit `configs/8b-specialist.json`: set model, data paths, a new output folder, sequence length and total steps.
5. Validate, then train:

```bash
python lab.py train configs/8b-specialist.json --check-data
python lab.py train configs/8b-specialist.json --pause-after 10
python lab.py resume runs/8b-specialist-v1/checkpoint-10
```

Each JSONL line contains one conversation:

```json
{"messages":[{"role":"user","content":"Summarize: backup failed because the destination is full."},{"role":"assistant","content":"The backup failed due to insufficient destination space. Check capacity and retention, rerun it, and verify the result."}]}
```

One user/assistant pair is supported, optionally preceded by a system message. Overlong examples fail validation instead of silently losing the answer. The trainer supervises only the assistant answer, using Qwen3 non-thinking mode.

| Config | Purpose | Total steps |
|---|---|---|
| `4b-smoke.json` | First test with `Qwen/Qwen3-4B` | 20 |
| `8b-pilot.json` | Fit test with `Qwen/Qwen3-8B` | 10 |
| `8b-specialist.json` | Your own reviewed dataset | 125; approximately one pass over 1000 examples at effective batch 8 |

If 8B runs out of memory, close other GPU workloads and try length 256 in a **new** run folder. If it still fails, use `Qwen/Qwen3-4B` for the real task. Change settings before starting a new experiment, not by editing an unfinished run's manifest.

## Files to work on

| Path | Purpose |
|---|---|
| `bootstrap.py` | Standard-library entry point; verifies the uv download against PyPI's SHA-256 |
| `lab.py` | Small command interface that uses the project environment |
| `configs/` | Experiment settings |
| `training_lab/train.py` | Tokenization, loss masking, QLoRA, checkpoints and resume |
| `training_lab/artifacts.py` | Base/adapter generation and CPU merge |
| `examples/` | Public synthetic fixtures; safe to publish |
| `tests/` | Bootstrap, data and resume-policy checks |
| `pyproject.toml`, `uv.lock` | Pinned direct dependencies and the resolved environment |
| `data/`, `runs/`, `models/`, `exports/` | Your local work; ignored by Git |

A completed adapter can be merged with `python lab.py merge runs/8b-specialist-v1 exports/specialist-merged`. CPU FP32 raw weights alone require about 16 GB for 4B or 32 GB for 8B, plus working memory. The output is a Hugging Face checkpoint; GGUF conversion is a separate step.

## If something fails

| Problem | Action |
|---|---|
| CUDA unavailable | Check `nvidia-smi`; in WSL fix the Windows driver |
| Bootstrap interrupted | Rerun the same command; it can reuse downloaded packages |
| Output folder already exists | Resume its checkpoint or choose a fresh output folder |
| Resume rejects changed inputs | Restore the original code/data/environment; do not bypass the check |
| Model download interrupted | Rerun the same fresh-run command if no run manifest was written |
| Wrong or weak answers | Compare the base and adapter on unseen prompts; improve data before adding steps |

## Test and publish

```bash
python -m unittest discover -s tests -v
python lab.py self-test
python tools/package.py
```

`dist/model-training-starter.zip` contains only the explicit source allowlist. Publish this folder as its own repository, not its parent workspace. No private datasets, credentials, weights, checkpoints or host addresses belong in the public repo. `.gitignore` is not a substitute for reviewing your staged files.

The CPU self-test exercises the real Trainer and PEFT checkpoint workflow with tiny random weights. It does not establish CUDA performance or that a particular 8B configuration fits. Run setup, tests and training manually on your own machine. This repository has no GitHub Actions workflow.

## License and references

Code and synthetic examples: [MIT](LICENSE). Downloaded models and your datasets retain their own licenses.

[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [uv installation](https://docs.astral.sh/uv/getting-started/installation/) · [PyTorch wheels](https://pytorch.org/get-started/previous-versions/) · [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
