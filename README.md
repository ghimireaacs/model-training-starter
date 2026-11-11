# Model Training Starter

Go from a folder of Python files to a local Qwen3 fine-tuning environment. Train a small adapter, save a full checkpoint, stop the process, and resume later.

Direct **Transformers + PEFT + bitsandbytes**. No hosted training service or Unsloth.

## Start here

You need **Git, Python 3.10+**, internet access and disk space. For 4B/8B training, install a working NVIDIA driver first. On Windows, Ubuntu WSL2 is the recommended route; the driver belongs on Windows.

On a fresh machine:

```bash
git clone https://github.com/ghimireaacs/model-training-starter.git
cd model-training-starter
python3 bootstrap.py
```

Already using the shared source checkout? Open that folder and run only `python3 bootstrap.py`. No second clone or ZIP extraction is needed.

Installation is complete when bootstrap prints `READY: CUDA training environment`. It installs the software on local disk and checks computation. Model downloads and training start only with the separate commands below:

```bash
python3 lab.py self-test
python3 lab.py train configs/4b-smoke.json --pause-after 10
python3 lab.py resume runs/4b-smoke/checkpoint-10
```

Commands below use `python3` for Linux/WSL. On native Windows, use `python`. No environment activation is needed.

| Command | Result |
|---|---|
| `bootstrap.py` | Installs uv, Python 3.12 and locked dependencies on local disk; checks computation |
| `self-test` | Proves pause/resume using a tiny random CPU model; no model download |
| `train … --pause-after 10` | Downloads the official 4B base, trains to step 10, saves and exits |
| `resume …/checkpoint-10` | Restores that experiment and finishes its original 20-step budget |

The 16 training and 4 validation examples are synthetic plumbing tests. The resulting adapter is not a useful specialist. Bootstrap does not download model weights or start training.

## Shared code, local storage

Keep this repository on the shared drive if you use one. Run commands from this same checkout; the scripts automatically keep the heavy files on the current machine.

| Location | Contents |
|---|---|
| This repository | Source code, README, JSON configs and dependency lock |
| `~/.local/share/model-training-starter/` on Linux/WSL | Python, `.venv`, caches, datasets, checkpoints and exports |
| `%LOCALAPPDATA%/model-training-starter/` on Windows | The same machine-local runtime files |

```bash
python3 lab.py paths
```

Config paths such as `data/specialist/train.jsonl` and CLI paths such as `runs/4b-smoke` are relative to that **local runtime directory**. Config filenames such as `configs/4b-smoke.json` remain relative to the source repository. The synthetic examples are copied locally during setup; existing copies are preserved.

For a different local disk, set `MODEL_TRAINING_HOME` to an absolute directory and keep the same setting for every command. On Linux/WSL:

```bash
export MODEL_TRAINING_HOME="$HOME/ai-training-local"
python3 bootstrap.py
```

Use a Linux filesystem on WSL local disk, rather than a NAS mount or `/mnt/c`. Existing environments and runs inside an older checkout are not moved or deleted automatically. Keep unfinished runs intact until their data and checkpoints have been deliberately transferred.

Keep global uv and Hugging Face caches intact; other applications may use them. This starter has its own caches under the local runtime directory. Any cleanup should target only confirmed project files there.

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
python3 bootstrap.py --backend cpu
python3 lab.py doctor
```

Rerunning bootstrap checks the existing environment without updating it. Set `MODEL_TRAINING_HOME` to a new local directory to change the backend or dependency lock while another run is unfinished. CPU mode is for code checks and the tiny self-test, not useful 4B/8B fine-tuning.

## Pause and resume

In a second terminal, from this folder:

```bash
python3 lab.py pause runs/4b-smoke
```

Wait for **PAUSED** and the training terminal's prompt. Then you can sleep, reboot or use the GPU elsewhere. Resume the checkpoint named in that output:

```bash
python3 lab.py resume runs/4b-smoke/checkpoint-10
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
python3 lab.py generate runs/4b-smoke "Summarize a backup failure caused by a full disk." --base-only
python3 lab.py generate runs/4b-smoke "Summarize a backup failure caused by a full disk."
```

Run sequentially. Each process releases its GPU allocations when it exits. Use held-out prompts to judge a real experiment.

## Train your own task

1. Complete the 4B test.
2. Run `python3 lab.py train configs/8b-pilot.json` to test the larger model's memory use.
3. Put reviewed examples in the local runtime’s `data/specialist/train.jsonl` and `data/specialist/valid.jsonl`. Keep an untouched test set separately.
4. Edit `configs/8b-specialist.json`: set model, data paths, a new output folder, sequence length and total steps.
5. Validate, then train:

```bash
python3 lab.py train configs/8b-specialist.json --check-data
python3 lab.py train configs/8b-specialist.json --pause-after 10
python3 lab.py resume runs/8b-specialist-v1/checkpoint-10
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
| `training_lab/paths.py` | Shared source and machine-local runtime paths |
| `training_lab/train.py` | Tokenization, loss masking, QLoRA, checkpoints and resume |
| `training_lab/artifacts.py` | Base/adapter generation and CPU merge |
| `examples/` | Public synthetic fixtures; safe to publish |
| `tests/` | Bootstrap, data and resume-policy checks |
| `pyproject.toml`, `uv.lock` | Pinned direct dependencies and the resolved environment |
| `data/`, `runs/`, `cache/`, `exports/` | Under the local runtime directory, outside the repository; downloaded models live in `cache/huggingface/` |

A completed adapter can be merged with `python3 lab.py merge runs/8b-specialist-v1 exports/specialist-merged`. CPU FP32 raw weights alone require about 16 GB for 4B or 32 GB for 8B, plus working memory. The output is a Hugging Face checkpoint; GGUF conversion is a separate step.

## If something fails

| Problem | Action |
|---|---|
| CUDA unavailable | Check `nvidia-smi`; in WSL fix the Windows driver |
| Bootstrap interrupted | Rerun the same command; it can reuse downloaded packages |
| Output folder already exists | Resume its checkpoint or choose a fresh output folder |
| Resume rejects changed inputs | Restore the original code/data/environment; do not bypass the check |
| Model download interrupted | Rerun the same fresh-run command if no run manifest was written |
| Wrong or weak answers | Compare the base and adapter on unseen prompts; improve data before adding steps |

## Check your changes

After editing the code, run both checks. Neither needs a GPU or a model download.

```bash
python3 -m unittest discover -s tests -v
python3 lab.py self-test
```

The unit tests cover bootstrap decisions, data masking and the resume rules. The self-test runs the real Trainer and PEFT checkpoint workflow on a tiny random model: train, pause, resume, compare with an uninterrupted run, merge. It proves the logic, not CUDA performance or whether a particular 8B configuration fits your card.

Editing `training_lab/train.py` changes its hash, and resume checks that hash. Finish or abandon unfinished runs before you change it.

`python3 tools/package.py` writes a source-only ZIP to `dist/` from an explicit file list, for copying the starter to a machine without Git.

## License and references

Code and synthetic examples: [MIT](LICENSE). Downloaded models and your datasets retain their own licenses.

[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [uv installation](https://docs.astral.sh/uv/getting-started/installation/) · [PyTorch wheels](https://pytorch.org/get-started/previous-versions/) · [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
