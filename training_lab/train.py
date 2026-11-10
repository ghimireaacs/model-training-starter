"""Direct Qwen3 QLoRA with full checkpoints. Run through lab.py."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


DEFAULTS = dict(model='Qwen/Qwen3-4B', revision='main', train='examples/train.jsonl',
                valid='examples/valid.jsonl', output='runs/4b-smoke', length=512,
                steps=20, save_every=5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value),
                            help=f'Fresh-run setting (default: {value})')
    parser.add_argument('--resume', help='Full checkpoint directory; restores the saved run settings')
    parser.add_argument('--pause-after', type=int, help='Pause at this absolute optimizer step')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args(argv)
    old = None
    if args.resume:
        if args.check_data or any(getattr(args, key) is not None for key in DEFAULTS):
            parser.error('Use --resume CHECKPOINT with only optional --pause-after; saved settings are restored')
        checkpoint = Path(args.resume).resolve()
        for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                     'adapter_config.json', 'adapter_model.safetensors'):
            if not (checkpoint / name).is_file():
                parser.error(f'Incomplete checkpoint: missing {name}')
        old = json.loads((checkpoint.parent / 'run.json').read_text())
        for key in DEFAULTS:
            setattr(args, key, old['settings'][key])
        args.output = str(checkpoint.parent)
        args.revision = old['revision']
        args.resume = str(checkpoint)
        completed = json.loads((checkpoint / 'trainer_state.json').read_text())['global_step']
        if completed >= args.steps:
            parser.error('This run is complete. Start a new experiment for a new training budget')
        if args.pause_after is not None and args.pause_after <= completed:
            parser.error('--pause-after must be later than the saved step')
    else:
        for key, value in DEFAULTS.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if min(args.length, args.steps, args.save_every) < 1:
        parser.error('Length, steps and save interval must be positive')
    if args.pause_after is not None and not 0 < args.pause_after < args.steps:
        parser.error('--pause-after must be between step 1 and the final step minus one')
    return args, old


def read_rows(filename, tokenizer, length):
    rows = []
    for line_number, line in enumerate(Path(filename).read_text().splitlines(), 1):
        if not line.strip():
            continue
        messages = json.loads(line)['messages']
        roles = [m['role'] for m in messages]
        if roles not in (['user', 'assistant'], ['system', 'user', 'assistant']):
            raise ValueError(f'{filename}:{line_number}: use one user/assistant turn, optionally preceded by system')
        answer = messages[-1]['content']
        if not answer.strip():
            raise ValueError(f'{filename}:{line_number}: empty answer')
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False,
                                               add_generation_prompt=True, enable_thinking=False)
        target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = prefix + target
        if len(ids) > length:
            raise ValueError(f'{filename}:{line_number}: {len(ids)} tokens exceeds {length}; edit the example or increase --length for a fresh run')
        pad = length - len(ids)
        rows.append({'input_ids': ids + [tokenizer.pad_token_id] * pad,
                     'attention_mask': [1] * len(ids) + [0] * pad,
                     'labels': [-100] * len(prefix) + target + [-100] * pad})
    if not rows:
        raise ValueError(f'{filename}: no examples')
    return rows


def pause_requested(step, total, pause_after, marker):
    return step < total and ((pause_after is not None and step >= pause_after) or marker.exists())


def main(argv=None):
    args, old = parse_args(argv)
    output = Path(args.output).resolve()
    if not args.check_data and not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty; choose a new --output or resume a checkpoint')
    hashes = {key: hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest()
              for key in ('train', 'valid')}
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'peft', 'accelerate', 'datasets')}
    try:
        versions['bitsandbytes'] = importlib.metadata.version('bitsandbytes')
    except importlib.metadata.PackageNotFoundError:
        versions['bitsandbytes'] = 'not-installed'
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if old and (old['dataset_sha256'] != hashes or old['versions'] != versions
                or old['script_sha256'] != script_hash):
        raise ValueError('Resume requires unchanged datasets, training script and library versions')

    import torch
    if not args.check_data and not torch.cuda.is_available():
        raise RuntimeError('CUDA training is unavailable. Use python lab.py self-test on CPU, or bootstrap with --backend cuda on a supported machine.')

    from datasets import Dataset
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    revision = args.revision if old else model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    train = Dataset.from_list(read_rows(args.train, tokenizer, args.length))
    valid = Dataset.from_list(read_rows(args.valid, tokenizer, args.length))
    print(f'Base revision: {revision}; train={len(train)}, validation={len(valid)}')
    print('First supervised answer:', tokenizer.decode([x for x in train[0]['labels'] if x != -100]))
    if args.check_data:
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, Trainer,
                              TrainerCallback, TrainingArguments, default_data_collator, set_seed)

    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, device_map={'': 0}, dtype=dtype,
        attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'], bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.print_trainable_parameters()
    output.mkdir(parents=True, exist_ok=True)
    if not old:
        settings = {key: getattr(args, key) for key in DEFAULTS}
        manifest = dict(model=args.model, revision=revision, settings=settings,
                        dataset_sha256=hashes, versions=versions, script_sha256=script_hash,
                        adapter=dict(rank=8, alpha=16, targets=['q_proj', 'v_proj']),
                        microbatch=1, accumulation=8, learning_rate=1e-4, seed=42)
        (output / 'run.json').write_text(json.dumps(manifest, indent=2))

    marker = output / 'PAUSE'

    class SaveAndPause(TrainerCallback):
        paused = False

        def on_step_end(self, training_args, state, control, **kwargs):
            self.paused = pause_requested(state.global_step, args.steps, args.pause_after, marker)
            if self.paused or state.global_step >= args.steps:
                control.should_save = True
            if self.paused:
                control.should_training_stop = True
                control.should_evaluate = False
            return control

    callback = SaveAndPause()
    trainer = Trainer(model=model, train_dataset=train, eval_dataset=valid,
        processing_class=tokenizer, data_collator=default_data_collator, callbacks=[callback],
        args=TrainingArguments(output_dir=str(output), per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            gradient_checkpointing=True, learning_rate=1e-4, max_steps=args.steps,
            warmup_steps=math.ceil(args.steps * 0.05), logging_steps=1, save_strategy='steps', save_steps=args.save_every,
            save_total_limit=2, save_only_model=False, eval_strategy='steps', eval_steps=args.save_every,
            bf16=bf16, fp16=not bf16, optim='adamw_torch', report_to='none', seed=42))
    trainer.train(resume_from_checkpoint=args.resume)
    if callback.paused:
        checkpoint = output / f'checkpoint-{trainer.state.global_step}'
        for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            if not (checkpoint / name).is_file():
                raise RuntimeError(f'Pause checkpoint is incomplete: {checkpoint / name}')
        marker.unlink(missing_ok=True)
        print(f'PAUSED. Checkpoint saved at {checkpoint}. Training process will now exit.')
    else:
        trainer.evaluate()
        trainer.save_model(str(output / 'adapter'))
        tokenizer.save_pretrained(output / 'adapter')
        marker.unlink(missing_ok=True)
        print(f'COMPLETE. Adapter saved at {output / "adapter"}')
    print('Peak allocated GPU GiB:', torch.cuda.max_memory_allocated() / 1024**3)


if __name__ == '__main__':
    main()
