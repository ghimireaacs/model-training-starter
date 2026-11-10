"""Exercise the production trainer with a tiny random CPU model and no Hub downloads."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def main():
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM,
                              Trainer, TrainingArguments)
    from . import train
    transformers = importlib.import_module('transformers')
    backend = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2, 'user': 3,
                                  'assistant': 4, 'answer': 5}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                       eos_token='<eos>', unk_token='<unk>')
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"

    def cpu_args(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, fp16=False, dataloader_pin_memory=False, disable_tqdm=True)
        return TrainingArguments(**kwargs)

    def tiny_model(*args, **kwargs):
        return Qwen3ForCausalLM(Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=64, pad_token_id=1, eos_token_id=1))

    def run(arguments, marker=None):
        available = iter([True])
        def load(*args, **kwargs):
            if marker:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            return tiny_model()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('huggingface_hub.model_info', return_value=SimpleNamespace(sha='a' * 40)))
            stack.enter_context(patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer))
            stack.enter_context(patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=load))
            stack.enter_context(patch.object(transformers, 'BitsAndBytesConfig', side_effect=lambda **kwargs: kwargs))
            stack.enter_context(patch.object(transformers, 'TrainingArguments', side_effect=cpu_args))
            stack.enter_context(patch.object(torch.cuda, 'is_available', side_effect=lambda: next(available, False)))
            stack.enter_context(patch.object(torch.cuda, 'is_bf16_supported', return_value=False))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            train.main(arguments)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with tempfile.TemporaryDirectory(prefix='training-selftest-') as directory:
            root = Path(directory)
            data = root / 'train.jsonl'
            valid = root / 'valid.jsonl'
            rows = [{'messages': [{'role': 'user', 'content': f'user {i}'},
                                  {'role': 'assistant', 'content': 'answer'}]} for i in range(16)]
            data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            valid.write_text(json.dumps(rows[0]) + '\n')
            common = ['--train', str(data), '--valid', str(valid), '--length', '16',
                      '--steps', '4', '--save-every', '2']
            resumed, full = root / 'resumed', root / 'full'
            run(common + ['--output', str(resumed), '--pause-after', '1'])
            checkpoint = resumed / 'checkpoint-1'
            assert not (resumed / 'adapter').exists()
            original = data.read_text()
            data.write_text(original + '\n')
            try:
                run(['--resume', str(checkpoint)])
            except ValueError as error:
                assert 'unchanged datasets' in str(error)
            else:
                raise AssertionError('Modified dataset was accepted on resume')
            data.write_text(original)
            run(['--resume', str(checkpoint)])
            run(common + ['--output', str(full)])
            a = load_file(str(resumed / 'adapter/adapter_model.safetensors'))
            b = load_file(str(full / 'adapter/adapter_model.safetensors'))
            assert a.keys() == b.keys()
            assert all(torch.allclose(a[key], b[key], atol=1e-6, rtol=0) for key in a)
            assert json.loads((resumed / 'checkpoint-4/trainer_state.json').read_text())['global_step'] == 4
            marker_run = root / 'marker'
            marker = marker_run / 'PAUSE'
            run(common + ['--output', str(marker_run)], marker=marker)
            assert (marker_run / 'checkpoint-1/optimizer.pt').is_file()
            assert not marker.exists()
            # A completed adapter must also survive a real CPU merge and reload.
            from .artifacts import merge
            base_path = root / 'base'
            torch.manual_seed(42)
            tiny_model().save_pretrained(base_path)
            tokenizer.save_pretrained(base_path)
            record = json.loads((full / 'run.json').read_text())
            record['model'] = str(base_path)
            (full / 'run.json').write_text(json.dumps(record))
            with contextlib.redirect_stdout(io.StringIO()):
                merge(full, root / 'merged')
                reloaded = AutoModelForCausalLM.from_pretrained(root / 'merged')
            assert reloaded.config.model_type == 'qwen3'
    finally:
        torch.set_num_threads(previous_threads)
    print('PASS: full checkpoints, mid-epoch resume, unchanged-data guard, pause marker, matching adapter weights and CPU merge/reload.')
    print('CPU test only. Large-model downloads, CUDA quantization and VRAM fit were not tested here.')


if __name__ == '__main__':
    main()
