"""Load or merge a completed adapter using its recorded base revision."""
import json
from pathlib import Path


def manifest(run):
    run = Path(run)
    if not (run / 'adapter/adapter_model.safetensors').is_file():
        raise ValueError('Complete the run before generating from or merging its final adapter')
    return run, json.loads((run / 'run.json').read_text())


def generate(run, prompt, base_only=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    run, record = manifest(run)
    if not torch.cuda.is_available():
        raise RuntimeError('This generation command uses CUDA and a four-bit base')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(record['model'], revision=record['revision'])
    model = AutoModelForCausalLM.from_pretrained(record['model'], revision=record['revision'],
        device_map={'': 0}, dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    if not base_only:
        model = PeftModel.from_pretrained(model, run / 'adapter')
    model.eval()
    inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors='pt', return_dict=True).to('cuda')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))


def merge(run, output):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    run, record = manifest(run)
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new export directory')
    print('Merging in CPU FP32. Raw weights alone: about 16 GB for 4B or 32 GB for 8B; working RAM must be higher.')
    base = AutoModelForCausalLM.from_pretrained(record['model'], revision=record['revision'],
        dtype=torch.float32, device_map={'': 'cpu'}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, run / 'adapter').merge_and_unload(safe_merge=True)
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(record['model'], revision=record['revision']).save_pretrained(output)
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Merged checkpoint: {output}. This is not yet a GGUF file.')
