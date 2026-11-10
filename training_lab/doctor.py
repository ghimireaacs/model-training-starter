"""Check actual computation, not just whether packages import."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args(argv)
    import torch
    import transformers
    import peft
    import datasets
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'peft', 'accelerate', 'datasets')}
    print('Packages:', json.dumps(versions))
    if args.backend == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Check nvidia-smi and the host NVIDIA driver; WSL uses the Windows driver.')
        import bitsandbytes as bnb
        device = torch.device('cuda')
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        layer = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=dtype, quant_type='nf4').to(device)
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        x = torch.randn(2, 32, device=device, dtype=dtype, requires_grad=True)
        layer(x).float().sum().backward()
        if not torch.isfinite(x.grad).all():
            raise RuntimeError('Four-bit kernel returned nonfinite gradients')
        print('PASS: CUDA computation and bitsandbytes NF4 backward')
    else:
        x = torch.ones(8, 8)
        assert (x @ x)[0, 0].item() == 8.0
        print('PASS: CPU computation. Large-model CUDA training is not enabled.')
    Path('.bootstrap').mkdir(exist_ok=True)
    Path('.bootstrap/environment.json').write_text(json.dumps(versions, indent=2) + '\n')


if __name__ == '__main__':
    main()
