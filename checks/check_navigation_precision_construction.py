"""Construct full model on meta device, checking module wiring without loading weights."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weight-bits',type=int,required=True)
    parser.add_argument('--activation-bits',type=int,required=True)
    args=parser.parse_args()
    run=ROOT/f'logs/navigation_precision_meta_w{args.weight_bits}a{args.activation_bits}_20261110'
    run.mkdir(exist_ok=False)
    os.environ.update(PROJECT2_STAGE2_METHOD='awq',PROJECT2_STAGE2_KEEP_RATIO='1.0',
                      PROJECT2_STAGE2_PROFILE='navigation',PROJECT2_STAGE2_RUN_DIR=str(run),
                      PROJECT2_PRECISION_WBITS=str(args.weight_bits),PROJECT2_PRECISION_ABITS=str(args.activation_bits))
    import torch
    from transformers import AutoConfig
    from lmdeploy.pytorch.model_inputs import StepContextManager
    from navigation_precision_runtime import PrecisionQwen3VLForConditionalGeneration
    config=AutoConfig.from_pretrained(str(ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104'),local_files_only=True)
    config.quantization_config=dict(config.quantization_config,bits=args.weight_bits)
    config.text_config.quantization_config=dict(config.quantization_config)
    model=PrecisionQwen3VLForConditionalGeneration(config,StepContextManager(),dtype=torch.float16,device=torch.device('meta'))
    assert all(p.device.type=='meta' for p in model.parameters())
    assert len(model._awq_objects)==144 and len(model._awq_expected_weights)==756
    for module in model._awq_objects.values():
        module.update_weights()
    report=dict(status='passed',weight_bits=args.weight_bits,activation_bits=args.activation_bits,
                projection_modules=144,expected_packed_tensors=756,all_parameters_meta=True,
                scope='Construction and backend update_weights wiring only; no weights loaded, no model inference')
    (run/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
