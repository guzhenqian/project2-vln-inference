"""Prepare a fresh, isolated copy of AWQ calibration tensors; no model loading."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy'
SAMPLES = ROOT / 'data/calibration/navigation_awq_v1/samples.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory relative to project root')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        parser.error('Output must be a new directory inside the project')
    if output.exists():
        parser.error('Output already exists; choose a new directory to preserve evidence')

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    samples = json.loads(SAMPLES.read_text(encoding='utf-8'))
    processor = AutoProcessor.from_pretrained(str(SOURCE), local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, sample in enumerate(samples):
        messages = copy.deepcopy(sample['messages'])
        images, image_hashes = [], []
        for message in messages:
            for item in message['content']:
                if item['type'] == 'image':
                    path = ROOT / item.pop('image')
                    image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                    with Image.open(path) as picture:
                        images.append(picture.convert('RGB'))
        if image_hashes != sample['image_sha256']:
            raise ValueError('Calibration image hash mismatch: ' + sample['id'])
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        features = processor(text=[text], images=images, return_tensors='pt')
        length = features['input_ids'].shape[1]
        if length + 512 > 4096:
            raise ValueError('Input exceeds context budget: ' + sample['id'])
        target = output / ('%02d.pt' % index)
        torch.save(dict(features), target)
        records.append(dict(id=sample['id'], path=target.relative_to(ROOT).as_posix(),
                            sequence_length=length, images=len(images),
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print('[%d/%d] %s: images=%d tokens=%d' %
              (index + 1, len(samples), sample['id'], len(images), length), flush=True)
    (output / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (output / 'provenance.json').write_text(json.dumps(dict(
        source_samples=SAMPLES.relative_to(ROOT).as_posix(),
        samples_sha256=hashlib.sha256(SAMPLES.read_bytes()).hexdigest(),
        processor_source=SOURCE.relative_to(ROOT).as_posix(),
        model_loaded=False, quantization_started=False), indent=2), encoding='utf-8')
    print('AWQ_LEARNING_INPUTS_READY samples=%d tokens_min=%d tokens_max=%d output=%s' %
          (len(records), min(r['sequence_length'] for r in records),
           max(r['sequence_length'] for r in records), output), flush=True)


if __name__ == '__main__':
    main()
