"""Recheck evaluator, scene, annotation, calibration and model identities once."""
from pathlib import Path
from datetime import datetime, timezone
from navigation_stage2_common import ROOT, RUN, PLAN, AWQ, GPTQ, read, write, digest


def main():
    output=RUN/'input_integrity.json'
    assert not output.exists(),'Preserve completed integrity check'
    contract=read(ROOT/'setup/navigation-stage2-execution-20261109.json')
    checked={}
    def check(relative,expected):
        if relative in checked:
            assert checked[relative]==expected
            return
        assert digest(ROOT/relative)==expected,relative
        checked[relative]=expected
    for name,sha in contract['reference_files_sha256'].items():
        check(name,sha)
    for item in contract['calibration_inputs']:
        check(item['path'],item['sha256'])
    roster=read(PLAN/'manifest.json')
    assert digest(ROOT/roster['execution_contract'])==roster['execution_contract_sha256']
    for batch in (1,2,3):
        manifest=read(PLAN/f'source_batch_{batch:02d}.json')
        check(manifest['dataset_path'].format(split=manifest['split']),manifest['source_annotation_sha256'])
        for scene in manifest['scene_files']:
            check(scene['path'],scene['sha256'])
    for model in (AWQ,GPTQ):
        audit=read(ROOT/model/'conversion_audit.json')
        check(model+'/model.safetensors',audit['sha256'])
    # FP16 source full hash was rechecked by the completed GPTQ run before calibration.
    quant=read(ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_CT_stage2_20261109/project2_quantization.json')
    assert quant['algorithm']=='GPTQ' and quant['samples']==32 and quant['fallback_events']==[]
    write(output,dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),
        checked_files_sha256=checked,
        fp16_source_full_hash=contract['model']['source_weights_sha256'],
        fp16_hash_evidence='Completed quantize_navigation_gptq_stage2.py checks this full hash before loading the model',
        quantization_metadata_sha256=digest(ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_CT_stage2_20261109/project2_quantization.json')))
    print('STAGE2_INPUTS_VERIFIED',len(checked),flush=True)


if __name__=='__main__':
    main()
