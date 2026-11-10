"""One additional attribution run; preserves the main comparison measurements."""
import json
import os
from pathlib import Path
import time
from navigation_quant_timing_runtime import QuantTimingModel


def log_event(name,start,end,extra=None):
    path=Path(os.environ['PROJECT2_STAGE2_RUN_DIR'])/f'attribution_{os.getpid()}.jsonl'
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=name,start=start,end=end,wall_ms=(end-start)*1000,extra=extra))+'\n')


class AttributionModel(QuantTimingModel):
    def prepare_inputs_for_generation(self,*args,**kwargs):
        start=time.perf_counter()
        result=super().prepare_inputs_for_generation(*args,**kwargs)
        if result.get('pixel_values') is not None:
            log_event('worker_prepare_inputs',start,time.perf_counter(),dict(pixel_shape=list(result['pixel_values'].shape)))
        return result

    def forward(self,*args,**kwargs):
        start=time.perf_counter()
        result=super().forward(*args,**kwargs)
        if kwargs.get('pixel_values') is not None:
            log_event('worker_model_dispatch',start,time.perf_counter(),dict(request_index=self._qt_index))
        return result


def instrument_frontend():
    import torch
    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
    from lmdeploy.vl.engine import ImageEncoder
    for cls,name,label in [(VLAsyncEngine,'_get_prompt_input','frontend_prompt'),
            (ImageEncoder,'preprocess','frontend_preprocess'),(ImageEncoder,'wrap_for_pytorch','frontend_wrap')]:
        original=getattr(cls,name)
        async def wrapper(self,*args,_original=original,_label=label,**kwargs):
            start=time.perf_counter()
            result=await _original(self,*args,**kwargs)
            log_event(_label,start,time.perf_counter(),dict(cpu_threads=torch.get_num_threads()))
            return result
        setattr(cls,name,wrapper)
