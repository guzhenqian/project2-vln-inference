"""CPU control-flow checks for prune-layer 2/3, DeepStack and decoded KV lengths."""
import copy
import importlib
import os
import sys
import types
import unittest
from unittest import mock
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import check_navigation_dart_runtime_unit as base


class RuntimeChecks(unittest.TestCase):
    prefill=base.DartRuntimeChecks.prefill
    decode=base.DartRuntimeChecks.decode

    @classmethod
    def setUpClass(cls):
        cls.fake=mock.patch.dict(sys.modules,base.fake_lmdeploy_modules())
        cls.fake.start()
        cls.runtime=importlib.import_module('navigation_stage2_runtime')

    @classmethod
    def tearDownClass(cls):
        cls.fake.stop()
        sys.modules.pop('navigation_stage2_runtime',None)

    def setUp(self):
        self.context=None
        self.manager=types.SimpleNamespace(current_context=lambda:self.context)
        self.config=types.SimpleNamespace(text_config=types.SimpleNamespace(),
            vision_config=types.SimpleNamespace(deepstack_visual_indexes=[5,11,17],spatial_merge_size=2),
            image_token_id=999998,video_token_id=999999)
        self.patches=[mock.patch.object(self.runtime,'version',return_value='0.11.1'),
            mock.patch.object(self.runtime.Stage2Qwen3VLForConditionalGeneration,'_write_json',
                new=base.DartRuntimeChecks.record_event)]
        for patch in self.patches: patch.start()

    def tearDown(self):
        for patch in reversed(self.patches): patch.stop()

    def model(self,method,keep):
        with mock.patch.dict(os.environ,{'PROJECT2_STAGE2_METHOD':method,
            'PROJECT2_STAGE2_KEEP_RATIO':str(keep),'PROJECT2_STAGE2_RUN_DIR':str(base.ROOT/'logs'),
            'PROJECT2_STAGE2_PROFILE':'short'}):
            return self.runtime.Stage2Qwen3VLForConditionalGeneration(self.config,self.manager,dtype=torch.float16,device='cpu')

    def test_both_recipes_and_budgets_keep_deepstack_rope_and_kv(self):
        for method,layer in [('fastv',2),('sparsevlm',3)]:
            for keep_ratio in (.75,.5):
                with self.subTest(method=method,keep=keep_ratio):
                    model=self.model(method,keep_ratio)
                    request=base.make_request()
                    result=self.prefill(model,request)
                    event=next(data for _,data in model.test_events if data['event']=='image_prefill')
                    kept=torch.tensor(event['kept_original_positions'])
                    shortened=len(kept)
                    removed=request['length']-shortened
                    self.assertEqual(removed,int(16*(1-keep_ratio)))
                    self.assertEqual([item.calls[0]['q'] for item in model.language_model.layers],
                        [request['length']]*layer+[shortened]*(base.LAYERS-layer))
                    low=model.language_model.layers[0].calls[0]
                    high=model.language_model.layers[layer].calls[0]
                    torch.testing.assert_close(high['cos'],low['cos'][kept],rtol=0,atol=0)
                    torch.testing.assert_close(result[:,kept],model.language_model.norm.last_normalized,rtol=0,atol=0)
                    self.assertEqual([len(call['features']) for call in model.language_model.deepstack_calls],
                        [16,16,16-removed] if layer==2 else [16,16,16])
                    prefix=copy.deepcopy(request['cache'])
                    with mock.patch.object(self.runtime,'select_attention_tokens',side_effect=AssertionError('Decode must not select')):
                        for appended in (1,2):
                            self.decode(model,request,appended)
                            full=request['length']+appended
                            self.assertEqual([item.calls[-1]['kv'] for item in model.language_model.layers],
                                [full]*layer+[full-removed]*(base.LAYERS-layer))
                    for index,tensors in enumerate(request['cache']):
                        valid=request['length'] if index<layer else shortened
                        for before,after in zip(prefix[index],tensors):
                            torch.testing.assert_close(before.reshape(-1,base.WIDTH)[:valid],after.reshape(-1,base.WIDTH)[:valid],rtol=0,atol=0)

    def test_full_keep_uses_unchanged_native_forward(self):
        for method in ('fp16','fastv','sparsevlm'):
            model=self.model(method,1.)
            request=base.make_request()
            with mock.patch.object(self.runtime,'select_attention_tokens',side_effect=AssertionError('Control selected')):
                actual=self.prefill(model,request)
            native=base.FakeNativeOuter(self.config,self.manager)
            ref=base.make_request()
            native.update_model_metas(ref['cache'],context=ref['context'])
            expected=native(ref['ids'],ref['positions'],ref['cache'],attn_metadata=ref['context'].attn_metadata,
                pixel_values=ref['pixel'],image_mask=ref['mask'],mrope_position_ids=ref['context'].mrope_position_ids)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
