"""Read-only checks and execution receipts for native LMDeploy W4A16 modules.

No dequantization, model construction or CUDA operations occur in this module.
Tensor metadata is checked; numeric equivalence still requires real GPU tests.
"""

import torch

from navigation_awq_dart_profiles import AWQ_SPEC

PROJECTIONS = ('self_attn.qkv_proj', 'self_attn.o_proj',
               'mlp.gate_up_proj', 'mlp.down_proj')
SOURCE_PROJECTIONS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                      'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def validate_config(config, dtype):
    if dtype != torch.float16:
        raise ValueError('AWQ+DART requires FP16 activations with packed INT4 weights')
    for label, part in (('root', config), ('text', config.text_config)):
        quant = getattr(part, 'quantization_config', None)
        if not isinstance(quant, dict) or quant != AWQ_SPEC:
            raise ValueError(f'{label} must retain the exact AWQ GEMM W4/group128/asymmetric configuration')
    if getattr(config.vision_config, 'quantization_config', None) is not None:
        raise ValueError('AWQ+DART requires the original unquantized vision encoder')


def expected_source_names(layer_count):
    return {f'language_model.layers.{i}.{projection}.{suffix}'
            for i in range(layer_count) for projection in SOURCE_PROJECTIONS
            for suffix in ('qweight', 'qzeros', 'scales')}


def normalize_source_name(name):
    return name.removeprefix('model.')


def inspect_awq_modules(model):
    """Check actual native module classes and packed tensor layout, never values."""
    from lmdeploy.pytorch.nn.linear.awq import AwqLinear, MergedAwqLinear, QKVAwqLinear

    config = model.config.text_config
    if len(model.language_model.layers) != config.num_hidden_layers:
        raise ValueError('AWQ decoder layer count does not match configuration')
    hidden, intermediate = config.hidden_size, config.intermediate_size
    head_dim = config.head_dim
    q_size, kv_size = config.num_attention_heads * head_dim, config.num_key_value_heads * head_dim
    specs = {
        'self_attn.qkv_proj': (QKVAwqLinear, hidden, q_size + 2 * kv_size),
        'self_attn.o_proj': (AwqLinear, q_size, hidden),
        'mlp.gate_up_proj': (MergedAwqLinear, hidden, intermediate * 2),
        'mlp.down_proj': (AwqLinear, intermediate, hidden),
    }
    entries, objects = [], {}
    for i, layer in enumerate(model.language_model.layers):
        for projection in PROJECTIONS:
            module = layer.get_submodule(projection)
            cls, in_features, out_features = specs[projection]
            name = f'language_model.layers.{i}.{projection}'
            if type(module) is not cls:
                raise ValueError(f'{name}: expected native {cls.__name__}; got {type(module).__name__}')
            if (module.w_bit, module.group_size, module.in_features, module.out_features) != (
                    4, 128, in_features, out_features):
                raise ValueError(f'{name}: AWQ native module dimensions/quantization mismatch (tp=1 required)')
            expected = {'qweight': ((in_features, out_features // 8), torch.int32),
                        'qzeros': ((in_features // 128, out_features // 8), torch.int32),
                        'scales': ((in_features // 128, out_features), torch.float16)}
            for suffix, (shape, dtype) in expected.items():
                tensor = getattr(module, suffix, None)
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(f'{name}.{suffix}: invalid packed shape or dtype')
            if type(module.impl).__name__ != 'AwqLinearW4A16Impl':
                raise ValueError(f'{name}: unsupported AWQ backend implementation')
            entries.append(dict(name=name, layer_index=i, projection=projection,
                                **{'class': type(module).__name__},
                                backend_class=type(module.impl).__name__,
                                in_features=in_features, out_features=out_features,
                                qweight_shape=list(module.qweight.shape), qweight_dtype='int32',
                                qzeros_shape=list(module.qzeros.shape), qzeros_dtype='int32',
                                scales_shape=list(module.scales.shape), scales_dtype='float16'))
            objects[name] = module
    actual_names = {name for name, module in model.named_modules() if isinstance(module, AwqLinear)}
    if actual_names != set(objects):
        raise ValueError('AWQ modules must be exactly the four native language projections per decoder layer')
    # Vision, embeddings and output head remain native FP16 parameters.
    for name, parameter in model.named_parameters():
        if not name.startswith('language_model.layers.') and parameter.is_floating_point():
            if parameter.dtype != torch.float16:
                raise ValueError(f'{name}: non-quantized modules must remain FP16')
    return dict(runtime_module_count=len(entries),
                checkpoint_projection_count=config.num_hidden_layers * 7,
                modules=entries), objects


def execution_receipt(calls, inventory, loaded, expected, lower_rows, upper_rows):
    if loaded != expected:
        raise ValueError(f'Native AWQ loader consumed {len(loaded)}/{len(expected)} required packed tensors')
    if set(calls) != {entry['name'] for entry in inventory['modules']}:
        raise ValueError('Not every native AWQ module executed in this forward')
    rows = []
    for entry in inventory['modules']:
        lengths = calls[entry['name']]
        expected_rows = lower_rows if entry['layer_index'] < 2 else upper_rows
        if lengths != [expected_rows]:
            raise ValueError(f"{entry['name']}: AWQ call rows {lengths} != [{expected_rows}]")
        rows.append(dict(name=entry['name'], layer_index=entry['layer_index'],
                         projection=entry['projection'], rows=lengths[0], calls=1))
    return dict(verified=True, loaded_quantized_tensor_count=len(loaded),
                module_count=len(rows), module_calls=rows)
