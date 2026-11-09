"""D-group identities; numerical/input limits deliberately match the C profile."""

from navigation_dart_profiles import profile_config

AWQ_SPEC = {'quant_method': 'awq', 'version': 'gemm', 'bits': 4,
            'group_size': 128, 'zero_point': True}
MODEL_RELATIVE = 'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104'


def served_model_name(keep_ratio, profile='navigation'):
    if keep_ratio not in (1.0, 0.75):
        raise ValueError('Only keep100 control and keep75 combination are supported')
    profile_config(profile)
    keep = 100 if keep_ratio == 1.0 else 75
    return f'SimpleVLN-Qwen3VL-4B-RL-AWQ-DART-keep{keep}-{profile}-20261108'
