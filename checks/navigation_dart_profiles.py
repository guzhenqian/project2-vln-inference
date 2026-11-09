"""Explicit limits and identities for short and multi-image DART services."""

PROFILES = {
    'short': {'max_prompt_tokens': 1024, 'min_images': 2, 'max_images': 2,
              'session_len': 4096, 'max_new_tokens': 512},
    # One complete prefill is required so selection sees every image and the
    # final text pivots. Reserve the original maximum generation budget.
    'navigation': {'max_prompt_tokens': 3072, 'min_images': 2, 'max_images': 9,
                   'session_len': 4096, 'max_new_tokens': 512},
}


def profile_config(name):
    if name not in PROFILES:
        raise ValueError('Unknown DART profile: ' + str(name))
    return dict(PROFILES[name])


def served_model_name(keep_ratio, profile='short'):
    if keep_ratio not in (1.0, 0.75):
        raise ValueError('Only keep_ratio 1.0 and 0.75 are supported')
    profile_config(profile)
    keep = 100 if keep_ratio == 1.0 else 75
    suffix = 'learning' if profile == 'short' else 'navigation'
    return f'SimpleVLN-Qwen3VL-4B-RL-FP16-DART-keep{keep}-{suffix}-20261108'
