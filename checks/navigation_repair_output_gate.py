"""Check natural smoke output with the pinned author's existing action parser."""
import re
from simplevln_policy import AuthorAgent


def inspect_answer(answer):
    choice = answer['choices'][0]
    text = choice['message']['content']
    receipt = dict(finish_reason=choice['finish_reason'], completion_tokens=answer['usage']['completion_tokens'],
        parser='Pinned AuthorAgent.extract_multi_result; same first-two-action queue rule')
    if choice['finish_reason'] != 'stop':
        return dict(receipt, passed=False, error='truncated_model_output')
    if not isinstance(text, str) or not text.strip():
        return dict(receipt, passed=False, error='empty_model_output')
    parser = AuthorAgent.__new__(AuthorAgent)
    parser.forward_distance, parser.turn_angle = 25, 15
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    try:
        actions = parser.extract_multi_result(cleaned)[:2]
        pending = []
        fallback = False
        for index, numeric in actions:
            if index == 0:
                pending.append(0)
            elif index in (1, 2, 3):
                unit = parser.forward_distance if index == 1 else parser.turn_angle
                pending.extend([index] * max(0, min(3, round(numeric/unit))))
            # This exactly preserves the original random-fallback trigger;
            # acceptance rejects it rather than inventing an action.
            if index is None or not pending:
                fallback = True
                pending.append(0)
        return dict(receipt, passed=bool(pending) and not fallback, parsed_actions=actions,
            queued_actions=pending, error='invalid_action_random_fallback_rejected' if fallback or not pending else None)
    except Exception as error:
        return dict(receipt, passed=False, error='output_parser_exception', exception=repr(error))
