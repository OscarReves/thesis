from .evaluators import GPT2Evaluator

EVALUATOR_REGISTRY = {
    "tiny-gpt2": GPT2Evaluator,
}

def get_evaluator(name, **kwargs):
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator: {name}")
    return EVALUATOR_REGISTRY[name](**kwargs)