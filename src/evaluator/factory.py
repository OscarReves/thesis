from .evaluators import GPT2Evaluator, NousHermesMistralEvaluator, NousHermesMistralBinary

EVALUATOR_REGISTRY = {
    "tiny-gpt2": GPT2Evaluator,
    "nous-hermes-mistral": NousHermesMistralEvaluator,
    "nous-hermes-mistral-binary": NousHermesMistralBinary
}

def get_evaluator(name, **kwargs):
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator: {name}")
    return EVALUATOR_REGISTRY[name](**kwargs)