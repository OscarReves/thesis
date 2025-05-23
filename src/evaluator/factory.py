from .evaluators import NousHermesMistralEvaluator, NousHermesMistralBinary, Gemma9bBinary, SuzumeLlama3Binary, Yi34BBinary, SnakModelBinary

EVALUATOR_REGISTRY = {
    "nous-hermes-mistral": NousHermesMistralEvaluator,
    "nous-hermes-mistral-binary": NousHermesMistralBinary,
    "gemma-9b-binary"           : Gemma9bBinary,
    "suzume-llama3-binary"      : SuzumeLlama3Binary,
    "yi-34b-binary"             : Yi34BBinary,
    "snakmodel-binary"          : SnakModelBinary
}

def get_evaluator(name, **kwargs):
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator: {name}")
    return EVALUATOR_REGISTRY[name](**kwargs)