from .generators import GPT2Generator

GENERATOR_REGISTRY = {
    "gpt2": GPT2Generator,
}

def get_generator(name, **kwargs):
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown embedder: {name}")
    return GENERATOR_REGISTRY[name](**kwargs)