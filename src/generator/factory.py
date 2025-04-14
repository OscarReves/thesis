from .generators import GPT2Generator, NousHermesMistral2Generator

GENERATOR_REGISTRY = {
    "gpt2": GPT2Generator,
    "nous-hermes-mistral" : NousHermesMistral2Generator,
}

def get_generator(name, **kwargs):
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator: {name}")
    return GENERATOR_REGISTRY[name](**kwargs)