from .generators import NousHermesMistralGenerator, SuzumeLlama3Generator, Yi34BGenerator

GENERATOR_REGISTRY = {
    "nous-hermes-mistral"   : NousHermesMistralGenerator,
    "suzume-llama3"         : SuzumeLlama3Generator,
    "yi-34b"                : Yi34BGenerator
}

def get_generator(name, **kwargs):
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator: {name}")
    return GENERATOR_REGISTRY[name](**kwargs)