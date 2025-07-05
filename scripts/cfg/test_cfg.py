from src.generator import get_generator

def main():
    # import generator
    generator = get_generator('suzume-llama3')
    # extract logits
    context = "The capital of France is in England."
    prompt = "The capital of France is"
    correct_answer="Paris"
    incorrect_answer="London"
    generator.check_if_flipped(context,prompt,correct_answer,incorrect_answer)

if __name__ == "__main__":
    main()