from datasets import load_dataset

def main():
    # download danish split
    dataset = load_dataset("PaDaS-Lab/webfaq", "dan", split='default')
    # save to disk
    dataset.save_to_disk("data/webfaq_danish")


if __name__ == "__main__":
    main()
