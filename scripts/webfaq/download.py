from datasets import load_dataset

def main(config_path):
    # Stream the Danish subset and take the first 100 examples
    dataset = load_dataset("PaDaS-Lab/webfaq", "dan", split='default')
    # Take the first 100 examples
    dataset.save_to_disk("/dtu/p1/webfaq_danish")


if __name__ == "__main__":
    main()
