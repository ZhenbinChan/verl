import datasets
from tqdm import tqdm

def main():
    dataset_name = "lucasmccabe/logiqa"
    dataset = datasets.load_dataset(dataset_name, "default")["test"]

    dataset_list = []
    for i,data in tqdm(enumerate(dataset), total=len(dataset)):
        example = {}
        example['id'] = i
        example['context'] = data["context"]
        example['query'] = data["query"]
        # example['options'] = data["options"]
        example['options'] = "\n".join([f"{chr(65 + i)}. {text}" for i, text in enumerate(data['options'])])
        example["question"] = example['context'] + "\n" + example['query'] + "\n" + example['options']
        example['answer'] = f"{chr(65 + data['correct_option'])}"
        dataset_list.append(example)

    # write to jsonl file
    import json
    with open("/home/chenzhb/Workspaces/verl/mcts_utils/data/logiqa.jsonl", "w", encoding="utf-8") as f:
        for example in dataset_list:
            f.write(json.dumps(example) + "\n")

if __name__ == '__main__':
    main()



