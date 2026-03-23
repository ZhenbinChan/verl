import datasets
from openai import OpenAI

def call_vllm_api(data, base_url="http://localhost:4869/v1", api_key="EMPTY", model='qwen2.5-3b', args=None):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": data},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return None
    
    
def process_fn(example):
    context = example['context']
    question = example['query']
    options = "\n".join([f"{chr(65 + i)}. {text}" for i, text in enumerate(example['options'])])
    gt = f"{chr(65 + example['correct_option'])}"
    return {
        "context": context,
        "question": question,
        "options": options,
        "answer": gt,
    }

if __name__ == "__main__":
    dataset_name = "lucasmccabe/logiqa"
    base_url = "http://localhost:4869/v1"
    api_key = "EMPTY"
    model = 'qwen2.5-3b'
    args = {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
    }
    dataset = datasets.load_dataset(dataset_name, "default")["train"]
    dataset = dataset.map(process_fn, batched=False)
    for data in dataset:
        generated_solution = call_vllm_api(data, base_url=base_url, api_key=api_key, model=model, args=args)
        print("generated_solution:", generated_solution)