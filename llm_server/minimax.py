import os

base_url = "https://api.minimaxi.com/v1"
api_key = os.environ.get("MINIMAX_API_KEY", "")
model_name = "MiniMax-M2.7"
provider = "minimax"
default_args = {}
extra_body = {"reasoning_split": True}


def main():
    from openai import OpenAI

    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is required.")
    client = OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi, how are you?"},
        ],
        # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
        extra_body=extra_body,
    )

    print(f"Thinking:\n{response.choices[0].message.reasoning_details[0]['text']}\n")
    print(f"Text:\n{response.choices[0].message.content}\n")


if __name__ == "__main__":
    main()
