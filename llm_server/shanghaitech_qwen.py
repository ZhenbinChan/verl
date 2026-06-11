import os

base_url = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
api_key = "a7db49a1c59a44b2b255c8a0fb83dda4"
model_name = "qwen-instruct"
provider = "openai_compatible"


def main():
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Calling: {base_url}/chat/completions")
    print(f"Model: {model_name}")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, please introduce yourself briefly."},
        ],
        stream=False,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
