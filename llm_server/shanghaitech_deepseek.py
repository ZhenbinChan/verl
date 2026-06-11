import os

base_url = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
api_key = "f7f4b08f3abf4632afdb26baacbfb76e"
model_name = "deepseek-pro"
provider = "openai_compatible"
default_args = {
    "reasoning_effort": "high",
}
extra_body = {"thinking": {"type": "enabled"}}


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
        reasoning_effort=default_args["reasoning_effort"],
        extra_body=extra_body,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
