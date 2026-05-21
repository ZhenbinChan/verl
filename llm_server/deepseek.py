import os

base_url = "https://api.deepseek.com"
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
model_name = "deepseek-v4-pro"
provider = "openai_compatible"
default_args = {
    "reasoning_effort": "high",
}
extra_body = {"thinking": {"type": "enabled"}}


def main():
    from openai import OpenAI

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False,
        reasoning_effort=default_args["reasoning_effort"],
        extra_body=extra_body,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
