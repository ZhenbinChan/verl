from openai import OpenAI

client = OpenAI(
    base_url="127.0.0.1:1029",
)

completion = client.chat.completions.create(
    model="deepseek-r1-250528",# deepseek-r1-250528 deepseek-v3-250324
    messages=[
        {
            "role": "user",
            "content": "Say Hi!",
        },
    ],
)

print(completion.choices[0].message.content)
