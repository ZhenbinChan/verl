base_url='https://api.minimaxi.com/v1'
api_key='sk-cp-3wmwcLqQq6GY8LTPH7nlV6qsAwuLYwqe2je8Su-FEWh2KpL4NaDN8lK93dtNcfgJfX1TnU5fIwBGh5d_b8iGKMPkNMaIHizj98k8jm_QVILHHmSJrs4SqqY'

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)

response = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, how are you?"},
    ],
    # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
    extra_body={"reasoning_split": True},
)

print(f"Thinking:\n{response.choices[0].message.reasoning_details[0]['text']}\n")
print(f"Text:\n{response.choices[0].message.content}\n")