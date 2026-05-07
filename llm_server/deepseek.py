from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-QpaRQWxuaYyNcROMt3qi5g", # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://litellm.mybigai.ac.cn/",
)

completion = client.chat.completions.create(
    model="deepseek-v3-250324",# deepseek-r1-250528
    messages=[
        {
            "role": "user",
            "content": "Say Hi!",
        },
    ],
)

print(completion.choices[0].message.content)
