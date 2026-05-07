from openai import OpenAI

# 初始化客户端
# 注意：base_url 必须以 /v1 结尾
client = OpenAI(
    base_url="http://localhost:4869/v1",  # 对应您启动命令中的 port 4869
    api_key="EMPTY",                      # vllm 默认不需要 key，但必须填一个占位符
)

# 发送请求 (非流式，一次性返回)
print("--- 普通对话测试 ---")
try:
    completion = client.chat.completions.create(
        model="qwen2.5-3b",  # 必须与您启动命令中的 --served-model-name 一致
        messages=[
            {"role": "system", "content": "你是一个乐于助人的AI助手。"},
            {"role": "user", "content": "请用Python写一个简单的Hello World程序。"}
        ],
        temperature=0.7,
        max_tokens=1024,     # 对应启动参数中的 --max-model-len (最大生成长度)
    )

    # 打印结果
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"调用出错: {e}")


# 发送请求 (流式 Streaming，像 ChatGPT 一样一个字一个字出)
print("\n--- 流式对话测试 ---")
try:
    stream = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=[
            {"role": "user", "content": "给我讲一个简短的鬼故事。"}
        ],
        stream=True,  # 开启流式输出
        temperature=0.8,
    )

    print("AI回复: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print() # 换行

except Exception as e:
    print(f"流式调用出错: {e}")
