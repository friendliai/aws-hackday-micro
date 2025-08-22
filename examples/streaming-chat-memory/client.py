from openai import OpenAI

client = OpenAI(base_url="http://0.0.0.0:8080/v1")

response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[{"role": "user", "content": "do you know my name?"}],
    user="user123",
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
