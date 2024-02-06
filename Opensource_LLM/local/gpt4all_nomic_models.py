# https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-bindings/python
# https://colab.research.google.com/drive/1PJgZ6gVeFneKmACQY0gN3wlmIf61yY70#scrollTo=apa9-Iak-ZCb
from gpt4all import GPT4All
model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")# This will download the model locally, in windows: C:\Users\<usernmame>\AppData\Local\nomic.ai\GPT4All
output = model.generate("The capital of France is ", max_tokens=3)
print(output)