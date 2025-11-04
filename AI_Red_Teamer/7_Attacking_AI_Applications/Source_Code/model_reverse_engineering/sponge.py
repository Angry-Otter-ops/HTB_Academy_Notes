from transformers import AutoTokenizer
import json

model = 'openai-community/gpt2'

while 1:
	text = input("> ")

	tokens = AutoTokenizer.from_pretrained(model).tokenize(text)
	print(f"Number of Input Characters: {len(text)}")
	print(f"Number of Tokens: {len(tokens)}")
	print(json.dumps(tokens, indent=2))
