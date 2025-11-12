import google.generativeai as genai

# Configure your API key
genai.configure(api_key="AIzaSyDv_yMAJUOqAWvhykk9c5I1mVKk92x981s")

# List available models
models = genai.list_models()

for model in models:
    print(model.name)
