import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file in the parent directory
env_path = os.path.join("..", '.env')
load_dotenv(env_path)

# Create an OpenAI client instance using the API key from the .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", default="sk-<your-api-key>"))

def chat_with_gpt_stream(prompt, model="gpt-3.5-turbo", temperature=0.7):
    try:
        # Create a chat completion using the OpenAI client
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            stream=True,
        )
        # Extract and return the response content
        return stream
    except Exception as e:
        # Return the error message if an exception occurs
        return str(e)

def main():
    print("Welcome to ChatGPT Clone!")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    temperature = float(os.getenv("TEMPERATURE", 0.7))
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting ChatGPT Clone. Goodbye!")
            break
        # Call the function to get a response from GPT-3.5-turbo
        stream = chat_with_gpt_stream(user_input, model=model_name, temperature=temperature)
        print("GPT: ", end="")
        for chunk in stream:
            #print(chunk or "", end="")
            print(chunk.choices[0].delta.content or "", end="")
        print()


if __name__ == "__main__":
    main()
