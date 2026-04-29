from dotenv import load_dotenv
import os
from openai import OpenAI

def load_api_key():
    load_dotenv()
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise ValueError("API key not found. Check your .env file.")
    return key

def make_client() -> OpenAI:
    return OpenAI(api_key=load_api_key())

def send_prompt(prompt, model="gpt-4.1", max_tokens=256, temperature=0.5, response_format=None):
    client = make_client()
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response_format:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()
