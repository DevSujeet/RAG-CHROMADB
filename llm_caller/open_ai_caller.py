import openai
from config.config import LLM_Settings

# Initialize OpenAI Client
settings = LLM_Settings()
openai_key = settings.openai_api_key

async def ask_llm(prompt: str) -> str:
    """
    Call OpenAI ChatGPT API to get a response based on the constructed prompt.
    """
    try:
        openai.api_key = openai_key
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling ChatGPT: {e}"
