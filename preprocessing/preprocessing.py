
import re
from transformers import AutoTokenizer  # Hugging Face tokenizer
from typing import List

# Load a Hugging Face tokenizer (e.g., GPT-4, BERT, etc.)
# Replace 'gpt2' with the tokenizer of your choice
local_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Step 2: Preprocessing the Text
def clean_text(text: str) -> str:
    """Clean the extracted text."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove unnecessary symbols
    return text


def tokenize_text_with_hf(cleaned_text: str, tokenizer = local_tokenizer) -> List[str]:
    """Tokenize text into sentences or logical units using Hugging Face tokenizer."""
    # tokens = tokenizer.tokenize(text)
    # return tokenizer.convert_ids_to_tokens(tokenizer.encode(text))

    # Tokenize text
    token_ids = tokenizer.encode(cleaned_text)  # Tokenize and encode to token IDs
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # Convert IDs to tokens
    return tokens


def chunk_text_hf(tokens: List[str], max_tokens: int = 500) -> List[str]:
    # """Chunk the text into smaller pieces using Hugging Face token limits."""
    # chunks = []
    # current_chunk = []
    # current_token_count = 0
    #
    # for token in tokens:
    #     token_count = len(token.split())
    #     if current_token_count + token_count > max_tokens:
    #         chunks.append(" ".join(current_chunk))
    #         current_chunk = []
    #         current_token_count = 0
    #     current_chunk.append(token)
    #     current_token_count += token_count
    #
    # if current_chunk:
    #     chunks.append(" ".join(current_chunk))
    #
    # return chunks
    """
        Chunk the tokenized text into smaller parts that fit the model's token limit.
        :param tokens: List of tokens from the tokenizer.
        :param max_tokens: Maximum number of tokens per chunk.
        :return: List of token chunks.
        """
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(" ".join(tokens[i:i + max_tokens]))
    return chunks


def preprocess_text_hf(tokenizer, raw_text: str, max_tokens: int = 500) -> List[str]:
    """Full preprocessing pipeline: clean, tokenize, and chunk using Hugging Face."""
    cleaned_text = clean_text(raw_text)
    tokens = tokenize_text_with_hf(cleaned_text=cleaned_text, tokenizer=tokenizer)
    return chunk_text_hf(tokens, max_tokens)



