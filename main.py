# NLTK tokenizer download
# import nltk
# print(f"nltk search at {nltk.data.path}")
# print(f"nltk found at {nltk.data.find('tokenizers/punkt')}")
# nltk.download("punkt")

import asyncio

from chroma_db_handler import ChromaDBHandler
from pdf_reader.pdf_reader_async import extract_text_combined
from preprocessing.preprocessing import preprocess_text_hf


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# from nltk.tokenize import word_tokenize
#
# text = "This is a test sentence."
# tokens = word_tokenize(text)
# print(tokens)


# Integrated Pipeline
async def process_pdf(tokenizer, pdf_path, use_ocr=False, max_tokens=500):
    """
    Extract and preprocess text from a PDF.
    :param pdf_path: Path to the PDF file.
    :param use_ocr: Whether to use OCR for text extraction.
    :param max_tokens: Maximum tokens per chunk.
    :return: List of processed text chunks.
    """
    # Step 1: Extract text
    raw_text = await extract_text_combined(pdf_path, use_ocr)

    # Step 2: Preprocess text
    processed_chunks = preprocess_text_hf(tokenizer=tokenizer, raw_text=raw_text, max_tokens=max_tokens)
    return processed_chunks


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print()


    async def main():
        # Initialize components
        chroma_handler = ChromaDBHandler()
        tokenizer = chroma_handler.embedding_model.tokenizer
        #---------------------------
        file_name = "ramayan-book-1-chap-1.pdf"
        pdf_path = f"documents/{file_name}"  # Replace with your PDF path
        use_ocr = False  # Set to True for scanned PDFs
        max_tokens = 256  # Adjust based on your LLM's token limit

        # Process the PDF
        chunks = await process_pdf(tokenizer, pdf_path, use_ocr, max_tokens)

        # Print the processed chunks
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:\n{chunk}\n")
        # store the chunk in chromaDB
        chroma_handler.store_chunks_protected(chunks=chunks, source_document=file_name)

        # chroma_handler = ChromaDBHandler()
        # Example query
        query = "who was Valmiki before he became a sage? "
        top_k = 3

        # Query ChromaDB
        results = chroma_handler.query_chunks(query, top_k=top_k)
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Document: {result['document']}")
            print(f"Metadata: {result['metadata']}\n")


    asyncio.run(main())

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
