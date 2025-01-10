import asyncio
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PyPDF2 import PdfReader


# Step 1: Text Extraction (Hybrid Approach)
async def extract_text_pypdf2_async(pdf_path):
    """Extract text from selectable PDFs using PyPDF2."""
    reader = PdfReader(pdf_path)
    tasks = [asyncio.to_thread(page.extract_text) for page in reader.pages]
    results = await asyncio.gather(*tasks)
    print(f" extracted part count {len(results)}")
    print(f" extract first part \n{results[0]}")
    return "\n".join(results)


async def extract_text_ocr_async(pdf_path, chunk_size=10):
    """Extract text from scanned PDFs using OCR."""
    images = await asyncio.to_thread(convert_from_path, pdf_path)
    text_results = []
    for i in range(0, len(images), chunk_size):
        chunk_images = images[i:i + chunk_size]
        tasks = [asyncio.to_thread(image_to_string, img) for img in chunk_images]
        text_results.extend(await asyncio.gather(*tasks))
    return "\n".join(text_results)


async def extract_text_combined(pdf_path, use_ocr=False):
    """Unified text extraction function."""
    if use_ocr:
        print("Extracting text using OCR...")
        return await extract_text_ocr_async(pdf_path)
    else:
        print("Extracting text using PyPDF2...")
        return await extract_text_pypdf2_async(pdf_path)