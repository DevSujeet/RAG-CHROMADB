import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid

class ChromaDBHandler:
    def __init__(self, persist_directory: str = "./chroma_storage", embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDBHandler with a ChromaDB client and embedding model.
        :param persist_directory: Directory to persist the ChromaDB database.
        :param embedding_model_name: Hugging Face model name for generating embeddings.
        """
        # check this out: https://docs.trychroma.com/docs/run-chroma/client-server
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection_name = "document_chunks"

        # Check if the collection exists, otherwise create it
        # Create or get the collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                get_or_create=True
            )
        except ValueError as e:
            print(f"Error creating or accessing the collection: {e}")
            raise


    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks using Sentence Transformers.
        :param chunks: List of text chunks.
        :return: List of embeddings.
        """
        return self.embedding_model.encode(chunks, convert_to_numpy=True).tolist()

    def store_chunks(self, chunks: List[str], source_document: str):
        """
        Store text chunks and their embeddings in ChromaDB.
        :param chunks: List of text chunks.
        :param source_document: Identifier for the source document.
        """
        # Generate embeddings for the chunks
        embeddings = self.generate_embeddings(chunks)

        # Add chunks to the ChromaDB collection
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {"source_document": source_document, "chunk_index": idx}
            self.collection.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())],  # Generate a unique ID for each chunk
                embeddings=[embedding],
            )
        print(f"Stored {len(chunks)} chunks from '{source_document}' in ChromaDB.")

    def query_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query the ChromaDB collection for the most relevant chunks based on the query.
        :param query: User's query string.
        :param top_k: Number of top results to retrieve.
        :return: List of dictionaries containing documents and metadata.
        """
        # Generate embedding for the query
        query_embedding = self.generate_embeddings([query])[0]

        # Search the collection for the most similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format and return the results
        return [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

