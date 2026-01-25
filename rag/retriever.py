"""Vector retrieval and search using ChromaDB + Langchain"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class LangChainRetrieverWrapper(BaseRetriever):
    """LangChain-compatible retriever wrapper for RAGRetriever"""
    
    rag_retriever: object
    top_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Get relevant documents for a query"""
        retrieved_docs = self.rag_retriever.retrieve(query, self.top_k)
        
        # Convert to LangChain Document format
        langchain_docs = []
        for doc in retrieved_docs:
            langchain_doc = Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            )
            langchain_docs.append(langchain_doc)
        
        return langchain_docs


class RAGRetriever:
    """Retrieve relevant context from ChromaDB"""
    
    def __init__(self, persist_directory: str = "./db/chromadb_store", collection_name: str = "sunmarke_content"):
        from db.chroma_manager import ChromaManager
        import google.generativeai as genai
        import os
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize Gemini for embeddings
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Initialize ChromaDB using ChromaManager
        self.chroma_manager = ChromaManager(persist_directory, collection_name)
        self.client = self.chroma_manager.client
        self.collection = self.chroma_manager.collection
        
        print(f"Connected to ChromaDB collection: {collection_name}")
        
        # Auto-load data if collection is empty
        self.chroma_manager.auto_load_if_empty()
    

    
    def embed_query(self, query: str) -> List[float]:
        """Generate query embedding using Gemini API"""
        import google.generativeai as genai
        
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            # Return full 768D to match new ChromaDB embeddings
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            # Return zero vector as fallback (768D)
            return [0.0] * 768
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant documents for query
        
        Args:
            query: User's question
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            doc = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            }
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc['metadata']
            content = doc['content']
            
            context_part = f"""
[Source {i}]
Title: {metadata.get('title', 'Unknown')}
Category: {metadata.get('category', 'Unknown')}
Content: {content}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        """
        Search and return formatted results
        
        Returns:
            Dict with 'context', 'sources', and 'retrieved_docs'
        """
        retrieved_docs = self.retrieve(query, top_k)
        context = self.format_context(retrieved_docs)
        
        # Extract source information
        sources = []
        for doc in retrieved_docs:
            sources.append({
                'title': doc['metadata'].get('title', 'Unknown'),
                'category': doc['metadata'].get('category', 'Unknown'),
                'relevance': 1 - doc['distance']  
            })
        
        return {
            'context': context,
            'sources': sources,
            'retrieved_docs': retrieved_docs
        }
    
    def as_langchain_retriever(self, top_k: int = 5):
        """Get LangChain-compatible retriever"""
        return LangChainRetrieverWrapper(
            rag_retriever=self,
            top_k=top_k
        )
