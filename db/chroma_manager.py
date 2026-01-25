"""ChromaDB manager for vector storage"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import json


class ChromaManager:
    """Manage ChromaDB operations"""
    
    def __init__(self, persist_directory: str = "./db/chromadb_store", collection_name: str = "sunmarke_content"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Sunmarke School content embeddings"}
        )
    
    def add_documents(self, documents: List[Dict]) -> int:
        """Add documents with embeddings to ChromaDB"""
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for i, doc in enumerate(documents):
            # Generate unique ID
            doc_id = f"chunk_{i}_{doc.get('title', 'unknown').replace(' ', '_')[:50]}"
            ids.append(doc_id)
            
            # Extract embedding
            embeddings.append(doc['embedding'])
            
            # Extract metadata (everything except embedding and content)
            # ChromaDB doesn't accept None values, so convert to empty strings
            metadata = {
                'title': str(doc.get('title', '')),
                'category': str(doc.get('category', '')),
                'subcategory': str(doc.get('subcategory', '')),
                'meta_description': str(doc.get('meta_description', '')),
                'chunk_index': int(doc.get('chunk_index', 0)),
                'total_chunks': int(doc.get('total_chunks', 1)),
                'embedding_model': str(doc.get('embedding_model', 'unknown'))
            }
            metadatas.append(metadata)
            
            # Store content text
            documents_text.append(doc['content'])
        
        # Add to ChromaDB in batches
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_documents = documents_text[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
            
            total_added += len(batch_ids)
            print(f"✅ Added batch {i//batch_size + 1}: {total_added}/{len(ids)} documents")
        
        return total_added
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'persist_directory': self.persist_directory
        }
    
    def is_empty(self) -> bool:
        """Check if collection is empty"""
        return self.collection.count() == 0
    
    def load_from_json(self, json_file_path: str) -> int:
        """Load embedded chunks from JSON file into ChromaDB"""
        print(f"📂 Loading embedded chunks from: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✅ Loaded {len(chunks)} embedded chunks from JSON")
        
        # Add documents to ChromaDB
        print("🔄 Adding documents to ChromaDB...")
        total_added = self.add_documents(chunks)
        
        print(f"✅ Successfully added {total_added} documents to ChromaDB!")
        return total_added
    
    def auto_load_if_empty(self, embedded_chunks_path: str = None) -> bool:
        """Auto-load embedded chunks if collection is empty (for Render deployments)"""
        if not self.is_empty():
            print(f"ChromaDB has {self.collection.count()} documents")
            return False
        
        print("ChromaDB is empty! Auto-loading embedded chunks...")
        
        # Default path for embedded chunks
        if embedded_chunks_path is None:
            from pathlib import Path
            embedded_chunks_path = Path(__file__).parent.parent / "data" / "embedded_chunks" / "sunmarke_embedded.json"
        
        if not Path(embedded_chunks_path).exists():
            print(f"ERROR: Embedded chunks file not found at {embedded_chunks_path}")
            print(" ChromaDB will remain empty. Please ensure the file exists.")
            return False
        
        try:
            total_added = self.load_from_json(str(embedded_chunks_path))
            print(f" Successfully auto-loaded {total_added} documents into ChromaDB!")
            return True
        except Exception as e:
            print(f" ERROR auto-loading embedded chunks: {e}")
            print("  ChromaDB will remain empty.")
            return False
    
    def reset_collection(self):
        """Delete and recreate collection (careful!)"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Sunmarke School content embeddings"}
        )
        print(f"⚠️  Collection '{self.collection_name}' reset!")


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("STEP 4: CHROMADB STORAGE")
    print("=" * 60)
    
    # Load embedded chunks
    input_file = Path('data/embedded_chunks/sunmarke_embedded.json')
    
    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print(" Run embeddings first: python data_ingestion/embeddings.py")
        exit(1)
    
    print(f"\nLoading embedded chunks from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} embedded chunks")
    
    # Initialize ChromaDB
    print("\nInitializing ChromaDB...")
    chroma = ChromaManager(
        persist_directory="./db/chromadb_store",
        collection_name="sunmarke_content"
    )
    print("ChromaDB initialized!")
    
    # Check if collection already has data
    stats = chroma.get_collection_stats()
    if stats['total_documents'] > 0:
        print(f"\nCollection already has {stats['total_documents']} documents")
        response = input("Reset and reload? (yes/no): ")
        if response.lower() == 'yes':
            chroma.reset_collection()
        else:
            print("Aborted. Keeping existing data.")
            exit(0)
    
    # Add documents to ChromaDB
    print("\nAdding documents to ChromaDB...")
    start_time = time.time()
    
    total_added = chroma.add_documents(chunks)
    
    elapsed = time.time() - start_time
    
    # Get final stats
    final_stats = chroma.get_collection_stats()
    
    print(" CHROMADB STORAGE COMPLETE!")
    print(f" Documents stored: {total_added}")
    print(f"  Collection: {final_stats['collection_name']}")
    print(f" Location: {final_stats['persist_directory']}")
    print(f"  Time: {elapsed:.2f}s")
    print(f" Speed: {total_added/elapsed:.1f} docs/sec")
    print("\n Next: Build RAG retriever")
    print("=" * 60)
