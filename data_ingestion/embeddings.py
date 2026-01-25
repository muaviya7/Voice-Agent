"""
Embedding generation for vector storage
Generates embeddings using OpenAI or Sentence Transformers
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
import time


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from config import settings
            self.client = None  # OpenAI client disabled for now
        except:
            pass
        self.model = model
        self.batch_size = 100
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                print(f"Generated embeddings for batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating embeddings for batch: {str(e)}")
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """Add embeddings to documents"""
        texts = [doc['content'] for doc in documents]
        
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to documents
        embedded_docs = []
        for doc, embedding in zip(documents, embeddings):
            if embedding:
                doc_copy = doc.copy()
                doc_copy['embedding'] = embedding
                embedded_docs.append(doc_copy)
            else:
                print(f"Skipping document without embedding: {doc.get('url')}")
        
        print(f"Successfully embedded {len(embedded_docs)}/{len(documents)} documents")
        return embedded_docs


if __name__ == "__main__":
    import json
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    
    print("STEP 3: GENERATING EMBEDDINGS")
    
    # Load chunks
    input_file = Path('data/scraped_content/sunmarke_chunks.json')
    
    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print("Run chunker first: python data_ingestion/chunker.py")
        exit(1)
    
    print(f"Loading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Load model
    print("Loading model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded!")
    
    # Generate embeddings
    print("Generating embeddings...")
    texts = [chunk['content'] for chunk in chunks]
    
    start_time = time.time()
    batch_size = 100
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())
    
    elapsed = time.time() - start_time
    
    # Add embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding
        chunk['embedding_model'] = 'all-MiniLM-L6-v2'
        chunk['embedding_dim'] = 384
    
    # Save
    output_file = Path('data/embedded_chunks/sunmarke_embedded.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print("EMBEDDING COMPLETE!")
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Speed: {len(chunks)/elapsed:.1f} chunks/sec")
    print(f"Chunks: {len(chunks)}")
    print(f"Dimensions: 384")
    print(f"Saved to: {output_file}")
    print(f"Size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print("Next: Store in ChromaDB")

