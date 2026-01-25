"""
Embedding generation for vector storage
Generates embeddings using Google Gemini API
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
    import os
    from pathlib import Path
    import google.generativeai as genai
    from dotenv import load_dotenv
    from tqdm import tqdm
    
    print("STEP 3: GENERATING EMBEDDINGS WITH GEMINI")
    
    # Load environment
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found!")
        exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    print("✅ Gemini configured")
    
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
    print("Using Gemini text-embedding-004 model (768D)")
    
    # Generate embeddings
    print("Generating embeddings with Gemini...")
    
    start_time = time.time()
    embeddings = []
    failed_count = 0
    
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding")):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk['content'],
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed chunk {i}: {e}")
            embeddings.append([0.0] * 768)  # Zero vector fallback
            failed_count += 1
    
    elapsed = time.time() - start_time
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding
        chunk['embedding_model'] = 'text-embedding-004'
        chunk['embedding_dim'] = 768
    
    # Save
    output_file = Path('data/embedded_chunks/sunmarke_embedded.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(" EMBEDDING COMPLETE!")
    print(f" Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f" Speed: {len(chunks)/elapsed:.1f} chunks/sec")
    print(f" Chunks: {len(chunks)}")
    print(f" Failed: {failed_count}")
    print(f" Dimensions: 768 (Gemini)")
    print(f" Saved to: {output_file}")
    print(f" Size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print(" Next: Store in ChromaDB")

