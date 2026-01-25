"""
Complete Data Ingestion Pipeline Runner
Runs scraping -> chunking -> embedding -> ChromaDB storage sequentially
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.scraper import LocalHTMLProcessor
from data_ingestion.chunker import ContentChunker
from db.chroma_manager import ChromaManager
import json
from tqdm import tqdm
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time


def run_full_pipeline():
    """Execute complete data ingestion pipeline"""
    
    print("\nSUNMARKE SCHOOL DATA INGESTION PIPELINE")
    
    # Configuration
    html_directory = "sunmarke"  # Local HTML files directory
    output_dir = Path("data/scraped_content")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: SCRAPE & EXTRACT CONTENT FROM HTML FILES
    # ========================================================================

    print("\nSTEP 1: SCRAPING HTML FILES")
    
    processor = LocalHTMLProcessor(html_directory=html_directory)
    
    print(f"Reading HTML files from: {html_directory}")
    processed_data = processor.process_all_files()
    
    if not processed_data:
        print("ERROR: No data extracted! Check HTML directory path.")
        return False
    
    print(f"\nExtracted content from {len(processed_data)} files")
    
    # Save raw data
    raw_output = output_dir / "sunmarke_raw.json"
    with open(raw_output, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"Saved raw data to: {raw_output}")
    
    # ========================================================================
    # STEP 2: CHUNK CONTENT INTO SMALLER PIECES
    # ========================================================================
    print("\nSTEP 2: CHUNKING CONTENT")
    
    chunker = ContentChunker(chunk_size=1200, chunk_overlap=200)
    
    print(f"Chunk size: 1200 characters")
    print(f"Chunk overlap: 200 characters")
    
    all_chunks = chunker.chunk_batch(processed_data)
    
    print(f"\nCreated {len(all_chunks)} chunks")
    
    # Get statistics
    stats = chunker.get_chunk_stats(all_chunks)
    print(f"\nChunk Statistics:")
    print(f"   - Average size: {stats['avg_chunk_size']:.0f} characters")
    print(f"   - Min size: {stats['min_chunk_size']} characters")
    print(f"   - Max size: {stats['max_chunk_size']} characters")
    print(f"   - Total characters: {stats['total_characters']:,}")
    
    # Save chunks
    chunks_output = output_dir / "sunmarke_chunks.json"
    with open(chunks_output, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved chunks to: {chunks_output}")
    
    # ========================================================================
    # STEP 3: GENERATE EMBEDDINGS
    # ========================================================================
    print("\nSTEP 3: GENERATING EMBEDDINGS")
    
    # Load environment for Gemini API
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment!")
        print("Please set your Google API key in .env file")
        return False
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured")
    
    print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
    print("Using Gemini text-embedding-004 model (768D)")
    
    embeddings = []
    failed_count = 0
    start_time = time.time()
    
    # Generate embeddings with progress bar
    for i, chunk in enumerate(tqdm(all_chunks, desc="Embedding chunks")):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk['content'],
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
            
            # Rate limiting to avoid quota issues
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nFailed chunk {i}: {e}")
            embeddings.append([0.0] * 768)  # Zero vector fallback
            failed_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\nGenerated {len(embeddings)} embeddings in {elapsed:.2f}s")
    print(f"   - Embedding dimensions: 768 (Gemini)")
    print(f"   - Speed: {len(embeddings)/elapsed:.1f} chunks/sec")
    print(f"   - Failed embeddings: {failed_count}")
    
    # Add embeddings to chunks
    embedded_chunks = []
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk_copy = chunk.copy()
        chunk_copy['embedding'] = embedding
        chunk_copy['embedding_model'] = 'text-embedding-004'
        chunk_copy['embedding_dim'] = 768
        embedded_chunks.append(chunk_copy)
    
    # Save embedded chunks
    embedded_output = Path("data/embedded_chunks")
    embedded_output.mkdir(parents=True, exist_ok=True)
    embedded_file = embedded_output / "sunmarke_embedded.json"
    
    with open(embedded_file, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved embedded chunks to: {embedded_file}")
    
    # ========================================================================
    # STEP 4: STORE IN CHROMADB
    # ========================================================================
    print("STEP 4: STORING IN CHROMADB")
    
    print("Initializing ChromaDB...")
    chroma_manager = ChromaManager(persist_directory="./db/chromadb_store")
    
    # Prepare data for ChromaDB
    documents = [chunk['content'] for chunk in embedded_chunks]
    embeddings_list = [chunk['embedding'] for chunk in embedded_chunks]
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(embedded_chunks):
        metadata = {
            'title': chunk.get('title', 'Unknown'),
            'category': chunk.get('category', 'General'),
            'subcategory': chunk.get('subcategory', ''),
            'chunk_index': chunk.get('chunk_index', 0),
            'total_chunks': chunk.get('total_chunks', 1)
        }
        metadatas.append(metadata)
        ids.append(f"doc_{i}")
    
    print(f"Storing {len(documents)} documents...")
    
    # Store in ChromaDB
    chroma_manager.store_documents(
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids,
        collection_name="sunmarke_content"
    )
    
    print(f"Successfully stored {len(documents)} documents in ChromaDB")
    
    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    print("\nPIPELINE COMPLETE")
    
    print("\nFinal Summary:")
    print(f"   - Files processed: {len(processed_data)}")
    print(f"   - Chunks created: {len(all_chunks)}")
    print(f"   - Embeddings generated: {len(embeddings)}")
    print(f"   - Failed embeddings: {failed_count}")
    print(f"   - Documents in ChromaDB: {len(documents)}")
    
    print("\nOutput Files:")
    print(f"   - Raw data: {raw_output}")
    print(f"   - Chunks: {chunks_output}")
    print(f"   - Embedded chunks: {embedded_file}")
    print(f"   - ChromaDB: ./db/chromadb_store/")
    
    print("\nReady to use! Run the chatbot with: streamlit run app.py")
    
    return True


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    success = run_full_pipeline()
    
    elapsed_time = time.time() - start_time
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    if success:
        print("Pipeline executed successfully!")
    else:
        print("Pipeline failed!")
        sys.exit(1)
