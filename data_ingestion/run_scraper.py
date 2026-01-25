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
from sentence_transformers import SentenceTransformer
from db.chroma_manager import ChromaManager
import json
from tqdm import tqdm


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
    
    print("Loading embedding model (sentence-transformers)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully")
    
    print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
    embeddings = []
    
    # Generate with progress bar
    for chunk in tqdm(all_chunks, desc="Embedding chunks"):
        embedding = embedding_model.encode(chunk['content'])
        embeddings.append(embedding.tolist())
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"   - Embedding dimensions: {len(embeddings[0])}")
    
    # Add embeddings to chunks
    embedded_chunks = []
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk_copy = chunk.copy()
        chunk_copy['embedding'] = embedding
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
