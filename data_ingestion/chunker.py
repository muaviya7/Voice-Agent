"""
Text chunking utilities for creating embeddings
Splits large content into manageable chunks
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
from utils.logger import logger


class ContentChunker:
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_content(self, content: str) -> List[str]:
        """Split content into chunks"""
        if not content:
            return []
        
        # If content is short enough, return as single chunk
        if len(content) <= self.chunk_size:
            return [content]
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        return chunks
    
    def chunk_page(self, page_data: Dict) -> List[Dict]:
        """Chunk a single page into multiple documents"""
        content = page_data.get('content', '')
        chunks = self.chunk_content(content)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            doc = {
                'content': chunk,
                'title': page_data.get('title'),
                'category': page_data.get('category'),
                'subcategory': page_data.get('subcategory'),
                'meta_description': page_data.get('meta_description', ''),
                'chunk_index': i,
                'total_chunks': len(chunks),
            }
            chunked_docs.append(doc)
        
        return chunked_docs
    
    def chunk_batch(self, pages: List[Dict]) -> List[Dict]:
        """Chunk multiple pages"""
        all_chunks = []
        
        for page in pages:
            chunks = self.chunk_page(page)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }
        
        return stats


if __name__ == "__main__":
    import json
    from pathlib import Path
    from processor import ContentProcessor
    
    print("STEP 2: CHUNKING") 
    # Load raw scraped data
    input_file = Path('data/scraped_content/sunmarke_raw.json')
    
    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print("Run scraper first: python data_ingestion/scraper.py")
        exit(1)
    
    print(f"Loading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} pages")
    
    # Initialize processor and chunker
    processor = ContentProcessor()
    chunker = ContentChunker(chunk_size=1200, chunk_overlap=200)
    
    # Process and chunk
    print("Processing and chunking...")
    all_chunks = []
    
    for page in raw_data:
        processed = processor.process_page(page)
        if processed:
            chunks = chunker.chunk_page(processed)
            all_chunks.extend(chunks)
    
    # Save chunks
    output_file = Path('data/scraped_content/sunmarke_chunks.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print("CHUNKING COMPLETE!")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"Avg chunk: {sum(len(c['content']) for c in all_chunks) / len(all_chunks):.0f} chars")
    print("Next: python data_ingestion/embeddings.py")
