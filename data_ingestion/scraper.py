"""SIMPLE HTML EXTRACTOR - NO BULLSHIT"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re
from utils.logger import logger
from config import settings


class LocalHTMLProcessor:
    def __init__(self, html_directory: str = None):
        self.html_directory = Path(html_directory) if html_directory else Path("sunmarke")
        self.processed_data: List[Dict] = []
    
    def extract_category_from_path(self, file_path: Path) -> str:
        """Extract category from file path"""
        parts = file_path.parts
        for part in parts:
            if part in ['about', 'learning', 'signature-programmes', 'for-parents', 'activities', 'contact-us', 'admissions']:
                return part.replace('-', ' ').title()
        return "General"
    
    def extract_subcategory_from_path(self, file_path: Path) -> Optional[str]:
        """Extract subcategory from file path"""
        parts = file_path.parts
        if len(parts) > 2:
            return parts[-2].replace('-', ' ').title()
        return None
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract ALL text, clean it, done."""
        # NUKE everything that's not content
        for tag in soup(['script', 'style', 'noscript', 'iframe', 'svg', 
                       'nav', 'header', 'footer', 'aside', 'button', 'input']):
            tag.decompose()
        
        # Get ALL remaining text
        text = soup.get_text(separator=' ', strip=True)
        
        # Aggressive cleaning - all whitespace to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove navigation junk and common boilerplate
        text = re.sub(r'Apply Visit Enquire VLE Login.*?About', '', text, flags=re.DOTALL)
        text = re.sub(r'©.*?\d{4}.*', '', text)
        text = re.sub(r'Privacy Policy.*Terms', '', text)
        
        return text.strip()
    
    
    def process_html_file(self, file_path: Path) -> Optional[Dict]:
        """Process a single HTML file and extract content"""
        try:
            # Read HTML file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract page title
            title = soup.title.string if soup.title else file_path.stem
            
            # Extract meta description
            meta_desc = ""
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag and meta_tag.get('content'):
                meta_desc = meta_tag['content'].strip()
            
            # Extract main content - THE SIMPLE WAY
            content = self.extract_main_content(soup)
            
            if not content or len(content) < 100:
                logger.warning(f"Short content: {file_path.name} ({len(content)} chars)")
                return None
            
            # Extract metadata
            category = self.extract_category_from_path(file_path)
            subcategory = self.extract_subcategory_from_path(file_path)
            
            # Create relative path for reference
            relative_path = file_path.relative_to(self.html_directory)
            
            page_data = {
                'file_path': str(relative_path),
                'title': title,
                'meta_description': meta_desc,
                'category': category,
                'subcategory': subcategory,
                'content': content,
                'content_length': len(content)
            }
            
            logger.info(f"✅ {file_path.name}: {len(content)} chars")
            return page_data
            
        except Exception as e:
            logger.error(f"Error: {file_path.name}: {e}")
            return None
    
    def process_all_files(self, max_files: int = None) -> List[Dict]:
        """Process all HTML files in the directory"""
        if not self.html_directory.exists():
            logger.error(f"Directory not found: {self.html_directory}")
            return []
        
        logger.info(f"Processing HTML files from: {self.html_directory}")
        
        # Find all HTML files recursively
        html_files = list(self.html_directory.rglob('*.html')) + list(self.html_directory.rglob('*.htm'))
        
        logger.info(f"Found {len(html_files)} HTML files")
        
        for file_path in html_files:
            if max_files and len(self.processed_data) >= max_files:
                break
            
            # Process file
            page_data = self.process_html_file(file_path)
            
            if page_data:
                self.processed_data.append(page_data)
        
        logger.info(f"✅ DONE: {len(self.processed_data)}/{len(html_files)} files processed")
        return self.processed_data
    
    def save_to_json(self, filepath: str):
        """Save processed data to JSON file"""
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to {filepath}")


if __name__ == "__main__":
    import json
    
    print("\nSTEP 1: HTML SCRAPING")
    
    # Initialize scraper
    scraper = LocalHTMLProcessor("sunmarke")
    
    # Process all HTML files
    print("\n Processing HTML files...")
    raw_data = scraper.process_all_files()
    
    # Save raw content
    output_file = settings.scraped_content_dir / 'sunmarke_raw.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    
    print(" SCRAPING COMPLETE!")
    print(f" Total pages scraped: {len(raw_data)}")
    print(f" Saved to: {output_file}")
    print(f" File size: {output_file.stat().st_size / 1024:.2f} KB")
    print("\n Next: python data_ingestion/chunker.py")