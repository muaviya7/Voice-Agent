"""
Content processor for scraped data
Cleans and structures content for embedding
"""
import re
from typing import List, Dict
from utils.logger import logger


class ContentProcessor:
    """Process and clean scraped content"""
    
    def __init__(self):
        self.min_content_length = 50  # Lowered for minified HTML
        self.max_content_length = 10000
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        # Remove empty lines
        lines = [line for line in lines if line]
        return '\n'.join(lines)
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters while keeping punctuation"""
        # Keep alphanumeric, basic punctuation, and common symbols
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"£$€%&@]', ' ', text)
        return text
    
    def fix_common_issues(self, text: str) -> str:
        """Fix common text issues from HTML extraction"""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        
        # Fix multiple punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        
        # Remove text like "Cookie Policy", "Terms & Conditions" at the end
        text = re.sub(r'(Cookie Policy|Terms and Conditions|Privacy Policy).*$', '', text, flags=re.IGNORECASE)
        
        return text
    
    def clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        if not content:
            return ""
        
        # Apply cleaning steps
        content = self.remove_extra_whitespace(content)
        content = self.remove_special_characters(content)
        content = self.fix_common_issues(content)
        content = self.remove_extra_whitespace(content)  # Final cleanup
        
        return content.strip()
    
    def is_valid_content(self, content: str) -> bool:
        """Check if content is valid and meaningful"""
        if not content:
            return False
        
        # Check length
        if len(content) < self.min_content_length:
            return False
        
        # Check if content has actual words (not just numbers/symbols)
        words = re.findall(r'\b[a-zA-Z]+\b', content)
        if len(words) < 10:  # At least 10 words (lowered from 20)
            return False
        
        return True
    
    def process_page(self, page_data: Dict) -> Dict:
        """Process a single page's data"""
        try:
            # Clean content
            cleaned_content = self.clean_content(page_data.get('content', ''))
            
            # Validate
            if not self.is_valid_content(cleaned_content):
                logger.warning(f"Invalid content for URL: {page_data.get('url')}")
                return None
            
            # Update page data with cleaned content
            processed_data = page_data.copy()
            processed_data['content'] = cleaned_content
            processed_data['content_length'] = len(cleaned_content)
            processed_data['word_count'] = len(cleaned_content.split())
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing page {page_data.get('url')}: {str(e)}")
            return None
    
    def process_batch(self, pages: List[Dict]) -> List[Dict]:
        """Process multiple pages"""
        processed_pages = []
        
        for page in pages:
            processed = self.process_page(page)
            if processed:
                processed_pages.append(processed)
        
        logger.info(f"Processed {len(processed_pages)}/{len(pages)} pages successfully")
        return processed_pages
    
    def deduplicate_content(self, pages: List[Dict]) -> List[Dict]:
        """Remove duplicate content based on URL and content similarity"""
        seen_urls = set()
        seen_content = set()
        unique_pages = []
        
        for page in pages:
            url = page.get('url')
            content = page.get('content', '')
            
            # Check URL duplication
            if url in seen_urls:
                logger.info(f"Duplicate URL found: {url}")
                continue
            
            # Check content similarity (first 500 chars as fingerprint)
            content_fingerprint = content[:500]
            if content_fingerprint in seen_content:
                logger.info(f"Duplicate content found for: {url}")
                continue
            
            seen_urls.add(url)
            seen_content.add(content_fingerprint)
            unique_pages.append(page)
        
        logger.info(f"Kept {len(unique_pages)}/{len(pages)} unique pages")
        return unique_pages
