import requests
import arxiv
import PyPDF2
import io
import os
import json
import hashlib
from typing import Optional, Dict, Tuple
from urllib.parse import quote

class PaperContentFetcher:
    def __init__(self):
        self.cache_dir = "paper_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def fetch_paper_content(self, paper_details: Dict) -> Tuple[Optional[str], str]:
        """
        Fetch full paper content from various sources
        
        Args:
            paper_details: Dictionary containing paper metadata from Semantic Scholar
            
        Returns:
            Tuple of (content, source) where content is the extracted text and source indicates where it came from
        """
        paper_id = paper_details.get("paperId", "")
        
        # Check cache first
        cached_content = self._get_cached_content(paper_id)
        if cached_content:
            return cached_content, "cache"
        
        # Try different methods to get paper content
        external_ids = paper_details.get("externalIds", {})
        
        # Method 1: Check if Semantic Scholar has openAccessPdf
        open_access_pdf = paper_details.get("openAccessPdf")
        if open_access_pdf and open_access_pdf.get("url"):
            content = self._fetch_pdf_content(open_access_pdf["url"])
            if content:
                self._cache_content(paper_id, content)
                return content, "semantic_scholar_pdf"
        
        # Method 2: Try ArXiv if available
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id:
            content = self._fetch_arxiv_content(arxiv_id)
            if content:
                self._cache_content(paper_id, content)
                return content, "arxiv"
        
        # Method 3: Try Unpaywall with DOI
        doi = external_ids.get("DOI")
        if doi:
            content = self._fetch_unpaywall_content(doi)
            if content:
                self._cache_content(paper_id, content)
                return content, "unpaywall"
        
        # Method 4: Try CORE API
        content = self._fetch_core_content(paper_details)
        if content:
            self._cache_content(paper_id, content)
            return content, "core"
        
        # If all methods fail, return None
        return None, "not_found"
    
    def _fetch_pdf_content(self, pdf_url: str) -> Optional[str]:
        """Download PDF and extract text content"""
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error fetching PDF from {pdf_url}: {e}")
        return None
    
    def _fetch_arxiv_content(self, arxiv_id: str) -> Optional[str]:
        """Fetch paper content from ArXiv"""
        try:
            # Clean the ArXiv ID - remove any prefixes
            if arxiv_id.startswith("arXiv:"):
                arxiv_id = arxiv_id[6:]
            
            print(f"    - Attempting to fetch ArXiv paper: {arxiv_id}")
            
            # Search for the paper
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Download PDF
            pdf_url = paper.pdf_url
            print(f"    - Found PDF URL: {pdf_url}")
            return self._fetch_pdf_content(pdf_url)
        except StopIteration:
            print(f"    - ArXiv paper {arxiv_id} not found")
        except Exception as e:
            print(f"    - Error fetching from ArXiv {arxiv_id}: {e}")
        return None
    
    def _fetch_unpaywall_content(self, doi: str) -> Optional[str]:
        """Fetch paper content using Unpaywall API"""
        try:
            # Unpaywall API endpoint
            url = f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?email=research@example.com"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Check if there's an open access PDF
                best_oa_location = data.get("best_oa_location")
                if best_oa_location and best_oa_location.get("url_for_pdf"):
                    pdf_url = best_oa_location["url_for_pdf"]
                    return self._fetch_pdf_content(pdf_url)
                
                # Try other OA locations
                oa_locations = data.get("oa_locations", [])
                for location in oa_locations:
                    if location.get("url_for_pdf"):
                        content = self._fetch_pdf_content(location["url_for_pdf"])
                        if content:
                            return content
        except Exception as e:
            print(f"Error fetching from Unpaywall {doi}: {e}")
        return None
    
    def _fetch_core_content(self, paper_details: Dict) -> Optional[str]:
        """Fetch paper content from CORE API"""
        try:
            # CORE API search by title
            title = paper_details.get("title", "")
            if not title:
                return None
                
            url = "https://api.core.ac.uk/v3/search/works"
            params = {
                "q": f'title:"{title}"',
                "limit": 1
            }
            
            # Note: CORE requires API key for production use
            # headers = {"Authorization": "Bearer YOUR_CORE_API_KEY"}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results and len(results) > 0:
                    paper = results[0]
                    
                    # Check if full text is available
                    if paper.get("fullText"):
                        return paper["fullText"]
                    
                    # Check for downloadUrl
                    download_url = paper.get("downloadUrl")
                    if download_url:
                        return self._fetch_pdf_content(download_url)
        except Exception as e:
            print(f"Error fetching from CORE: {e}")
        return None
    
    def _get_cached_content(self, paper_id: str) -> Optional[str]:
        """Retrieve cached paper content"""
        cache_file = os.path.join(self.cache_dir, f"{paper_id}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading cache for {paper_id}: {e}")
        return None
    
    def _cache_content(self, paper_id: str, content: str):
        """Cache paper content to disk"""
        cache_file = os.path.join(self.cache_dir, f"{paper_id}.txt")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Error caching content for {paper_id}: {e}")