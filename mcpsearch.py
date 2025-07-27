import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime
from typing import Dict, List, Optional
import time


class MCPSearchFunction:
    def __init__(self, google_api_key: str, search_engine_id: str):
        self.api_key = google_api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def enhanced_search(self, query: str, num_results: int = 5, return_single_text: bool = True) -> Dict:
        """
        MCP-style search function that returns structured, processed content
        """
        try:
            # Step 1: Get search results from Google
            search_results = self._get_search_results(query, num_results)

            # Step 2: Process top 3 results only (for performance)
            processed_results = []
            top_results = search_results.get('items', [])[:3]

            for result in top_results:
                processed_result = self._process_search_result(result, query)
                if processed_result and processed_result.get('scores', {}).get('overall', 0) > 0.2:
                    processed_results.append(processed_result)

            print(f"🔎 Raw processed results count: {len(processed_results)}")
            for r in processed_results:
                print(f"- URL: {r['url']}")
                print(f"  Title: {r['title']}")
     

            # Step 3: Sort by relevance/quality
            processed_results.sort(key=lambda x: x.get('scores', {}).get('overall', 0), reverse=True)

            if return_single_text:
                # Combine top results into single comprehensive text
                combined_text = self._combine_results_to_text(processed_results[:3], query)
                return {
                    "query": query,
                    "combined_answer": combined_text,
                    "sources": [{"title": r["title"], "url": r["url"]} for r in processed_results[:3]],
                    "metadata": {
                        "processed_at": datetime.now().isoformat(),
                        "sources_used": len(processed_results[:3])
                    }
                }
            else:
                # Return structured response
                return {
                    "query": query,
                    "total_results": search_results.get('searchInformation', {}).get('totalResults', '0'),
                    "search_time": search_results.get('searchInformation', {}).get('searchTime', 0),
                    "processed_results": processed_results,
                    "metadata": {
                        "processed_at": datetime.now().isoformat(),
                        "results_processed": len(processed_results),
                        "quality_score": self._calculate_overall_quality(processed_results)
                    }
                }

        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "processed_results": [],
                "metadata": {"error_type": type(e).__name__}
            }

    def _get_search_results(self, query: str, num_results: int) -> Dict:
        """Get raw search results from Google Custom Search API"""
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)  # Google API limit
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def _process_search_result(self, result: Dict, query: str) -> Optional[Dict]:
        """Process individual search result with content extraction"""
        try:
            url = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')

            # Fetch and process full webpage content
            content_data = self._extract_webpage_content(url)
            if not content_data.get('content'):
                print(f"⚠️ No content extracted from {url}")
                return None

            # Calculate relevance and quality scores
            relevance_score = self._calculate_relevance(content_data.get('content', ''), query)
            quality_score = self._calculate_quality(content_data, url)

            return {
                "title": title,
                "url": url,
                "original_snippet": snippet,
                "content": {
                    "main_text": content_data.get('content', ''),
                    "headings": content_data.get('headings', []),
                    "key_points": content_data.get('key_points', []),
                    "word_count": content_data.get('word_count', 0)
                },
                "metadata": {
                    "domain": urlparse(url).netloc,
                    "content_type": content_data.get('content_type', 'unknown'),
                    "publish_date": content_data.get('publish_date'),
                    "last_modified": content_data.get('last_modified'),
                    "language": content_data.get('language', 'en')
                },
                "scores": {
                    "relevance": relevance_score,
                    "quality": quality_score,
                    "overall": ((relevance_score or 0) + (quality_score or 0)) / 2
                },
                "semantic_context": {
                    "topics": self._extract_topics(content_data.get('content', '')),
                    "entities": self._extract_entities(content_data.get('content', ''))
                }
            }

        except Exception as e:
            # Return minimal data if processing fails
            print(f"⚠️ Failed to process {result.get('link')}: {e}")
            return {
                "title": result.get('title', ''),
                "url": result.get('link', ''),
                "original_snippet": result.get('snippet', ''),
                "content": {"main_text": result.get('snippet', '')},
                "metadata": {"processing_error": str(e)},
                "scores": {"relevance": 0.1, "quality": 0.1, "overall": 0.1}
            }

    def _extract_webpage_content(self, url: str) -> Dict:
        """Extract and clean webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Fix: decode with fallback
            encoding = response.encoding if response.encoding else 'utf-8'
            content = response.content.decode(encoding, errors='replace')
            soup = BeautifulSoup(content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()

            # Extract main content
            main_content = self._find_main_content(soup)
            if not main_content:
                main_content = soup.get_text(separator=" ", strip=True)

            # Extract headings
            headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3']) if h.get_text().strip()]

            # Extract key points (from lists, bold text, etc.)
            key_points = self._extract_key_points(soup)

            # Get metadata
            publish_date = self._extract_publish_date(soup)
            content_type = self._determine_content_type(soup, url)

            return {
                "content": main_content,
                "headings": headings,
                "key_points": key_points,
                "word_count": len(main_content.split()),
                "content_type": content_type,
                "publish_date": publish_date,
                "last_modified": response.headers.get('Last-Modified'),
                "language": soup.get('lang', 'en')
            }

        except Exception as e:
            return {"content": "", "error": str(e)}



    def _find_main_content(self, soup: BeautifulSoup) -> str:
        """Find and extract main content from webpage"""
        # Try common content containers
        content_selectors = [
            'article', 'main', '[role="main"]',
            '.content', '.post-content', '.entry-content',
            '.article-body', '.story-body'
        ]

        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Minimum content length
                    return text

        # Fallback: get text from body, excluding common noise
        if soup.body:
            # Remove common noise elements
            for noise in soup.body(['nav', 'header', 'footer', 'sidebar', 'menu']):
                noise.decompose()
            return soup.body.get_text(separator=' ', strip=True)

        return soup.get_text(separator=' ', strip=True)

    def _extract_key_points(self, soup: BeautifulSoup) -> List[str]:
        """Extract key points from lists and emphasized text"""
        key_points = []

        # Extract from lists
        for ul in soup.find_all(['ul', 'ol']):
            for li in ul.find_all('li'):
                text = li.get_text().strip()
                if text and len(text) > 10:
                    key_points.append(text)

        # Extract from bold/strong text
        for element in soup.find_all(['strong', 'b']):
            text = element.get_text().strip()
            if text and len(text) > 10 and len(text) < 200:
                key_points.append(text)

        return key_points[:10]  # Limit to top 10

    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publish date from webpage"""
        date_selectors = [
            'time[datetime]', '[property="article:published_time"]',
            '.publish-date', '.post-date', '.date'
        ]

        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get('datetime') or element.get('content') or element.get_text().strip()

        return None

    def _determine_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """Determine the type of content"""
        if any(indicator in url.lower() for indicator in ['blog', 'article', 'news']):
            return 'article'
        elif 'wikipedia.org' in url:
            return 'encyclopedia'
        elif any(soup.find(tag) for tag in ['article']):
            return 'article'
        elif soup.find('table'):
            return 'data'
        else:
            return 'webpage'

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score based on query terms in content"""
        if not content:
            return 0.0

        query_terms = query.lower().split()
        content_lower = content.lower()

        matches = sum(1 for term in query_terms if term in content_lower)
        return min(matches / len(query_terms), 1.0)

    def _calculate_quality(self, content_data: Dict, url: str) -> float:
        """Calculate quality score based on various factors"""
        score = 0.0

        # Content length (more content = higher quality, up to a point)
        word_count = content_data.get('word_count', 0)
        if word_count > 100:
            score += min(word_count / 1000, 0.5)

        # Domain authority (simple heuristic)
        domain = urlparse(url).netloc
        if any(trusted in domain for trusted in ['wikipedia', 'gov', 'edu', 'reuters', 'bbc']):
            score += 0.3

        # Structure indicators
        if content_data.get('headings'):
            score += 0.1
        if content_data.get('key_points'):
            score += 0.1

    def _combine_results_to_text(self, results: List[Dict], query: str) -> str:
        """Combine multiple search results into single comprehensive text"""
        if not results:
            return "No relevant information found."

        combined_sections = []

        for i, result in enumerate(results, 1):
            content = result.get('content', {}).get('main_text', '')
            title = result.get('title', '')

            if content:
                # Take most relevant portion (first 500 words)
                words = content.split()
                relevant_content = ' '.join(words[:500])

                section = f"{relevant_content}\n\nSource: {title}\nLink: {result.get('url', '')}"
                combined_sections.append(section)

        # Join sections with clear separators
        combined_text = "\n\n---\n\n".join(combined_sections)

        return combined_text

    def _calculate_overall_quality(self, results: List[Dict]) -> float:
        """Calculate overall quality of search results"""
        if not results:
            return 0.0

        total_score = sum(r.get('scores', {}).get('overall', 0) for r in results)
        return total_score / len(results)

    def _extract_topics(self, content: str) -> List[str]:
        """Simple topic extraction (you could use NLP libraries here)"""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Return top frequent words as topics
        return [word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]

    def _extract_entities(self, content: str) -> List[str]:
        """Simple entity extraction (capitalized words)"""
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        return list(set(entities))[:10]



# Usage example:
# searcher = MCPSearchFunction("your_api_key", "your_search_engine_id")
# results = searcher.enhanced_search("artificial intelligence trends 2024")
# print(results)
