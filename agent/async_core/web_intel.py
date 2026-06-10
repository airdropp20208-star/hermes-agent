"""
Web Intelligence — structured scraping, API discovery, web automation.
Agent can navigate, extract, and understand web content autonomously.
"""
import re
import json
import time
import logging
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


@dataclass
class WebPage:
    """A fetched web page."""
    url: str
    title: str = ""
    content: str = ""
    html: str = ""
    links: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    status_code: int = 0
    fetched_at: float = field(default_factory=time.time)
    content_type: str = ""


@dataclass
class ExtractedData:
    """Structured data extracted from a page."""
    source_url: str
    data_type: str  # article, product, api, code, table, list
    title: str = ""
    content: str = ""
    structured: Dict = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class APIEndpoint:
    """A discovered API endpoint."""
    url: str
    method: str = "GET"
    parameters: Dict = field(default_factory=dict)
    response_format: str = ""
    description: str = ""
    auth_required: bool = False


class WebExtractor:
    """
    Web content extraction with:
    - HTML parsing (title, meta, content)
    - Link extraction and normalization
    - Content cleaning (remove boilerplate)
    - Structured data extraction (JSON-LD, OpenGraph, tables)
    - API endpoint discovery
    - Code block extraction
    - Article content extraction
    - Metadata extraction
    """

    def extract_page(self, html: str, url: str = "") -> WebPage:
        """Extract structured content from HTML."""
        page = WebPage(url=url, html=html, status_code=200)

        # Title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
        if title_match:
            page.title = title_match.group(1).strip()

        # Meta tags
        for meta in re.finditer(r'<meta\s+([^>]*)/?>', html, re.IGNORECASE):
            attrs = meta.group(1)
            name = re.search(r'(?:name|property)=["\']([^"\']+)["\']', attrs)
            content = re.search(r'content=["\']([^"\']+)["\']', attrs)
            if name and content:
                page.metadata[name.group(1)] = content.group(1)

        # Links
        for link in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = link.group(1)
            if href.startswith(('http://', 'https://')):
                page.links.append(href)
            elif url:
                page.links.append(urljoin(url, href))

        # Content extraction (remove tags)
        content = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        page.content = content[:10000]

        return page

    def extract_article(self, html: str, url: str = "") -> ExtractedData:
        """Extract article content from HTML."""
        page = self.extract_page(html, url)

        # Try JSON-LD
        jsonld = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL)
        if jsonld:
            try:
                data = json.loads(jsonld.group(1))
                return ExtractedData(
                    source_url=url, data_type="article",
                    title=data.get("headline", page.title),
                    content=data.get("articleBody", page.content[:5000]),
                    structured=data,
                    confidence=0.9,
                )
            except json.JSONDecodeError:
                pass

        # Try OpenGraph
        og = {k: v for k, v in page.metadata.items() if k.startswith("og:")}
        if og:
            return ExtractedData(
                source_url=url, data_type="article",
                title=og.get("og:title", page.title),
                content=og.get("og:description", page.content[:2000]),
                structured=og, confidence=0.7,
            )

        # Fallback: clean text
        return ExtractedData(
            source_url=url, data_type="article",
            title=page.title, content=page.content[:5000],
            confidence=0.5,
        )

    def extract_tables(self, html: str) -> List[List[List[str]]]:
        """Extract tables from HTML as nested lists."""
        tables = []
        for table_match in re.finditer(r'<table[^>]*>(.*?)</table>', html, re.DOTALL | re.IGNORECASE):
            table_html = table_match.group(1)
            rows = []
            for row_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE):
                cells = []
                for cell_match in re.finditer(r'<t[dh][^>]*>(.*?)</t[dh]>', row_match.group(1), re.DOTALL | re.IGNORECASE):
                    cell = re.sub(r'<[^>]+>', '', cell_match.group(1)).strip()
                    cells.append(cell)
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables

    def extract_code_blocks(self, html: str) -> List[Dict]:
        """Extract code blocks from HTML."""
        blocks = []
        for match in re.finditer(r'<(?:pre|code)[^>]*class=["\'][^"\']*language-(\w+)[^"\']*["\'][^>]*>(.*?)</(?:pre|code)>', html, re.DOTALL | re.IGNORECASE):
            language = match.group(1)
            code = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            blocks.append({"language": language, "code": code})
        return blocks

    def extract_api_endpoints(self, html: str, url: str = "") -> List[APIEndpoint]:
        """Discover API endpoints from page content."""
        endpoints = []

        # Look for API URLs in JavaScript
        for match in re.finditer(r'(?:fetch|axios|XMLHttpRequest|\.get|\.post)\s*\(\s*["\']([^"\']+)["\']', html):
            api_url = match.group(1)
            if api_url.startswith(('/api', 'http')):
                endpoints.append(APIEndpoint(
                    url=urljoin(url, api_url) if not api_url.startswith('http') else api_url,
                    description="Discovered in JavaScript",
                ))

        # Look for Swagger/OpenAPI
        for match in re.finditer(r'(https?://[^"\']+(?:swagger|openapi|api-docs)[^"\']*)', html):
            endpoints.append(APIEndpoint(
                url=match.group(1),
                description="OpenAPI/Swagger spec",
            ))

        # REST patterns in links
        for match in re.finditer(r'href=["\']([^"\']*(?:/api/|/v\d+/)[^"\']*)["\']', html):
            api_url = match.group(1)
            endpoints.append(APIEndpoint(
                url=urljoin(url, api_url),
                description="REST API endpoint",
            ))

        return endpoints

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text (simple regex-based)."""
        entities = set()

        # URLs
        for m in re.finditer(r'https?://\S+', text):
            entities.add(m.group())

        # Email addresses
        for m in re.finditer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            entities.add(m.group())

        # File paths
        for m in re.finditer(r'(?:/[\w.-]+){2,}', text):
            entities.add(m.group())

        # Version numbers
        for m in re.finditer(r'\bv?\d+\.\d+(?:\.\d+)?\b', text):
            entities.add(m.group())

        # IP addresses
        for m in re.finditer(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text):
            entities.add(m.group())

        return list(entities)

    def clean_content(self, html: str) -> str:
        """Extract clean readable content from HTML."""
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def summarize_page(self, url: str, html: str) -> Dict:
        """Generate a structured summary of a page."""
        page = self.extract_page(html, url)
        tables = self.extract_tables(html)
        code = self.extract_code_blocks(html)
        apis = self.extract_api_endpoints(html, url)
        entities = self.extract_entities(page.content)

        return {
            "url": url,
            "title": page.title,
            "content_length": len(page.content),
            "links_count": len(page.links),
            "tables": len(tables),
            "code_blocks": len(code),
            "api_endpoints": len(apis),
            "entities": entities[:20],
            "metadata": page.metadata,
        }
