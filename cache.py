import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from loguru import logger
import hashlib
from paper import ArxivPaper
import arxiv
from datetime import datetime, timedelta

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

class PaperCache:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PaperCache, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, cache_dir: str = ".cache", zotero_cache_expiry_days: int = 7, debug: bool = False):
        if not hasattr(self, 'initialized'):
            self.cache_dir = Path(cache_dir)
            self.zotero_cache_expiry_days = zotero_cache_expiry_days
            self.debug = debug
            self.zotero_cache_file = self.cache_dir / "zotero_corpus.json"
            self.tldr_cache_file = self.cache_dir / "tldr_cache.json"
            self.score_cache_file = self.cache_dir / "paper_scores.json"
            self.zotero_cache = {}
            self.tldr_cache = {}
            self.score_cache = {}
            self._load_cache()
            self.initialized = True

    def _load_cache(self):
        """Load existing cache from disk"""
        if self.zotero_cache_file.exists():
            try:
                with open(self.zotero_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Check if cache is expired
                    cache_time = datetime.fromisoformat(cache_data.get('timestamp', '1970-01-01'))
                    if datetime.now() - cache_time < timedelta(days=self.zotero_cache_expiry_days):
                        self.zotero_cache = cache_data.get('corpus', {})
                        logger.debug("Loaded Zotero corpus from cache")
                    else:
                        logger.debug("Zotero cache expired, will fetch fresh data")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load Zotero cache: {e}")

        if self.tldr_cache_file.exists():
            try:
                with open(self.tldr_cache_file, 'r') as f:
                    self.tldr_cache = json.load(f)
                logger.debug("Loaded TLDR from cache")
            except Exception as e:
                logger.warning(f"Failed to load TLDR cache: {e}")

        if self.score_cache_file.exists():
            try:
                with open(self.score_cache_file, 'r') as f:
                    self.score_cache = json.load(f)
                logger.debug("Loaded paper scores from cache")
            except Exception as e:
                logger.warning(f"Failed to load score cache: {e}")

    def _save_cache(self):
        """Save cache to disk"""
        if self.debug:
            if self.zotero_cache_file.exists():
                logger.debug("Debug mode: Cache files exist, skipping save")
                return
            logger.debug("Debug mode: Cache files not found, saving cache")
            
        with open(self.zotero_cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'corpus': self.zotero_cache
            }, f, cls=CustomEncoder)
        with open(self.tldr_cache_file, 'w') as f:
            json.dump(self.tldr_cache, f, cls=CustomEncoder)
        with open(self.score_cache_file, 'w') as f:
            json.dump(self.score_cache, f, cls=CustomEncoder)

    def get_embedding_cache_file(self, model: str) -> Path:
        """Get the cache file path for embeddings of a specific model"""
        # Handle None model name
        if model is None:
            model = 'default'
        
        # Create a safe model name by replacing special characters
        safe_model_name = model.replace('/', '_').replace('\\', '_')
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Return the cache file path
        return self.cache_dir / f"embeddings_{safe_model_name}.json"

    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache for a specific model"""
        cache_file = self.get_embedding_cache_file(model)
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                text_hash = hashlib.md5(text.encode()).hexdigest()
                return cache_data.get(text_hash)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def get_embeddings_batch(self, texts: List[str], model: str) -> Dict[str, Optional[List[float]]]:
        """Get embeddings for a batch of texts from cache"""
        cache_file = self.get_embedding_cache_file(model)
        if not cache_file.exists():
            if self.debug:
                logger.debug(f"Debug mode: Embedding cache file not found for model {model}")
            return {text: None for text in texts}
            
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                results = {}
                cache_hits = 0
                for text in texts:
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    if text_hash in cache_data:
                        cache_hits += 1
                    results[text] = cache_data.get(text_hash)
                if self.debug:
                    logger.debug(f"Debug mode: Cache hit rate: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
                return results
        except (json.JSONDecodeError, FileNotFoundError):
            if self.debug:
                logger.debug(f"Debug mode: Failed to load embedding cache for model {model}")
            return {text: None for text in texts}

    def save_embedding(self, text: str, embedding: List[float], model: str):
        """Save embedding to cache for a specific model"""
        if self.debug:
            logger.debug("Debug mode: Skipping embedding cache save")
            return
            
        cache_file = self.get_embedding_cache_file(model)
        cache_data = {}
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load embedding cache for model {model}, starting fresh")
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_data[text_hash] = embedding
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    def save_embeddings_batch(self, texts: List[str], embeddings: List[List[float]], model: str):
        """Save a batch of embeddings to cache"""
        if self.debug:
            logger.debug(f"Debug mode: Would save {len(texts)} embeddings to cache for model {model}")
            return
            
        cache_file = self.get_embedding_cache_file(model)
        cache_data = {}
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load embedding cache for model {model}, starting fresh")
        
        logger.info(f"Saving {len(texts)} embeddings to cache...")
        for text, embedding in zip(texts, embeddings):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_data[text_hash] = embedding
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        logger.info("Embeddings saved to cache successfully")

    def get_zotero_corpus(self, zotero_id: str) -> Optional[Dict]:
        """Get Zotero corpus from cache"""
        return self.zotero_cache.get(zotero_id)

    def save_zotero_corpus(self, zotero_id: str, corpus: Dict):
        """Save Zotero corpus to cache"""
        self.zotero_cache[zotero_id] = corpus
        self._save_cache()

    def get_tldr(self, paper_id: str) -> str:
        """获取论文的 TLDR
        
        Args:
            paper_id: 论文 ID
            
        Returns:
            str: TLDR 文本，如果不存在则返回 None
        """
        return self.tldr_cache.get(paper_id)
    
    def save_tldr(self, paper_id: str, tldr: str):
        """保存论文的 TLDR
        
        Args:
            paper_id: 论文 ID
            tldr: TLDR 文本
        """
        self.tldr_cache[paper_id] = tldr
        self._save_cache()

    def get_paper_score(self, paper_id: str) -> Optional[float]:
        """Get cached score for a paper"""
        return self.score_cache.get(paper_id)

    def save_paper_score(self, paper_id: str, score: float):
        """Save paper score to cache"""
        self.score_cache[paper_id] = score
        self._save_cache()

    def can_use_cached_tldr(self, paper_id: str, current_score: float) -> bool:
        """Check if we can use cached TLDR based on score similarity"""
        cached_score = self.get_paper_score(paper_id)
        if cached_score is None:
            return False
        # Consider scores similar if they differ by less than 5%
        return abs(cached_score - current_score) < 5.0

# Global model instance to prevent repeated downloads
_global_model = None
# Global cache instance
_global_cache = None

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str, model: str = None) -> List[float]:
    """Cache embeddings using LRU cache"""
    from sentence_transformers import SentenceTransformer
    from recommender import get_embedding_model
    
    global _global_cache
    
    # Get the appropriate model
    encoder = get_embedding_model(model)
    
    # Get or create cache instance
    if _global_cache is None:
        _global_cache = PaperCache()
    cache = _global_cache
    
    # Try to get from cache first
    cached_embedding = cache.get_embedding(text, model)
    if cached_embedding is not None:
        # Validate the cached embedding
        if not isinstance(cached_embedding, list):
            logger.warning(f"Invalid cached embedding type for text: {text[:50]}...")
            cached_embedding = None
        elif len(cached_embedding) == 0:
            logger.warning(f"Empty cached embedding for text: {text[:50]}...")
            cached_embedding = None
        elif not all(isinstance(x, (int, float)) for x in cached_embedding):
            logger.warning(f"Invalid cached embedding values for text: {text[:50]}...")
            cached_embedding = None
        else:
            # Get the expected embedding dimension from the model
            expected_dim = encoder.get_sentence_embedding_dimension()
            if len(cached_embedding) != expected_dim:
                logger.warning(f"Cached embedding dimension mismatch for text: {text[:50]}... (got {len(cached_embedding)}, expected {expected_dim})")
                cached_embedding = None
    
    if cached_embedding is not None:
        logger.debug(f"Using valid cached embedding for text: {text[:50]}...")
        return cached_embedding
        
    # Compute and cache if not found or invalid
    logger.debug(f"Computing new embedding for text: {text[:50]}...")
    embedding = encoder.encode(text).tolist()
    
    # Save to both file cache and LRU cache
    cache.save_embedding(text, embedding, model)
    return embedding

def process_papers_parallel(papers: List[arxiv.Result], max_workers: int = 4) -> List[ArxivPaper]:
    """Process papers in parallel"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_paper = {
            executor.submit(ArxivPaper, paper): paper 
            for paper in papers
        }
        
        processed_papers = []
        for future in as_completed(future_to_paper):
            try:
                processed_papers.append(future.result())
            except Exception as e:
                logger.error(f"Error processing paper: {e}")
                
    return processed_papers


class Cache:
    def __init__(self, fast_mode=False):
        self.fast_mode = fast_mode

    def get_cached_embedding(self, text: str) -> Optional[list[float]]:
        if self.fast_mode:
            return self.cache.get(text)
        
        cached_embedding = self.cache.get(text)
        # Validate the cached embedding
        if not isinstance(cached_embedding, list):
            logger.warning(f"Invalid cached embedding type for text: {text[:50]}...")
            cached_embedding = None
        elif len(cached_embedding) == 0:
            logger.warning(f"Empty cached embedding for text: {text[:50]}...")
            cached_embedding = None
        elif not all(isinstance(x, (int, float)) for x in cached_embedding):
            logger.warning(f"Invalid cached embedding values for text: {text[:50]}...")
            cached_embedding = None
        else:
            # Get the expected embedding dimension from the model
            expected_dim = encoder.get_sentence_embedding_dimension()
            if len(cached_embedding) != expected_dim:
                logger.warning(f"Cached embedding dimension mismatch for text: {text[:50]}... (got {len(cached_embedding)}, expected {expected_dim})")
                cached_embedding = None
        
        if cached_embedding is not None:
            logger.debug(f"Using valid cached embedding for text: {text[:50]}...")
            return cached_embedding
            
        # Compute and cache if not found or invalid
        logger.debug(f"Computing new embedding for text: {text[:50]}...")
        embedding = encoder.encode(text).tolist()
        
        # Save to both file cache and LRU cache
        cache.save_embedding(text, embedding, model)
        return embedding