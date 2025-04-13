import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime
from cache import get_cached_embedding, process_papers_parallel, PaperCache
from typing import List, Dict
from loguru import logger
from tqdm import tqdm
import signal
import sys
import os
from embeddings import get_embedding_provider

# Global model instance to prevent repeated downloads
_global_model = None
# Global cache instance
_global_cache = None
# Paper cache instance
paper_cache = PaperCache()

def signal_handler(signum, frame):
    logger.warning("\nKeyboard interrupt detected. Stopping embedding computation...")
    sys.exit(1)

def get_embedding_model(model_name: str = None) -> SentenceTransformer:
    """Get the appropriate embedding model based on configuration"""
    global _global_model
    
    if _global_model is not None:
        return _global_model
        
    if model_name is None:
        model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
    
    logger.info(f"Using embedding model: {model_name}")
    
    try:
        # First try to load the model
        _global_model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded model: {model_name}")
        return _global_model
    except Exception as e:
        error_msg = str(e)
        if "No sentence-transformers model found" in error_msg:
            logger.error(f"Model {model_name} not found in sentence-transformers library")
            logger.warning("Available models in sentence-transformers:")
            logger.warning("- BAAI/bge-m3 (default)")
            logger.warning("- BAAI/bge-large-en")
            logger.warning("- BAAI/bge-base-en")
            logger.warning("- all-MiniLM-L6-v2")
            logger.warning("- all-mpnet-base-v2")
            logger.warning("Falling back to BAAI/bge-m3")
            _global_model = SentenceTransformer('BAAI/bge-m3')
            return _global_model
        elif "PEFT" in error_msg:
            logger.error("PEFT error: Please install sentence-transformers with PEFT support")
            logger.warning("Try: pip install sentence-transformers[peft]")
            sys.exit(1)
        else:
            logger.error(f"Error loading model: {error_msg}")
            sys.exit(1)

def get_embeddings_batch(texts: List[str], model: str = None) -> List[List[float]]:
    """Get embeddings for a batch of texts"""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = PaperCache()
    
    # First try to get from cache
    cached_results = _global_cache.get_embeddings_batch(texts, model)
    uncached_texts = [text for text, embedding in cached_results.items() if embedding is None]
    
    if not uncached_texts:
        logger.info(f"Using cached embeddings for all {len(texts)} texts")
        return [cached_results[text] for text in texts]
    
    # Get embeddings for uncached texts
    logger.info(f"Found {len(uncached_texts)} uncached texts out of {len(texts)} total texts")
    provider = get_embedding_provider()
    if provider.use_api:
        logger.info(f"Using {provider.api_type} API for embeddings (model: {provider.api_model})")
    else:
        logger.info(f"Using local embedding model: {provider.model_name}")
    
    # Process in batches with progress bar
    all_embeddings = []
    for i in tqdm(range(0, len(uncached_texts), provider.batch_size), desc="Computing embeddings"):
        batch = uncached_texts[i:i + provider.batch_size]
        embeddings = provider.encode(batch, show_progress=False)
        all_embeddings.extend(embeddings)
    
    # Save to cache
    _global_cache.save_embeddings_batch(uncached_texts, all_embeddings, model)
    
    # Combine results
    results = []
    for text in texts:
        if text in cached_results and cached_results[text] is not None:
            results.append(cached_results[text])
        else:
            idx = uncached_texts.index(text)
            results.append(all_embeddings[idx])
    
    return results

def rerank_paper(candidate: List[ArxivPaper], corpus: List[Dict], model: str = None) -> List[ArxivPaper]:
    """Rerank papers based on similarity to corpus"""
    # Set up signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    
    if not candidate or not corpus:
        return candidate
    
    # Prepare texts for embedding
    candidate_texts = [f"{paper.title} {paper.summary}" for paper in candidate]
    corpus_texts = [f"{paper['data']['title']} {paper['data']['abstractNote']}" for paper in corpus]
    
    # Get embeddings in batch
    logger.info(f"Getting embeddings for {len(candidate_texts)} candidate papers and {len(corpus_texts)} corpus papers...")
    candidate_embeddings = get_embeddings_batch(candidate_texts, model)
    corpus_embeddings = get_embeddings_batch(corpus_texts, model)
    
    # Convert to numpy arrays
    candidate_embeddings = np.array(candidate_embeddings)
    corpus_embeddings = np.array(corpus_embeddings)
    
    # Compute similarities
    logger.info("Computing similarities between papers...")
    similarities = np.dot(candidate_embeddings, corpus_embeddings.T)
    max_similarities = np.max(similarities, axis=1)
    
    # Normalize scores to 0-100 range
    min_score = np.min(max_similarities)
    max_score = np.max(max_similarities)
    if max_score > min_score:
        normalized_scores = (max_similarities - min_score) / (max_score - min_score) * 100
    else:
        normalized_scores = np.ones_like(max_similarities) * 50  # If all scores are the same, set to 50
    
    # Sort papers by similarity and assign scores
    sorted_indices = np.argsort(normalized_scores)[::-1]
    sorted_papers = []
    for idx in sorted_indices:
        paper = candidate[idx]
        paper.score = normalized_scores[idx]
        # Save score to cache
        paper_cache.save_paper_score(paper.get_short_id(), paper.score)
        sorted_papers.append(paper)
    
    return sorted_papers