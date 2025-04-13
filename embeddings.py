import os
from typing import List, Optional
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI, AzureOpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

class EmbeddingProvider:
    def __init__(self):
        self.use_api = int(os.getenv('USE_EMBEDDING_API', '0'))
        self.api_type = os.getenv('EMBEDDING_API_TYPE', 'openai')
        self.api_key = os.getenv('EMBEDDING_API_KEY', '')
        self.api_base = os.getenv('EMBEDDING_API_BASE', '')
        self.api_model = os.getenv('EMBEDDING_API_MODEL', 'text-embedding-3-small')
        self.timeout = int(os.getenv('EMBEDDING_API_TIMEOUT', '30'))
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
        
        # 初始化 model 属性
        self.model = None
        self.model_name = os.getenv('EMBEDDING_MODEL', 'allenai/specter2_base')
        
        if self.use_api:
            self._setup_api_client()
    
    def _setup_api_client(self):
        """设置 API 客户端"""
        if not self.api_key:
            raise ValueError("API key is required when using API for embeddings")
            
        if self.api_type == 'openai':
            if self.api_base:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key)
        elif self.api_type == 'azure':
            if not self.api_base:
                raise ValueError("API base URL is required for Azure")
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=self.api_base
            )
        elif self.api_type == 'huggingface':
            self.client = None  # 使用 requests 直接调用
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def _get_local_model(self):
        """获取或加载本地模型"""
        if self.model is None:
            logger.info(f"Loading local model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Error loading model {self.model_name}: {e}")
                logger.warning("Falling back to allenai/specter2_base")
                self.model = SentenceTransformer('allenai/specter2_base')
        return self.model
    
    def _encode_api_batch(self, texts: List[str]) -> List[List[float]]:
        """使用 API 进行批量编码"""
        try:
            if self.api_type in ['openai', 'azure']:
                try:
                    response = self.client.embeddings.create(
                        model=self.api_model,
                        input=texts,
                        timeout=self.timeout
                    )
                    # Ensure we return a list of lists
                    embeddings = [embedding.embedding for embedding in response.data]
                    if not isinstance(embeddings[0], list):
                        embeddings = [embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding) for embedding in embeddings]
                    return embeddings
                except Exception as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        logger.error(f"API Response: {e.response.text}")
                    raise
                    
            elif self.api_type == 'huggingface':
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json={"inputs": texts},
                    timeout=self.timeout
                )
                response.raise_for_status()
                embeddings = response.json()
                # Ensure we return a list of lists
                if not isinstance(embeddings[0], list):
                    embeddings = [embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding) for embedding in embeddings]
                return embeddings
                
        except Exception as e:
            logger.warning(f"API embedding failed ({self.api_type}): {e}")
            logger.info("Falling back to local model")
            return self._get_local_model().encode(texts).tolist()
    
    def encode(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """编码文本列表"""
        if not texts:
            return []
            
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            if self.use_api:
                all_embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    if show_progress:
                        logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
                    embeddings = self._encode_api_batch(batch)
                    all_embeddings.extend(embeddings)
                return all_embeddings
            else:
                # Get embeddings from local model
                embeddings = self._get_local_model().encode(
                    texts,
                    show_progress_bar=show_progress,
                    convert_to_numpy=False
                )
                # Ensure we return a list of lists
                if not isinstance(embeddings, list):
                    embeddings = embeddings.tolist()
                elif not isinstance(embeddings[0], list):
                    embeddings = [embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding) for embedding in embeddings]
                return embeddings
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            # 如果出现错误，尝试使用本地模型
            logger.info("Attempting to use local model as fallback")
            try:
                embeddings = self._get_local_model().encode(
                    texts,
                    show_progress_bar=show_progress,
                    convert_to_numpy=False
                )
                # Ensure we return a list of lists
                if not isinstance(embeddings, list):
                    embeddings = embeddings.tolist()
                elif not isinstance(embeddings[0], list):
                    embeddings = [embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding) for embedding in embeddings]
                return embeddings
            except Exception as fallback_error:
                logger.error(f"Fallback embedding error: {fallback_error}")
                raise

# 全局实例
_global_provider = None

def get_embedding_provider() -> EmbeddingProvider:
    """获取全局嵌入提供者实例"""
    global _global_provider
    if _global_provider is None:
        _global_provider = EmbeddingProvider()
    return _global_provider 