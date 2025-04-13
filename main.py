import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange, tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser
from cache import PaperCache, process_papers_parallel
from typing import List, Dict
from datetime import datetime
import json
import requests

# Initialize cache
paper_cache = PaperCache()

# 定义缓存目录和文件
CACHE_DIR = '.cache'
ARXIV_CACHE_FILE = os.path.join(CACHE_DIR, 'arxiv_papers.json')

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

def get_zotero_corpus(id: str, key: str) -> List[Dict]:
    # Check cache first
    if cached_corpus := paper_cache.get_zotero_corpus(id):
        logger.debug(f"Using cached Zotero corpus for user {id}")
        return cached_corpus

    logger.debug(f"Fetching fresh Zotero corpus for user {id}")
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']: c for c in collections}
    
    # Get all items
    all_items = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    logger.info(f"Retrieved {len(all_items)} total papers from Zotero")
    
    # Analyze and filter items
    filtered_items = []
    no_abstract = 0
    empty_abstract = 0
    valid_items = 0
    
    for item in all_items:
        abstract = item['data'].get('abstractNote', '')
        if abstract is None:
            no_abstract += 1
            logger.debug(f"Paper filtered out - No abstract: {item['data'].get('title', 'Untitled')}")
        elif abstract.strip() == '':
            empty_abstract += 1
            logger.debug(f"Paper filtered out - Empty abstract: {item['data'].get('title', 'Untitled')}")
        else:
            valid_items += 1
            filtered_items.append(item)
    
    # Log filtering statistics
    logger.info(f"Papers filtered out due to missing abstract: {no_abstract}")
    logger.info(f"Papers filtered out due to empty abstract: {empty_abstract}")
    logger.info(f"Papers with valid abstracts: {valid_items}")
    
    def get_collection_path(col_key: str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    
    # Add collection paths to valid items
    for c in filtered_items:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths

    # Save to cache
    paper_cache.save_zotero_corpus(id, filtered_items)
    return filtered_items

def filter_corpus(corpus: List[Dict], pattern: str) -> List[Dict]:
    _, filename = mkstemp()
    with open(filename, 'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus

def get_arxiv_paper(query: str, debug: bool = False) -> List[ArxivPaper]:
    """Get papers from arXiv API"""
    # 检查缓存文件是否存在
    cache_exists = os.path.exists(ARXIV_CACHE_FILE)
    
    # 在非调试模式下，如果缓存存在则使用缓存
    if cache_exists:
        logger.debug(f"Found cache file: {ARXIV_CACHE_FILE}")
        try:
            with open(ARXIV_CACHE_FILE, 'r', encoding='utf-8') as f:
                papers_data = json.load(f)
                # 在调试模式下提示用户正在使用缓存
                if debug:
                    logger.debug("Debug mode: Using existing cache for verification")
                else:
                    logger.debug("Using cached arXiv papers")
                return [ArxivPaper(paper_data) for paper_data in papers_data]
        except Exception as e:
            logger.warning(f"Failed to load cache from {ARXIV_CACHE_FILE}: {e}")
            # 如果缓存文件损坏，删除它
            try:
                os.remove(ARXIV_CACHE_FILE)
                logger.debug("Removed corrupted cache file")
            except Exception as e:
                logger.warning(f"Failed to remove corrupted cache file: {e}")
    else:
        logger.debug("No cache file found")
    
    # 如果没有缓存或者缓存无效，从 API 获取数据
    logger.debug(f"Fetching papers from arXiv API for query: {query}")
    
    # 构建 API 查询
    categories = query.split("+")
    combined_query = " OR ".join([f"cat:{cat.strip()}" for cat in categories])
    logger.debug(f"Combined API query: {combined_query}")
    
    # 设置搜索参数并获取论文
    client = arxiv.Client()
    search = arxiv.Search(
        query=combined_query,
        max_results=50,  # 获取50篇论文
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    # 获取论文
    papers = []
    try:
        results = client.results(search)
        for result in results:
            paper = ArxivPaper(result)
            papers.append(paper)
    except Exception as e:
        logger.error(f"Error fetching papers from arXiv: {e}")
        return []
        
    logger.info(f"Retrieved {len(papers)} papers from arXiv.")
    
    # 如果没有找到论文，提前返回
    if not papers:
        logger.warning("No papers found.")
        return []

    # 将论文对象转换为可序列化的字典
    papers_data = []
    for paper in papers:
        paper_dict = {
            'title': paper.title,
            'summary': paper.summary,
            'authors': paper.authors,
            'published': paper.published.isoformat() if paper.published else None,
            'updated': paper.updated.isoformat() if paper.updated else None,
            'entry_id': paper.entry_id,
            'pdf_url': paper.pdf_url,
            'primary_category': paper.primary_category,
            'categories': paper.categories,
            'comment': paper.comment
        }
        papers_data.append(paper_dict)
    
    # 总是保存缓存，即使在调试模式下
    try:
        with open(ARXIV_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved {len(papers_data)} papers to cache: {ARXIV_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {ARXIV_CACHE_FILE}: {e}")
    
    return papers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast-cache', action='store_true',
                       help='Load cache in bulk without individual validation')
    return parser.parse_args()

parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    add_argument('--fast-cache', type=bool, help='Load cache in bulk without validation', default=False)
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='Send email even if no papers found',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    add_argument(
        "--no-email-send",
        type=bool,
        help="Disable email sending and output content to file",
        default=False,
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    # Initialize cache with debug mode
    paper_cache = PaperCache(debug=args.debug)

    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    # 在发送邮件之前，对论文进行排序并只保留前10篇
    if len(papers) > 0:
        total_papers = len(papers)
        # 按相关性得分排序
        papers.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
        # 只保留前10篇最相关的论文
        papers = papers[:10]
        
        logger.info(f"Selected top 10 most relevant papers from {total_papers} papers.")
        html = render_email(papers)
        if args.no_email_send:
            logger.info("Email sending disabled. Content has been saved to output-email.html")
        else:
            logger.info("Sending email...")
            send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
            logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")
    else:
        logger.warning("No papers found.")
        html = render_email([])
        if not args.no_email_send:
            send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)

