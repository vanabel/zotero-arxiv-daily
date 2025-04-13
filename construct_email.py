from paper import ArxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
from loguru import logger
from typing import List
import time
from llm import get_llm
from cache import PaperCache
import os
import re
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
import traceback
from urllib.parse import urlencode, quote
import logging
import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG")
# 初始化缓存
paper_cache = PaperCache()

framework = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            padding: 0 20px;
            background: #f5f5f5;
            color: #333;
        }
        ol {
            padding: 0;
            margin: 0;
            list-style-position: outside;
            padding-left: 2em;
        }
        .paper {
            padding: 20px 0;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
            margin-left: 0;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }
        .title {
            font-size: 1.2em;
            font-weight: bold;
            flex: 1;
            margin-right: 20px;
            padding-left: 0;
        }
        .title a {
            color: #2c3e50;
            text-decoration: none;
        }
        .title a:hover {
            text-decoration: underline;
        }
        .rate {
            white-space: nowrap;
            color: #666;
        }
        .stars {
            color: #f1c40f;
            letter-spacing: 2px;
        }
        .similarity {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        .meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .authors {
            color: #666;
            font-style: italic;
            flex: 1;
            margin-right: 20px;
        }
        .links {
            white-space: nowrap;
        }
        .links a {
            display: inline-block;
            padding: 4px 8px;
            margin-left: 8px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .links a:hover {
            background: #2980b9;
        }
        .abstract {
            line-height: 1.5;
            color: #444;
        }
        .empty-result {
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .empty-result h2 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .empty-result p {
            color: #666;
        }
    </style>
</head>
<body>
    <h1>__TITLE__</h1>
    <ol>
        __PAPERS__
    </ol>
</body>
</html>
"""

def get_empty_html(lang: str = "English") -> str:
    """Generate empty result HTML"""
    if lang.lower() == "chinese":
        return """
        <div class="empty-result">
            <h2>没有找到相关论文</h2>
            <p>今天没有与您的兴趣相关的新论文。请明天再来查看！</p>
        </div>
        """
    else:
        return """
        <div class="empty-result">
            <h2>No related papers found</h2>
            <p>No new papers related to your interests today. Please check back tomorrow!</p>
        </div>
        """

def normalize_math_delimiters(text: str) -> str:
    """Normalize all math delimiters to $ and $$ format"""
    logger.debug(f"Normalizing math delimiters in text: {text}")
    # Convert \( and \) to $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    # Convert \[ and \] to $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
    return text

def process_math_for_email(text: str) -> str:
    """Process math expressions in text and convert to images for email"""
    logger.debug(f"Processing math for email in text: {text}")
    
    # Math pattern to match both inline ($...$) and display ($$...$$) math
    math_pattern = r'(\$\$.*?\$\$|\$(?!\$).*?(?<!\$)\$)'
    parts = re.split(math_pattern, text)
    result = []
    
    for part in parts:
        if part and (part.startswith('$') or part.startswith('$$')):
            # Extract the math content without delimiters
            is_display = part.startswith('$$')
            math_content = part[2:-2] if is_display else part[1:-1]
            
            # Keep single backslashes for LaTeX commands
            math_content = math_content.replace('\\\\', '\\')
            
            logger.debug(f"Found math expression: {math_content} (display: {is_display})")
            
            try:
                # Prepare the parameters with proper escaping
                params = {'tex': math_content if not is_display else '\\displaystyle ' + math_content}
                # Make the request with proper URL encoding and error handling
                response = requests.get('https://latex.vanabel.cn/api/', params=params, timeout=10)
                
                if response.status_code == 200 and response.content:
                    logger.debug(f"Successfully rendered math: {math_content}")
                    if is_display:
                        result.append(
                            f'<div style="text-align: center; margin: 1em 0;">'
                            f'<img src="https://latex.vanabel.cn/api/?{urlencode(params)}" '
                            f'alt="{math_content}" style="max-width: 100%; transform: scale(1); transform-origin: center;"/>'
                            f'</div>'
                        )
                    else:
                        result.append(
                            f'<img src="https://latex.vanabel.cn/api/?{urlencode(params)}" '
                            f'alt="{math_content}" '
                            f'style="vertical-align: bottom; transform: scale(1); transform-origin: center; display: inline-block; margin: 0 0.1em;"/>'
                        )
                else:
                    logger.error(f"Failed to fetch math image: {response.status_code} for {math_content}")
                    result.append(f'<code style="font-family: monospace;">{part}</code>')
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error while rendering math expression: {math_content}, error: {e}")
                result.append(f'<code style="font-family: monospace;">{part}</code>')
            except Exception as e:
                logger.error(f"Failed to render math expression: {math_content}, error: {e}")
                result.append(f'<code style="font-family: monospace;">{part}</code>')
        else:
            result.append(part)
    return ''.join(result)

def process_math(text: str) -> str:
    """Process math expressions in text and keep them as LaTeX"""
    logger.debug(f"Processing math in text: {text}")
    # First normalize all math delimiters to $ and $$
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
    # Clean up any remaining escaped characters
    text = text.replace('\\', '')
    return text

def get_block_html(paper: ArxivPaper, lang: str = "English") -> str:
    """Generate HTML block for a single paper"""
    # Build similarity info HTML
    if lang.lower() == "chinese":
        rate_text = "相关度："
        pdf_text = "PDF"
        similarity_text = "相似度："
    else:
        rate_text = "Relevance: "
        pdf_text = "PDF"
        similarity_text = "Similarity: "
    
    similarity_html = f'<div class="similarity">{similarity_text}{paper.score:.2f}</div>' if hasattr(paper, 'score') else ''
    
    # Process title and abstract to handle math expressions
    title = paper.title
    abstract = paper.tldr
    
    logger.debug(f"Processing paper {paper.get_short_id()}")
    logger.debug(f"Original title: {title}")
    logger.debug(f"Original abstract: {abstract}")
    
    # Normalize math delimiters in title and abstract
    title = re.sub(r'\\\((.*?)\\\)', r'$\1$', title)
    title = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', title)
    abstract = re.sub(r'\\\((.*?)\\\)', r'$\1$', abstract)
    abstract = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', abstract)
    
    logger.debug(f"Normalized title: {title}")
    logger.debug(f"Normalized abstract: {abstract}")
    
    # Process math in title and abstract
    title = process_math(title)
    abstract = process_math(abstract)
    
    logger.debug(f"Processed title: {title}")
    logger.debug(f"Processed abstract: {abstract}")
    
    return f"""
    <li>
    <div class="paper">
        <div class="header">
            <div class="title">
                    <a href="https://arxiv.org/abs/{paper.get_short_id()}" target="_blank">{title}</a>
                </div>
                <div class="rate">
                    <div>{rate_text}<span class="stars">{get_stars(paper.score)}</span></div>
                    {similarity_html}
                </div>
        </div>
        <div class="meta">
                <div class="authors">{', '.join(paper.authors)}</div>
            <div class="links">
                    <a href="{paper.pdf_url}" target="_blank">{pdf_text}</a>
                </div>
            </div>
            <div class="abstract">{abstract}</div>
        </div>
    </li>
    """

def get_stars(score: float) -> str:
    """根据相关性分数生成星星评分
    
    Args:
        score: 相关性分数 (0-1)
        
    Returns:
        str: 星星评分字符串
    """
    # 将分数转换为 1-5 星
    stars = min(5, max(1, round(score * 5)))
    # 使用 HTML 实体编码的星星
    return "★" * stars + "☆" * (5 - stars)

def save_html_output(html: str, output_path: str = "output-email.html") -> None:
    """Save the HTML content to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Email content saved to {output_path}")
        logger.info(f"You can open this file in a web browser to preview the email content")
    except Exception as e:
        logger.error(f"Failed to save HTML output: {e}")

def render_email(papers: List[ArxivPaper], lang: str = "English", for_email: bool = True) -> str:
    """Render email content
    
    Args:
        papers: List of papers to render
        lang: Language for the email (default: "English")
        for_email: Whether to convert math to images (True) or keep as LaTeX (False)
    """
    if not papers:
        return get_empty_html(lang)
    
    # Add MathJax script for HTML output
    mathjax_script = """
    <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$','$']],
                displayMath: [['$$','$$']],
                processEscapes: true
            }
        });
    </script>
    """
    
    # Initialize LLM for TLDR generation
    llm = get_llm()
    if isinstance(llm.llm, OpenAI):
        logger.info(f"Using OpenAI API for TLDR generation (model: {llm.model})")
    else:
        logger.info(f"Using local LLM for TLDR generation: {llm.model}")
    
    # Generate TLDR for each paper with progress bar
    logger.info("Generating TLDR summaries...")
    for paper in tqdm(papers, desc="Generating TLDRs"):
        # Try to get cached TLDR first
        cached_tldr = paper_cache.get_tldr(paper.get_short_id())
        if cached_tldr:
            paper.tldr = cached_tldr
            logger.debug(f"Using cached TLDR for paper {paper.get_short_id()}")
        else:
            # Generate new TLDR if not cached
            paper.tldr = llm.get_tldr(paper.title, paper.summary)
            # Cache the new TLDR
            paper_cache.save_tldr(paper.get_short_id(), paper.tldr)
            logger.debug(f"Cached new TLDR for paper {paper.get_short_id()}")
    
    # Render email content
    html = framework
    # Add MathJax script for HTML output
    if not for_email:
        html = html.replace('</head>', f'{mathjax_script}</head>')
    
    html = html.replace("__TITLE__", "ArXiv Daily Digest" if lang == "English" else "ArXiv 每日文摘")
    
    # Add papers
    papers_html = []
    for paper in papers:
        # Process title and abstract to handle math expressions
        title = paper.title
        abstract = paper.tldr
        
        logger.debug(f"Processing paper {paper.get_short_id()}")
        logger.debug(f"Original title: {title}")
        logger.debug(f"Original abstract: {abstract}")
        
        # Normalize math delimiters in both title and abstract
        title = normalize_math_delimiters(title)
        abstract = normalize_math_delimiters(abstract)
        
        logger.debug(f"Normalized title: {title}")
        logger.debug(f"Normalized abstract: {abstract}")
        
        if for_email:
            # For email, convert all math to SVG images
            title = process_math_for_email(title)
            abstract = process_math_for_email(abstract)
        # For HTML output, keep the math expressions as is (with $ and $$)
        
        logger.debug(f"Processed title: {title}")
        logger.debug(f"Processed abstract: {abstract}")
        
        # Build similarity info HTML
        if lang.lower() == "chinese":
            rate_text = "相关度："
            pdf_text = "PDF"
            similarity_text = "相似度："
        else:
            rate_text = "Relevance: "
            pdf_text = "PDF"
            similarity_text = "Similarity: "
        
        similarity_html = f'<div class="similarity">{similarity_text}{paper.score:.2f}</div>' if hasattr(paper, 'score') else ''
        
        # Generate HTML block for this paper
        paper_html = f"""
        <li>
        <div class="paper">
            <div class="header">
                <div class="title">
                        <a href="https://arxiv.org/abs/{paper.get_short_id()}" target="_blank">{title}</a>
                    </div>
                    <div class="rate">
                        <div>{rate_text}<span class="stars">{get_stars(paper.score)}</span></div>
                        {similarity_html}
                    </div>
            </div>
            <div class="meta">
                    <div class="authors">{', '.join(paper.authors)}</div>
                <div class="links">
                        <a href="{paper.pdf_url}" target="_blank">{pdf_text}</a>
                    </div>
                </div>
                <div class="abstract">{abstract}</div>
            </div>
        </li>
        """
        papers_html.append(paper_html)
    
    html = html.replace("__PAPERS__", "\n".join(papers_html))
    if not for_email:
        save_html_output(html)
    return html

def send_email(sender: str, receiver: str, password: str, smtp_server: str, smtp_port: int, html: str, lang: str = "English"):
    """Send email with pre-rendered math content
    
    Args:
        sender: Email sender address
        receiver: Email receiver address
        password: Email password
        smtp_server: SMTP server address
        smtp_port: SMTP server port
        html: HTML content to send
        lang: Language for the email (default: "English")
    """
    def _format_addr(s, name):
        return formataddr((Header(name, 'utf-8').encode(), s))

    msg = MIMEText(html, 'html', 'utf-8')
    
    # Set sender and receiver names based on language
    if lang.lower() == "chinese":
        sender_name = "arXiv 论文推荐"
        receiver_name = "您"
        subject = f'每日 arXiv 论文推荐 {datetime.datetime.now().strftime("%Y/%m/%d")}'
    else:
        sender_name = "arXiv Paper Recommendations"
        receiver_name = "You"
        subject = f'Daily arXiv Paper Recommendations {datetime.datetime.now().strftime("%Y/%m/%d")}'
    
    msg['From'] = _format_addr(sender, sender_name)
    msg['To'] = _format_addr(receiver, receiver_name)
    msg['Subject'] = Header(subject, 'utf-8').encode()
    msg['Content-Type'] = 'text/html; charset=utf-8'
    msg['MIME-Version'] = '1.0'

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
