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
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            },
            svg: {
                fontCache: 'global'
            }
        };
    </script>
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
    """生成空结果的 HTML"""
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

def get_block_html(paper: ArxivPaper, lang: str = "English") -> str:
    """Generate HTML block for a single paper
    
    Args:
        paper: ArxivPaper object
        lang: Language for the content (default: "English")
        
    Returns:
        str: HTML block
    """
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
    
    return f"""
    <li>
        <div class="paper">
            <div class="header">
                <div class="title">
                    <a href="https://arxiv.org/abs/{paper.get_short_id()}" target="_blank">{paper.title}</a>
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
            <div class="abstract">{paper.tldr}</div>
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
    """Save the HTML content to a file with both interactive and pre-rendered versions"""
    # Save interactive version (with MathJax)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Interactive email content saved to {output_path}")
        
        # Save pre-rendered version
        pre_rendered_path = output_path.replace('.html', '-pre-rendered.html')
        pre_rendered_html = pre_render_math(html)
        with open(pre_rendered_path, 'w', encoding='utf-8') as f:
            f.write(pre_rendered_html)
        logger.info(f"Pre-rendered email content saved to {pre_rendered_path}")
        
        logger.info(f"You can open these files in a web browser to preview the email content")
    except Exception as e:
        logger.error(f"Failed to save HTML output: {e}")

def render_email(papers: List[ArxivPaper], lang: str = "English") -> str:
    """Render email content"""
    if not papers:
        return get_empty_html(lang)
        
    # Initialize LLM for TLDR generation
    llm = get_llm()
    if isinstance(llm.llm, OpenAI):
        logger.info(f"Using OpenAI API for TLDR generation (model: {llm.model})")
    else:
        logger.info(f"Using local LLM for TLDR generation: {llm.model}")
    
    # Generate TLDR for each paper with progress bar
    logger.info("Generating TLDR summaries...")
    for paper in tqdm(papers, desc="Generating TLDRs"):
        paper.tldr = llm.get_tldr(paper.title, paper.summary)
    
    # Render email content
    html = framework
    html = html.replace("__TITLE__", "ArXiv Daily Digest" if lang == "English" else "ArXiv 每日文摘")
    
    # Add papers
    papers_html = ""
    for paper in papers:
        papers_html += get_block_html(paper, lang)
    
    html = html.replace("__PAPERS__", papers_html)
    save_html_output(html)
    return html

def pre_render_math(html_content: str) -> str:
    """Pre-render math expressions to HTML using MathJax Node API"""
    # Extract math expressions
    inline_math = re.findall(r'\$([^$]+)\$', html_content)
    display_math = re.findall(r'\$\$([^$]+)\$\$', html_content)
    
    # Replace math expressions with placeholders
    math_expressions = inline_math + display_math
    placeholders = []
    for i, expr in enumerate(math_expressions):
        placeholder = f'__MATH_{i}__'
        html_content = html_content.replace(f'${expr}$', placeholder)
        html_content = html_content.replace(f'$${expr}$$', placeholder)
        placeholders.append(expr)
    
    # Use MathJax Node API to render math
    rendered_math = []
    for expr in placeholders:
        try:
            # Use MathJax Node API to render the expression
            response = requests.post(
                'https://mathjax-node.herokuapp.com/tex2svg',
                json={'math': expr},
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                rendered_math.append(response.text)
            else:
                rendered_math.append(f'<span class="math-error">Error rendering: {expr}</span>')
        except Exception as e:
            logger.error(f"Failed to render math expression: {expr}, error: {e}")
            rendered_math.append(f'<span class="math-error">Error rendering: {expr}</span>')
    
    # Replace placeholders with rendered math
    for i, rendered in enumerate(rendered_math):
        html_content = html_content.replace(f'__MATH_{i}__', rendered)
    
    return html_content

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
    # Pre-render math expressions
    html = pre_render_math(html)
    
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
