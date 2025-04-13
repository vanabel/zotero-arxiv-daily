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
    <h1>今日 arXiv 论文推荐</h1>
    <ol>
        __CONTENT__
    </ol>
</body>
</html>
"""

def get_empty_html() -> str:
    """生成空结果的 HTML"""
    return """
    <div class="empty-result">
        <h2>没有找到相关论文</h2>
        <p>今天没有与您的兴趣相关的新论文。请明天再来查看！</p>
    </div>
    """

def get_block_html(title: str, authors: str, rate: str, arxiv_id: str, abstract: str, pdf_url: str, code_url: str = None, similarity: float = None) -> str:
    """生成单个论文的 HTML 块
    
    Args:
        title: 论文标题
        authors: 作者列表
        rate: 相关性评分（星星）
        arxiv_id: arXiv ID
        abstract: 摘要
        pdf_url: PDF 链接
        code_url: 代码链接（可选）
        similarity: 相似度分数（可选）
        
    Returns:
        str: HTML 块
    """
    # 构建代码链接 HTML
    code_html = f'<a href="{code_url}" target="_blank">Code</a>' if code_url else ''
    
    # 构建相似度信息 HTML
    similarity_html = f'<div class="similarity">相似度: {similarity:.2f}</div>' if similarity is not None else ''
    
    return f"""
    <li>
        <div class="paper">
            <div class="header">
                <div class="title">
                    <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{title}</a>
                </div>
                <div class="rate">
                    <div>相关度：<span class="stars">{rate}</span></div>
                    {similarity_html}
                </div>
            </div>
            <div class="meta">
                <div class="authors">{authors}</div>
                <div class="links">
                    <a href="{pdf_url}" target="_blank">PDF</a>
                    {code_html}
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
    """Save the HTML content to a file
    
    Args:
        html: The HTML content to save
        output_path: The path to save the HTML file (default: output-email.html)
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Email content saved to {output_path}")
        logger.info(f"You can open this file in a web browser to preview the email content")
    except Exception as e:
        logger.error(f"Failed to save HTML output: {e}")

def render_email(papers: List[ArxivPaper]) -> str:
    """渲染邮件内容"""
    logger.info("Rendering email content...")
    
    if not papers:
        html = get_empty_html()
        save_html_output(html)
        return html
    
    # 设置 LLM 模型
    llm = get_llm()
    logger.info(f"Using {llm.model} for TLDR generation")
    
    # 渲染每篇论文
    blocks = []
    for paper in tqdm(papers, desc="Processing papers", unit="paper"):
        logger.debug(f"Processing paper: {paper.title}")
        
        # 获取 TLDR
        tldr = None
        paper_id = paper.get_short_id()
        
        # Check if we can use cached TLDR
        if paper_cache.can_use_cached_tldr(paper_id, paper.score):
            tldr = paper_cache.get_tldr(paper_id)
            if tldr is not None:
                logger.debug(f"Using cached TLDR for {paper.title} (score: {paper.score:.2f})")
        
        # If no cached TLDR or score changed significantly, generate new one
        if tldr is None:
            try:
                tldr = llm.get_tldr(paper.title, paper.summary)
                paper_cache.save_tldr(paper_id, tldr)
                logger.debug(f"Generated new TLDR for {paper.title} (score: {paper.score:.2f})")
            except Exception as e:
                logger.error(f"Failed to generate TLDR for {paper.title}: {e}")
                tldr = "Failed to generate TLDR"
        
        # 获取代码链接
        code_url = paper.code_url
        
        # 获取相关性评分的星星显示
        stars = get_stars(paper.score)
        
        # 构建论文 HTML 块
        block = get_block_html(
            title=paper.title,
            authors=', '.join(paper.authors),
            rate=stars,
            arxiv_id=paper_id,
            abstract=tldr,
            pdf_url=paper.pdf_url,
            code_url=code_url,
            similarity=paper.score  # Add similarity score
        )
        blocks.append(block)
    
    # 使用框架模板
    content = '<br>'.join(blocks)
    html = framework.replace('__CONTENT__', content)
    
    # Save the HTML output to a file
    save_html_output(html)
    
    return html

def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'每日 arXiv 论文推荐 {today}', 'utf-8').encode()
    # 添加额外的头信息以确保正确的字符编码
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
