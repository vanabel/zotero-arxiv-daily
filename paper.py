from typing import Optional, List
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
from llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ArxivPaper:
    def __init__(self, paper: arxiv.Result):
        """Initialize an ArxivPaper object"""
        self._paper = paper
        self.score = 0.0  # Similarity score
        if not self._validate_paper():
            logger.warning(f"Paper data validation failed for {self.get_short_id()}")
    
    def _validate_paper(self) -> bool:
        """验证论文数据的完整性"""
        if isinstance(self._paper, dict):
            required_fields = ['title', 'summary', 'authors', 'pdf_url', 'primary_category']
            # Check for either 'id' or 'entry_id'
            if not self._paper.get('id') and not self._paper.get('entry_id'):
                logger.warning("Missing required field: either 'id' or 'entry_id'")
                return False
            missing_fields = [field for field in required_fields if not self._paper.get(field)]
            if missing_fields:
                logger.warning(f"Missing required fields for paper: {missing_fields}")
                return False
        else:
            try:
                if not all([
                    self._paper.title,
                    self._paper.summary,
                    self._paper.authors,
                    self._paper.entry_id,
                    self._paper.pdf_url,
                    self._paper.primary_category
                ]):
                    return False
            except AttributeError as e:
                logger.warning(f"Failed to access paper attributes: {e}")
                return False
        return True

    @property
    def title(self) -> str:
        """Get paper title"""
        try:
            if isinstance(self._paper, dict):
                return self._paper.get('title', '[No Title Available]')
            return self._paper.title or '[No Title Available]'
        except Exception as e:
            logger.warning(f"Error getting title: {e}")
            return '[No Title Available]'
    
    @property
    def summary(self) -> str:
        """Get paper summary/abstract"""
        try:
            if isinstance(self._paper, dict):
                return self._paper.get('summary', '[No Abstract Available]')
            return self._paper.summary or '[No Abstract Available]'
        except Exception as e:
            logger.warning(f"Error getting summary: {e}")
            return '[No Abstract Available]'
    
    @property
    def authors(self) -> List[str]:
        """Get paper authors"""
        try:
            if isinstance(self._paper, dict):
                authors = self._paper.get('authors', ['[Unknown Author]'])
                # 如果作者是字符串列表，直接返回
                if all(isinstance(author, str) for author in authors):
                    return authors
                # 如果作者是对象列表，提取 name 属性
                return [getattr(author, 'name', str(author)) for author in authors]
            # 如果是 arxiv.Result 对象，提取作者名字
            return [author.name for author in self._paper.authors] or ['[Unknown Author]']
        except Exception as e:
            logger.warning(f"Error getting authors: {e}")
            return ['[Unknown Author]']
    
    @property
    def published(self) -> datetime:
        """Get paper publication date"""
        if isinstance(self._paper, dict):
            pub_date = self._paper.get('published')
            return datetime.fromisoformat(pub_date) if pub_date else datetime.now()
        return self._paper.published
    
    @property
    def updated(self) -> datetime:
        """Get paper last update date"""
        if isinstance(self._paper, dict):
            update_date = self._paper.get('updated')
            return datetime.fromisoformat(update_date) if update_date else datetime.now()
        return self._paper.updated
    
    @property
    def entry_id(self) -> str:
        """Get paper ID"""
        try:
            if isinstance(self._paper, dict):
                # 优先使用 id 字段，这是 arXiv API 返回的标准字段
                return self._paper.get('id', '') or self._paper.get('entry_id', '')
            return self._paper.entry_id or ''
        except Exception as e:
            logger.warning(f"Error getting entry_id: {e}")
            return ''
    
    @property
    def pdf_url(self) -> str:
        """Get paper PDF URL"""
        try:
            if isinstance(self._paper, dict):
                # 从 entry_id 构建 PDF URL
                entry_id = self.entry_id
                if entry_id:
                    if 'arxiv.org/abs/' in entry_id:
                        return entry_id.replace('abs', 'pdf')
                    elif 'arxiv.org/pdf/' not in entry_id:
                        short_id = self.get_short_id()
                        return f'http://arxiv.org/pdf/{short_id}'
                return self._paper.get('pdf_url', '')
            return self._paper.pdf_url or ''
        except Exception as e:
            logger.warning(f"Error getting pdf_url: {e}")
            return ''
    
    @property
    def primary_category(self) -> str:
        """Get paper primary category"""
        try:
            if isinstance(self._paper, dict):
                # 优先使用 primary_category 字段
                primary = self._paper.get('primary_category', '')
                if isinstance(primary, dict):
                    return primary.get('term', 'unknown')
                elif isinstance(primary, str):
                    return primary
                # 如果没有 primary_category，尝试从 categories 中获取第一个
                categories = self._paper.get('categories', [])
                if categories and isinstance(categories[0], dict):
                    return categories[0].get('term', 'unknown')
                elif categories and isinstance(categories[0], str):
                    return categories[0]
                return 'unknown'
            return self._paper.primary_category or 'unknown'
        except Exception as e:
            logger.warning(f"Error getting primary_category: {e}")
            return 'unknown'
    
    @property
    def categories(self) -> List[str]:
        """Get paper categories"""
        try:
            if isinstance(self._paper, dict):
                cats = self._paper.get('categories', [])
                if cats and isinstance(cats[0], dict):
                    return [cat.get('term', '') for cat in cats if cat.get('term')]
                elif cats and isinstance(cats[0], str):
                    return cats
                return [self.primary_category]
            return self._paper.categories or [self.primary_category]
        except Exception as e:
            logger.warning(f"Error getting categories: {e}")
            return [self.primary_category]
    
    def get_short_id(self) -> str:
        """Get short paper ID (e.g. '2103.12345')"""
        entry_id = self.entry_id
        if not entry_id:
            logger.warning("Empty entry_id")
            return ""
        
        # 如果 entry_id 是字典类型，尝试获取 '$t' 键的值
        if isinstance(entry_id, dict):
            entry_id = entry_id.get('$t', '') or entry_id.get('term', '')
            if not entry_id:
                logger.warning("Empty entry_id after getting '$t' key")
                return ""
        
        # 从 URL 中提取 ID
        patterns = [
            r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',  # 标准格式：arxiv.org/abs/2103.12345v1 或 arxiv.org/pdf/2103.12345v1
            r'^(\d{4}\.\d{4,5}(?:v\d+)?)$'                        # 直接的 ID 格式：2103.12345v1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, entry_id)
            if match:
                return match.group(1)
        
        # 如果没有匹配到任何模式，尝试从 URL 中提取最后一部分
        if '/' in entry_id:
            last_part = entry_id.split('/')[-1]
            # 验证提取的部分是否符合 arXiv ID 格式
            if re.match(r'^\d{4}\.\d{4,5}(?:v\d+)?$', last_part):
                return last_part
        
        logger.warning(f"Could not extract arxiv ID from entry_id: {entry_id}")
        return entry_id
    
    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self.get_short_id())
    
    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=10, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count',0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count',0) == 0:
            return None
        return repo_list['results'][0]['url']
    
    @cached_property
    def tex(self) -> dict[str,str]:
        """获取论文的 TeX 源代码"""
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            
            try:
                # 如果是字典类型，需要先构建下载 URL
                if isinstance(self._paper, dict):
                    arxiv_id = self.get_short_id()
                    if not arxiv_id:
                        logger.warning("Cannot get arxiv ID for downloading source")
                        return None
                    
                    # 构建源代码下载 URL
                    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
                    
                    # 下载源代码
                    response = requests.get(source_url)
                    if response.status_code != 200:
                        logger.warning(f"Failed to download source for {arxiv_id}: {response.status_code}")
                        return None
                        
                    # 保存到临时文件
                    file_path = os.path.join(tmpdirname, f"{arxiv_id}")
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    file = file_path
                else:
                    # 使用 arxiv.Result 对象的方法下载
                    file = self._paper.download_source(dirpath=tmpdirname)
                
                try:
                    tar = stack.enter_context(tarfile.open(file))
                except tarfile.ReadError:
                    logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                    return None
     
                tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
                if len(tex_files) == 0:
                    logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                    return None
                
                bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
                match len(bbl_file):
                    case 0:
                        if len(tex_files) > 1:
                            logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file.")
                            main_tex = None
                        else:
                            main_tex = tex_files[0]
                    case 1:
                        main_name = bbl_file[0].replace('.bbl','')
                        main_tex = f"{main_name}.tex"
                        if main_tex not in tex_files:
                            logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file.")
                            main_tex = None
                    case _:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files.")
                        main_tex = None
                
                if main_tex is None:
                    logger.debug(f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")
                
                # 读取所有 tex 文件
                file_contents = {}
                for t in tex_files:
                    f = tar.extractfile(t)
                    if f is None:
                        continue
                    content = f.read().decode('utf-8',errors='ignore')
                    # 移除注释
                    content = re.sub(r'%.*\n', '\n', content)
                    content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                    content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                    # 移除多余的换行
                    content = re.sub(r'\n+', '\n', content)
                    content = re.sub(r'\\\\', '', content)
                    # 移除连续的空格
                    content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                    if main_tex is None and re.search(r'\\begin\{document\}', content):
                        main_tex = t
                        logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                    file_contents[t] = content
                
                # 处理主文件
                if main_tex is not None:
                    main_source = file_contents[main_tex]
                    # 查找并替换所有包含的子文件
                    include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
                    for f in include_files:
                        if not f.endswith('.tex'):
                            file_name = f + '.tex'
                        else:
                            file_name = f
                        main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                        main_source = main_source.replace(f'\\include{{{f}}}', file_contents.get(file_name, ''))
                    file_contents["all"] = main_source
                else:
                    logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                    file_contents["all"] = None
                
                return file_contents
            except Exception as e:
                logger.warning(f"Error processing tex files for {self.arxiv_id}: {e}")
                return None
    
    @cached_property
    def tldr(self) -> str:
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            #remove cite
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            #remove figure
            content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            #remove table
            content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            #find introduction and conclusion
            # end word can be \section or \end{document} or \bibliography or \appendix
            match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                introduction = match.group(0)
            match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                conclusion = match.group(0)
        llm = get_llm()
        prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:
        
        \\title{__TITLE__}
        \\begin{abstract}__ABSTRACT__\\end{abstract}
        __INTRODUCTION__
        __CONCLUSION__
        """
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        tldr = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return tldr

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            #search for affiliations
            possible_regions = [r'\\author.*?\\maketitle',r'\\begin{document}.*?\\begin{abstract}']
            matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
            match = next((m for m in matches if m), None)
            if match:
                information_region = match.group(0)
            else:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
                return None
            prompt = f"Given the author information of a paper in latex format, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]'. Following is the author information:\n{information_region}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
            prompt = enc.decode(prompt_tokens)
            llm = get_llm()
            affiliations = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. You should return a python list of affiliations sorted by the author order, like ['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            try:
                affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
                affiliations = eval(affiliations)
                affiliations = list(set(affiliations))
                affiliations = [str(a) for a in affiliations]
            except Exception as e:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
                return None
            return affiliations

    @property
    def comment(self) -> str:
        return self._paper.comment if hasattr(self._paper, 'comment') else ''