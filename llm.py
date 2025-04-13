from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
import os
from huggingface_hub import list_repo_files

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model or "gpt-4"  # Default to gpt-4 if no model specified
        else:
            model_name = os.getenv('TLDR_MODEL_NAME', 'Qwen/Qwen2.5-3B-Instruct-GGUF')
            
            # Get available files from the repository
            available_files = list_repo_files(model_name)
            
            # Find GGUF files
            gguf_files = [f for f in available_files if f.endswith('.gguf')]
            if not gguf_files:
                raise ValueError(
                    f"No GGUF files found in {model_name}. "
                    f"Please use a repository that contains GGUF files. "
                    f"Available files: {available_files}"
                )
            
            # Prefer q4_k_m version if available
            q4_k_m_files = [f for f in gguf_files if 'q4_k_m' in f]
            filename = q4_k_m_files[0] if q4_k_m_files else gguf_files[0]
            
            logger.info(f"Using GGUF file: {filename}")
            
            self.llm = Llama.from_pretrained(
                repo_id=model_name,
                filename=filename,
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
            self.model = model_name  # Use environment variable for local model name
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            response = self.llm.chat.completions.create(messages=messages,temperature=0,model=self.model)
            return response.choices[0].message.content
        else:
            # Add language-specific system message
            if self.lang.lower() == "chinese":
                system_message = {
                    "role": "system",
                    "content": "你是一个专业的科研助手，擅长用简洁的一句话总结学术论文的核心内容。请用中文回答。"
                }
            else:  # Default to English
                system_message = {
                    "role": "system",
                    "content": "You are a professional research assistant, skilled at summarizing academic papers in one concise sentence. Please respond in English."
                }
            
            # Insert system message at the beginning
            messages.insert(0, system_message)
            
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

    def get_tldr(self, title: str, abstract: str) -> str:
        """Generate TLDR summary of a paper"""
        if self.lang.lower() == "chinese":
            prompt = f"""请用一句话总结这篇论文的主要内容：

标题：{title}

摘要：{abstract}

一句话总结："""
        else:  # Default to English
            prompt = f"""Please summarize the main content of this paper in one sentence:

Title: {title}

Abstract: {abstract}

One-sentence summary:"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            return self.generate(messages)
        except Exception as e:
            logger.error(f"Failed to generate TLDR: {e}")
            return "无法生成摘要" if self.lang.lower() == "chinese" else "Failed to generate summary"

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM