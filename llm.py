from llama_cpp import Llama
from openai import OpenAI
from loguru import logger

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model or "gpt-4"  # Default to gpt-4 if no model specified
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
            self.model = "Qwen2.5-3B-Instruct"  # Local model name
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            response = self.llm.chat.completions.create(messages=messages,temperature=0,model=self.model)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

    def get_tldr(self, title: str, abstract: str) -> str:
        """生成论文的 TLDR 摘要"""
        prompt = f"""请用一句话总结这篇论文的主要内容（如果语言是英文则用英文回答）：

标题：{title}

摘要：{abstract}

一句话总结："""

        messages = [
            {"role": "system", "content": "你是一个专业的科研助手，擅长用简洁的一句话总结学术论文的核心内容。"},
            {"role": "user", "content": prompt}
        ]

        try:
            return self.generate(messages)
        except Exception as e:
            logger.error(f"生成 TLDR 失败: {e}")
            return "无法生成摘要"

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM