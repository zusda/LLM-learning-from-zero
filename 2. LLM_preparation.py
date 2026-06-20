



import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMModelLoader:
    """检测本地是否已有模型,没有则从镜像源下载并保存,有则直接加载。"""

    def __init__(
        self,
        model_id="Qwen/Qwen3-0.6B",
        local_root="./LLM_models",
        hf_endpoint="https://hf-mirror.com",  # huggingface.co
        torch_dtype=torch.float16,
        device_map="auto",
    ):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map

        # 用模型名作为本地子目录,把斜杠替换成下划线避免多级目录歧义
        self.local_dir = Path(local_root) / model_id.replace("/", "_")  # LLM_models\Qwen_Qwen3-0.6B


        # 必须在第一次触发网络请求之前设置好
        os.environ["HF_ENDPOINT"] = hf_endpoint

    def model_exists(self):
        if not self.local_dir.exists():
            return False

        has_config = (self.local_dir / "config.json").exists()
        has_weights = any(self.local_dir.glob("*.safetensors")) or any(
            self.local_dir.glob("*.bin")
        )
        has_tokenizer = (self.local_dir / "tokenizer_config.json").exists()

        return has_config and has_weights and has_tokenizer

    def download_model(self):
        print(f"本地未找到模型,从 {os.environ['HF_ENDPOINT']} 下载 {self.model_id} ...")

        self.local_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)

        # 固化保存到本地目录,后续直接从这里加载,不再走网络
        model.save_pretrained(self.local_dir)
        tokenizer.save_pretrained(self.local_dir)

        print(f"下载完成,已保存到 {self.local_dir}")
        return model, tokenizer

    def load_model(self):
        print(f"检测到本地模型,直接从 {self.local_dir} 加载...")

        model = AutoModelForCausalLM.from_pretrained(
            self.local_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.local_dir, use_fast=False)

        return model, tokenizer

    def get_model_and_tokenizer(self):
        if self.model_exists():
            return self.load_model()
        return self.download_model()


if __name__ == "__main__":
    # .\LLM_models\Qwen_Qwen3-0.6B
    loader = LLMModelLoader(model_id="Qwen/Qwen3-0.6B", local_root="./LLM_models")
    model_trained, tokenizer = loader.get_model_and_tokenizer()





