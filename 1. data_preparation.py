



# 下载数据
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", cache_dir="data/gsm8k")

# # 查看数据集结构
# print(dataset)
#
# # 查看训练集第一条
# print(dataset["train"][0])


# def create_sft_dataset(dataset,max_sample=1000):


import os
import json
from datasets import load_dataset, Dataset


class GSM8K_SFT_Dataset:
    """GSM8K数学推理数据集 -> SFT格式

    自动处理：下载原始数据 -> 转换为SFT格式 -> 本地缓存为jsonl -> 后续直接加载，
    不用每次都重新走HuggingFace下载+转换的流程。
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = None,
        raw_cache_dir: str = "data/gsm8k",
        sft_cache_path: str = None,
    ):
        """
        Args:
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数
            raw_cache_dir: 原始GSM8K数据集的缓存目录
            sft_cache_path: SFT格式jsonl文件的保存路径，默认根据split自动生成
        """
        self.split = split
        # self.max_samples = max_samples
        # 统一把 None 和 'all' 归一化成 None,表示不限制数量
        if max_samples is None or (isinstance(max_samples, str) and max_samples.lower() == "all"):
            self.max_samples = None
        else:
            self.max_samples = int(max_samples)


        self.raw_cache_dir = raw_cache_dir
        if sft_cache_path:
            self.sft_cache_path=sft_cache_path
        elif max_samples:
            self.sft_cache_path=f"data/gsm8k/gsm8k_sft_{split}_{max_samples}.jsonl"
        else:
            self.sft_cache_path=f"data/gsm8k/gsm8k_sft_{split}_all.jsonl"


        self.data = self.load_or_create()

    # ---------- 单条样本转换逻辑 ----------
    def format_for_sft(self, example):
        question = example["question"]
        answer = example["answer"]

        if "####" in answer:
            reasoning, final_answer = answer.split("####")
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = answer
            final_answer = ""

        prompt = f"Question: {question}\n\nLet's solve this step by step:\n"
        completion = f"{reasoning}\n\nFinal Answer: {final_answer}"

        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion
        }

    # ---------- 下载原始数据并批量转换 ----------
    def create_sft_data(self):
        print(f"📥 加载原始 GSM8K 数据集 (split={self.split})...")
        raw = load_dataset("openai/gsm8k", "main", split=self.split, cache_dir=self.raw_cache_dir)

        if self.max_samples:
            raw = raw.select(range(min(self.max_samples, len(raw))))

        sft_data = [self.format_for_sft(example) for example in raw]    # 对于GSM8K数据集应该不会太慢，一共就9000多条数据
        print(f"⚙️ 已转换 {len(sft_data)} 条样本为SFT格式")
        return sft_data

    # ---------- jsonl 读写 ----------
    def save_to_jsonl(self, data):
        dir_name = os.path.dirname(self.sft_cache_path)
        if dir_name:  # 只有当路径里确实包含目录部分时才创建
            os.makedirs(dir_name, exist_ok=True)
        # os.makedirs(os.path.dirname(self.sft_cache_path), exist_ok=True)
        with open(self.sft_cache_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"💾 已保存至 {self.sft_cache_path}")

    def load_from_jsonl(self):
        data = []
        with open(self.sft_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        print(f"📂 从本地缓存加载了 {len(data)} 条样本: {self.sft_cache_path}")
        return data

    # ---------- 核心：存在则load，不存在则create+save ----------
    def load_or_create(self):
        if os.path.exists(self.sft_cache_path):
            return self.load_from_jsonl()
        else:
            data = self.create_sft_data()
            self.save_to_jsonl(data)
            return data

    # ---------- 对外接口 ----------
    def to_hf_dataset(self) -> Dataset:
        """转换为HuggingFace Dataset对象，方便配合SFTTrainer等工具使用"""
        return Dataset.from_list(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------- 使用示例 ----------
# train_dataset = GSM8K_SFT_Dataset(split="train", max_samples=1000,sft_cache_path='sft_train.jsonl')
train_dataset = GSM8K_SFT_Dataset(split="test", max_samples=1000,sft_cache_path='sft_train.jsonl')
print(len(train_dataset))
print(train_dataset[2])

# # 如果后续要接trl的SFTTrainer，转一下格式
# train_hf = train_dataset.to_hf_dataset()



import os
import json
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset


class GSM8K_RL_Dataset:
    """GSM8K数学推理数据集 -> RL格式（已应用chat template）

    自动处理：下载原始数据 -> 转换为RL格式(standard format, prompt为字符串)
    -> 本地缓存为jsonl -> 后续直接加载。
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = None,
        raw_cache_dir: str = "data/gsm8k",
        rl_cache_path: str = None,
        tokenizer = None,
    ):
        """
        Args:
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数
            raw_cache_dir: 原始GSM8K数据集的缓存目录
            rl_cache_path: RL格式jsonl文件的保存路径，默认根据split自动生成
            tokenizer: 用于应用chat template的tokenizer，传None则直接用原始文本作为prompt
        """
        self.split = split
        self.max_samples = max_samples
        self.raw_cache_dir = raw_cache_dir

        # self.rl_cache_path = rl_cache_path or f"data/gsm8k_rl_{split}.jsonl"
        if rl_cache_path:
            self.rl_cache_path=rl_cache_path
        elif max_samples:
            self.rl_cache_path=f"data/gsm8k/gsm8k_rl_{split}_{max_samples}.jsonl"
        else:
            self.rl_cache_path=f"data/gsm8k/gsm8k_rl_{split}_all.jsonl"

        self.tokenizer = tokenizer

        self.data = self.load_or_create()

    # ---------- 单条样本转换逻辑 ----------
    def format_for_rl(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为RL训练格式(Standard Format with Chat Template Applied)

        Returns:
            - prompt: 应用chat template后的文本字符串
            - ground_truth: 正确答案
            - question: 原始问题
            - full_answer: 完整答案
        """
        question = example["question"]
        answer = example["answer"]

        if "####" in answer:
            _, final_answer = answer.split("####")
            final_answer = final_answer.strip()
        else:
            final_answer = answer.strip()

        prompt_content = f"Question: {question}\n\nLet's solve this step by step:"

        # 如果提供了tokenizer,应用chat template
        if self.tokenizer:
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = prompt_content
        # prompt要进行这个处理，主要是后续需要LLM生成多个答案，用以训练。

        return {
            "prompt": prompt_text,
            "ground_truth": final_answer,
            "question": question,
            "full_answer": answer
        }

    # ---------- 下载原始数据并批量转换 ----------
    def create_rl_data(self):
        print(f"📥 加载原始 GSM8K 数据集 (split={self.split})...")
        raw = load_dataset("openai/gsm8k", "main", split=self.split, cache_dir=self.raw_cache_dir)

        if self.max_samples:
            raw = raw.select(range(min(self.max_samples, len(raw))))

        rl_data = [self.format_for_rl(example) for example in raw]
        print(f"⚙️ 已转换 {len(rl_data)} 条样本为RL格式")
        return rl_data

    # ---------- jsonl 读写 ----------
    def save_to_jsonl(self, data):
        os.makedirs(os.path.dirname(self.rl_cache_path), exist_ok=True)
        with open(self.rl_cache_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"💾 已保存至 {self.rl_cache_path}")

    def load_from_jsonl(self):
        data = []
        with open(self.rl_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        print(f"📂 从本地缓存加载了 {len(data)} 条样本: {self.rl_cache_path}")
        return data

    # ---------- 核心：存在则load，不存在则create+save ----------
    def load_or_create(self):
        if os.path.exists(self.rl_cache_path):
            return self.load_from_jsonl()
        else:
            data = self.create_rl_data()
            self.save_to_jsonl(data)
            return data

    # ---------- 对外接口 ----------
    def to_hf_dataset(self) -> Dataset:
        return Dataset.from_list(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]












