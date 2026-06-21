




import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup


class SFTDataset(Dataset):
    """把 {prompt, completion} 数据转成带 labels mask 的 input_ids。

    核心逻辑:
    prompt 部分的 labels 设为 -100(不参与loss),
    completion 部分的 labels 等于真实 token id(参与loss)。
    """

    def __init__(self, data, tokenizer, max_length=512, use_chat_template=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template and tokenizer.chat_template is not None

    def __len__(self):
        return len(self.data)


    def build_ids(self, prompt, completion):
        def to_id_list(x):
            """把apply_chat_template可能返回的各种类型统一转成纯int列表"""
            if hasattr(x, "ids"):  # tokenizers.Encoding对象
                return list(x.ids)
            if hasattr(x, "input_ids"):  # BatchEncoding对象
                return list(x.input_ids)
            if isinstance(x, dict):  # 普通dict
                return list(x["input_ids"])
            return list(x)  # 已经是list[int],直接返回

        if self.use_chat_template:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                # 添加上
                # <|im_start|>assistant
            )
            full_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                    # 添加上
                    # <|im_start|>assistant
                    # {completion}<|im_end|>
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
            # print(type(full_ids), full_ids)
            prompt_ids = to_id_list(prompt_ids)
            full_ids = to_id_list(full_ids)
            # input()
        else:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
            full_ids = full_ids + [self.tokenizer.eos_token_id]

        return prompt_ids, full_ids

    # __getitem__是PyTorch Dataset协议规定要实现的方法
    # DataLoader组batch时会按索引一条条调用它,要求每次返回该索引对应的单条训练样本。
    def __getitem__(self, idx):
        example = self.data[idx]  # 取出第idx条原始数据，这时候还是纯文本的{"prompt": ..., "completion": ...}字典,没经过任何tokenize
        prompt_ids, full_ids = self.build_ids(example["prompt"], example["completion"])

        # 添加sft的mask。将prompt部分遮蔽。
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 安全截断, 防止某条样本特别长直接把显存撑爆——这里截断方式很简单粗暴, 就是从尾部直接砍掉超出部分
        full_ids = full_ids[: self.max_length]
        labels = labels[: self.max_length]


        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id):
    # 把“不同长度的样本”整理成一个标准的训练 batch（对齐 + padding + mask）

    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    # 初始化空矩阵(以max_len=3，batch=2为例)
    # input_ids:
    # [
    #  [PAD, PAD, PAD],
    #  [PAD, PAD, PAD]
    # ]
    #
    # labels:
    # [
    #  [-100, -100, -100],
    #  [-100, -100, -100]
    # ]
    #
    # attention_mask:
    # [
    #  [0, 0, 0],
    #  [0, 0, 0]
    # ]

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1
    # 第二条样本举例
    # item[1]:
    # input_ids = [20, 21]
    # labels    = [20, 21]
    # 变成
    # input_ids
    # [
    #  [20, 21, PAD]
    # ]
    # labels
    # [
    #  [20, 21, -100]
    # ]
    # attention_mask
    # [
    #  [1, 1, 0]
    # ]


    # attention_mask是告诉模型哪些位置是真实token、哪些是padding。在attention计算里,0的位置在计算attention score时会被mask掉。
    # labels是告诉模型哪些位置需要计算loss。

    # 用一个具体的例子,假设两条样本padding后长度都是8:
    # 样本1: [user问话token×3] [assistant回答token×3] [pad×2]
    # 样本2: [user问话token×2] [assistant回答token×4] [pad×2]
    # 两个mask长这样:
    # 样本1
    # attention_mask = [1, 1, 1, 1, 1, 1, 0, 0]  # 后两个是pad,不参与attention
    # labels         = [-100, -100, -100, id, id, id, -100, -100]  # 前3个是prompt不算loss,后两个是pad也不算loss
    #
    # 样本2
    # attention_mask = [1, 1, 1, 1, 1, 1, 0, 0]
    # labels         = [-100, -100, id, id, id, id, -100, -100]

    # Decoder的causal attention本身就自带一个下三角mask,强制每个位置只能看到自己和左边的token。右边的padding天然被这个下三角结构隔断,根本不需要attention_mask额外去屏蔽它。
    # 所以在右padding+decoder模型的训练场景下,attention_mask全为1和正确设置等价,真正起作用的只有labels里的-100。

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

from tqdm import tqdm
class SimpleSFTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_data,
        output_dir="./checkpoints",
        max_length=512,
        batch_size=2,
        learning_rate=2e-5,
        num_epochs=3,
        save_steps=100,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        device=None,
    ):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # if gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable()
            # 用时间换显存的技术。用时间换显存的技术。
            # 开启gradient checkpointing之后,前向传播时不保存中间激活值,只保存少数几个"检查点"位置的结果。
            # 反向传播需要某一层的激活值时,从最近的检查点重新跑一次前向把它算出来,用完就扔。

        if gradient_checkpointing:
            # LoRA场景下base model被冻结,输入端默认没有梯度,
            # 必须显式打开,否则gradient checkpointing会导致梯度链路断裂
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()


        dataset = SFTDataset(train_data, tokenizer, max_length=max_length)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        )

        # self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )


        # 整个训练过程里optimizer.step()会被调用的总次数
        # 每攒够 gradient_accumulation_steps 个batch才会真正调用一次optimizer.step()
        total_steps = (len(self.dataloader) // gradient_accumulation_steps) * num_epochs
        # 举个具体数字:
        # 假如self.dataloader一共有100个batch,    gradient_accumulation_steps=4,  num_epochs=3,
        # 那么total_steps = (100 // 4) * 3 = 25 * 3 = 75
        # 也就是说整个训练只会真正更新75次权重,而不是300次(3个epoch乘100个batch)。
        # 这个数字必须算对,因为调度器要根据它来规划"什么时候该把学习率降到0"。

        # 学习率调度器,行为分两段:
        # 前num_warmup_steps步,学习率从0线性爬升到你在AdamW里设的初始学习率(比如2e-5);
        # 爬升结束后,从num_training_steps减去warmup剩下的步数里,学习率再线性下降,一直降到0,刚好在最后一步(也就是total_steps)降到0。
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, int(total_steps * 0.03)),
            num_training_steps=max(1, total_steps),
        )
        # 接着上面的例子,num_warmup_steps = max(1, int(75 * 0.03)) = max(1, 2) = 2,
        # 意味着前2次权重更新,学习率是从0慢慢爬到2e-5的,而不是一开始就用满学习率;
        # 从第3步到第75步,学习率再匀速降回0。

        # 为什么要这样设计:
        # 训练刚开始时模型参数(尤其是新初始化的LoRA层或者刚松绑的参数)离损失最优点还很远,梯度方向往往不稳定,
        # 如果一上来就用满学习率,Adam这类优化器的二阶矩估计还没积累够统计量,很容易出现loss剧烈震荡甚至直接发散——warmup相当于让优化器先"看几步路再加速"。
        # 后段线性衰减到0则是让模型在训练快结束时步子越走越小,避免在最优点附近来回震荡,有利于收敛得更精细。

        # 两处max(1, ...)是防止边界情况报错:如果你的数据集很小,导致total_steps算出来是0(比如batch数还不够一次梯度累积),调度器内部按0步去算线性衰减斜率会出问题

        os.makedirs(output_dir, exist_ok=True)

    def save_checkpoint(self, tag):
        save_path = os.path.join(self.output_dir, f"checkpoint-{tag}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"已保存checkpoint到 {save_path}")

    def train(self):
        self.model.to(self.device)
        self.model.train()

        self.model.config.use_cache = False  # 训练时一次性处理完整序列，不需要缓存历史KV，节省显存
        # attention 理论计算中：
        # 输入 X (B, N, D)
        # → 通过 WQ/WK/WV
        # → 得到 Q (B, N, D)、K (B, N, D)、V (B, N, D)

        # Decoder 模型采用自回归方式逐 token 生成下一词，因此必须一步步生成（不能并行生成未来 token）。
        # 例子：生成“早上好”
        # use_cache = True：

        # Step 1：生成“早”
        # 序列：[早]（N=1）
        # 缓存：1个 token 的 K/V（早）
        #
        # Step 2：生成“上”
        # 序列：[早, 上]（N=2）
        # 缓存：K/V:[早]+[上]    复用前面1个 token 的 K/V（早），仅新增“上”的 K/V
        #
        # Step 3：生成“好”
        # 输入：[早, 上, 好]
        # 缓存：K/V:[早,上]+[好]    复用前面2个 token 的 K/V（早、上），仅新增“好”的 K/V

        # use_cache = False：
        # 每一步都重新计算当前完整序列的 K/V（不保留历史）
        # 比如：生成“好”的时候，重新计算[早,上]的K/V

        # 推理：逐 token 生成，需要实时等待输出（用户交互/生成文本），延迟直接影响体验，
        #       所以目标是“越快越好” → 通过 KV cache 复用历史计算来加速。
        #
        # 训练：一次性处理长序列，计算本身是离线的，时间相对可接受，
        #       但显存决定能不能跑、batch size 能不能上去 → 显存是硬瓶颈。

        global_step = 0
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc=f"epoch {epoch}",
            )
            for batch_idx, batch in progress_bar:
            # for batch_idx, batch in enumerate(self.dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # {
                #     "input_ids": tensor形状(2, 18),
                #     "labels": tensor形状(2, 18),
                #     "attention_mask": tensor形状(2, 18),
                # }
                # batch.items()遍历这3个键值对。把这3个tensor分别搬到GPU(或者CPU,取决于self.device)

                # 训练时model(**batch)调用的是model.forward(),是PyTorch nn.Module的标准协议,
                # (**batch)只是把字典展开成关键字参数的语法糖,等价于:
                # outputs = self.model(input_ids=batch["input_ids"],labels=batch["labels"],attention_mask=batch["attention_mask"],)
                outputs = self.model(**batch)
                # 推理时你用的model.generate(ids)是HuggingFace在GenerationMixin里实现的一个高层方法,
                # 内部封装了自回归循环——每次forward一步,取出logits,采样或者greedy选出下一个token,
                # 把这个token拼回input_ids,再forward下一步,直到遇到eos或者达到max_new_tokens。
                # 你看到的是"输入一段prompt,输出一段完整的新文本"。

                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    real_loss = loss.item() * self.gradient_accumulation_steps
                    progress_bar.set_postfix(step=global_step, loss=f"{real_loss:.4f}")

                    if global_step % 10 == 0:
                        # real_loss = loss.item() * self.gradient_accumulation_steps
                        print(f"epoch {epoch} step {global_step} loss {real_loss:.4f}")

                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)

        self.save_checkpoint("final")



import importlib.util
def import_things(file,class_name):
    spec = importlib.util.spec_from_file_location(
        "data_preparation",
        file
    )
    data_preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_preparation)

    # 然后从这个module对象里取出你要的类
    out_class = getattr(data_preparation, class_name)

    return out_class



from peft import LoraConfig, get_peft_model, TaskType

def stf_train_main():
    GSM8K_SFT_Dataset = import_things('1. data_preparation.py','GSM8K_SFT_Dataset')
    LLMModelLoader = import_things('2. LLM_preparation.py','LLMModelLoader')

    loader = LLMModelLoader(model_id="Qwen/Qwen3-0.6B", local_root="./LLM_models")
    model, tokenizer = loader.get_model_and_tokenizer()


    train_data = GSM8K_SFT_Dataset(split="train", max_samples=None)
    # print(train_data[0])


    # 加载模型之后,传进trainer之前包一层
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Qwen3的attention投影层
        # Transformer的attention层一共有4个投影矩阵:q_proj、k_proj、v_proj、o_proj。
        # 论文做了消融实验,在同样的参数预算下(LoRA的总秩固定),对比"只调q和v"、"调全部4个但每个秩更小"等不同组合的下游任务效果,
        # 发现q_proj+v_proj这个组合性价比最高

    )
    model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()  # 确认可训练参数量

    trainer = SimpleSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        output_dir="./checkpoints",
        batch_size=2,
        num_epochs=3,
        save_steps=50,
        gradient_checkpointing=True,  # 8GB显存建议开启
    )
    trainer.train()


def check_model_layers():
    LLMModelLoader = import_things('2. LLM_preparation.py','LLMModelLoader')
    loader = LLMModelLoader(model_id="Qwen/Qwen3-0.6B", local_root="./LLM_models")
    model, tokenizer = loader.get_model_and_tokenizer()
    for name, _ in model.named_modules():
        print(name)

def load_model_from_checkpoint(checkpoint,merge=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import AutoPeftModelForCausalLM

    adapter_path = f"./checkpoints/checkpoint-{checkpoint}"

    # 之所以能省略base model路径,是因为adapter_config.json里其实已经记录了base_model_name_or_path这个字段
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)

    if merge:
        merged_model = model.merge_and_unload()
        return merged_model,tokenizer
    return model,tokenizer


import torch


class LLM_wrapper:
    def __init__(self, model, tokenizer):
        model.to("cuda")
        self.model = model.eval()
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat(self, query, max_new_tokens=256, do_sample=False, temperature=1.0, top_p=0.95):
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # 只解码新生成的部分,去掉重复的prompt
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response




if __name__ == '__main__':
    # stf_train_main()

    model,tokenizer = load_model_from_checkpoint(2800)

    agent = LLM_wrapper(model,tokenizer)

    # query = "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nLet's solve this step by step:\n"

    # query = "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nLet's solve this step by step:\n"

    query = "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nLet's solve this step by step:\n"
    response = agent.chat(query)
    print(response)

    print('======================')
    LLMModelLoader = import_things('2. LLM_preparation.py','LLMModelLoader')
    loader = LLMModelLoader(model_id="Qwen/Qwen3-0.6B", local_root="./LLM_models")
    model2, tokenizer2 = loader.get_model_and_tokenizer()
    agent2 = LLM_wrapper(model2,tokenizer2)

    response = agent2.chat(query)
    print(response)



