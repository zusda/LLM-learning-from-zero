




import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class RLPromptDataset(Dataset):
    """只需要prompt和ground_truth,不需要completion(模型自己生成)。

    对应你的GSM8K_RL_Dataset.load_or_create()产出的数据格式:
    {"prompt": ..., "ground_truth": ..., "question": ..., "full_answer": ...}
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {"prompt": example["prompt"], "ground_truth": example["ground_truth"]}


def rl_collate_fn(batch):
    # prompt和ground_truth保持字符串list,不在这里tokenize
    # 因为同一个prompt要生成G个不同completion,tokenize放到生成阶段统一处理
    return {
        "prompts": [item["prompt"] for item in batch],
        "ground_truths": [item["ground_truth"] for item in batch],
    }


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

# reward_func = import_things('3. reward_preparation.py', 'reward_func')
reward_func = import_things('3. reward_2.py', 'reward_func')
from time import time
class SimpleGRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_data,
        reward_func,
        output_dir="./grpo_checkpoints",
        batch_size=4,
        num_generations=4,  # 每个prompt生成几条completion(GRPO的"组"大小)
        max_new_tokens=256,
        learning_rate=1e-6,
        num_epochs=1,
        save_steps=50,
        gradient_accumulation_steps=1,
        kl_coef=0.0,  # 设为0则不使用KL惩罚,简化版默认关闭;需要稳定性时设0.01~0.1
        gradient_checkpointing=False,
        device=None,
        show_time=False
    ):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # 生成时左padding,保证所有序列右端对齐方便截取completion

        self.show_time = show_time
        self.model = model
        self.tokenizer = tokenizer
        self.reward_func = reward_func
        self.output_dir = output_dir
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.kl_coef = kl_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # "全量训练兼容、LoRA也兼容"的通用写法
        if gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
            # # gradient_checkpointing_enable()内部有这样的逻辑:只要开启了gradient checkpointing,就会自动把config.use_cache设为False并打印警告(原因是gradient checkpointing和KV cache在训练时的forward逻辑上是有冲突的)

        dataset = RLPromptDataset(train_data)
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=rl_collate_fn
        )

        # "全量训练兼容、LoRA也兼容"的通用写法
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        total_steps = (len(self.dataloader) // gradient_accumulation_steps) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, int(total_steps * 0.03)),
            num_training_steps=max(1, total_steps),
        )

        os.makedirs(output_dir, exist_ok=True)

    def save_checkpoint(self, tag):
        save_path = os.path.join(self.output_dir, f"checkpoint-{tag}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"已保存checkpoint到 {save_path}")

    # ---------- 第一步:为每个prompt生成G条completion ----------
    def generate_completions(self, prompts):
        # 每个prompt重复G次,一次性batch生成,效率比for循环高
        repeated_prompts = [p for p in prompts for _ in range(self.num_generations)]

        inputs = self.tokenizer(
            repeated_prompts,
            return_tensors="pt",  # 让tokenizer直接返回PyTorch tensor,而不是普通的python list
            padding=True,   # 因为这16条prompt长度不一样,要padding到同一batch内的最大长度才能拼成矩形tensor喂给模型。
                            # 结合我们之前设的tokenizer.padding_side = "left",padding会补在左边。
                # 只要传了padding=True,tokenizer就会在padding的同时同步产出一个对应的mask,告诉模型"哪些位置是真实token,哪些是补的pad"

            truncation=True, # 如果某条prompt长度超过tokenizer.model_max_length,会从那头截断,防止个别异常长的样本撑爆显存或者报错。
            add_special_tokens=False,  # chat template已经包含特殊token
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens, # 限制最多生成多少个新token(不算prompt长度),防止模型生成停不下来,无限占用显存和时间。
                do_sample=True, # 开启随机采样,而不是贪婪解码(do_sample=False会每次都选概率最高的token,结果完全确定)。这一点对GRPO是必须的——同一个prompt要生成
                temperature=1.0, # 控制采样的"随机程度"。数值越大,生成结果越随机发散;越接近0,越趋向于贪婪解码那种确定性选择。
                                 # 1.0是不做额外缩放的原始分布,是比较中性的默认值,
                                 # 如果你发现生成内容过于离谱可以调低(比如0.7),
                                 # 如果发现生成内容缺乏多样性、组内方差太小可以调高。
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,  # 必须显式传True,覆盖gradient_checkpointing_enable()
                # 强制设置的config.use_cache=False,否则生成会退化成
                # 每个新token都重算整个序列,速度暴跌(这才是真正的根因)
                # cache_implementation="static",
            )
        self.model.train()

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = generated[:, prompt_len:]  # 只取新生成的部分
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # inputs["input_ids"] = tensor([
        #     [PAD, PAD, PAD, PAD, tok1, tok2, tok3, tok4, tok5],   # prompt1,前面补4个pad
        #     [tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8, tok9] # prompt2,刚好9个,不需要pad
        # ])

        # inputs["attention_mask"] = tensor([
        #     [0, 0, 0, 0, 1, 1, 1, 1, 1],   # 前4个pad位置标0,后5个真实token标1
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1]    # 全部都是真实token
        # ])

        # generated = tensor([
        #     [PAD, PAD, PAD, PAD, tok1, tok2, tok3, tok4, tok5, new1, new2, new3, new4, new5, new6],
        #     [tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8, tok9, new1, new2, new3, new4, new5, new6]
        # ])

        # completions_text = [
        #     "1+1等于2。",
        #     "中国的首都是北京,推理过程是..."
        # ]

        return inputs["input_ids"], inputs["attention_mask"], completion_ids, completions_text

    # ---------- 第二步:算reward,按组做归一化得到advantage ----------
    def compute_advantages(self, completions_text, ground_truths):
        # ground_truths需要按num_generations重复对齐
        repeated_truths = [gt for gt in ground_truths for _ in range(self.num_generations)]
        # 举例：repeated_truths = ["2", "2", "2", "2", "北京", "北京", "北京", "北京", "25", "25", "25", "25"]

        # 根据 生成的结果 与 答案 计算奖励。 这部分的内部操作后续需要修改，因为提取答案的方式似乎跟之前sft训练的不一样
        rewards = self.reward_func(completions_text, ground_truth=repeated_truths)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # 按组(每num_generations条为一组,对应同一个prompt)做归一化
        # 这是GRPO的核心:不需要value model,直接用组内相对好坏作为advantage
        num_prompts = len(ground_truths)
        rewards = rewards.view(num_prompts, self.num_generations)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-4  # 防止除0
        advantages = (rewards - mean) / std
        advantages = advantages.view(-1)  # 展平回(num_prompts * num_generations,)

        return advantages, rewards.view(-1)

    # ---------- 第三步:重新算log prob(generate不保留梯度,必须重新forward) ----------
    def compute_log_probs(self, prompt_ids, prompt_mask, completion_ids):
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        outputs = self.model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
        # outputs.logits的形状是(batch_size, seq_len, vocab_size),最后一维就是词表大小

        # 是把最后一个位置(位置5)的logits砍掉,只保留位置0~4这5个"有真实答案可核对"的预测:
        logits = outputs.logits[:, :-1, :]  # 预测下一个token,所以logits要错位
        # 举例：位置0的token，经过LLM后计算得到logit(0) 这个logint会转换成token，然后放到位置1上面。
        # 那么，full_ids中位置1所在的token就是与位置0的logit匹配的一对。

        # 把full_ids的第一个token(位置0对应的101)砍掉,保留位置1~5这5个token作为"真实答案":
        targets = full_ids[:, 1:]
        # (batch_size, prompt_len + completion_len) id只是个数字，因此只有2维，无需像token一样用第三维表示

        log_probs = F.log_softmax(logits, dim=-1)

        # token id 就是token在vocab中的位置
        # 从log_probs这个(batch, seq_len, vocab_size)的大张量里,每个位置只挑出"真实token对应的那一个数值",把维度从词表大小(可能十几万)压缩到1。
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)   ).squeeze(-1)
        # token_log_probs的大小是（batch, seq_len-1），里面每个数都是对应答案的log_prob （错位对齐之后的长度-1，因此seq_len-1）

        # 只保留completion部分的log prob(prompt部分不需要算梯度来源)
        prompt_len = prompt_ids.shape[1]

        completion_log_probs = token_log_probs[:, prompt_len - 1 :]
        # 第一个token是没有任何logit与它对应的，组不成token-logit对，因此才会产生之前偏差
        # 因此我们在计算len(prompt)后，需要减去1，这才是 token-logit对 中propmt的长度。

        completion_mask_for_loss = full_mask[:, prompt_len:]
        # 下面是一组token，和一组logits，l2可以通过softmax等方式变成t2，因此l2和t2是一对的
        # 可以发现 仅中间4对构成匹配的数据。
        # 我们以t1，t2,t3为prompt举例，我们要获得completion的mask。则full_mask中将prompt_len长度的前面部分截断即可
        # completion_log_probs中，token_log_probs内容是每个匹配的数据，所以长度是下面重合部分，因此要去除prompt部分的token_log_probs，将prompt_len - 1长度的前面部分截断即可
        # [
        #   t1,t2,t3,  t4,t5
        #      l2,l3,  l4,l5,l6
        # ]

        # 经过以上操作
        # completion_log_probs中仅存在 l4,l5对应的log_prob值
        # completion_mask_for_loss 仅存在 t4,t5 两个位置的mask。（其实和前面completion_mask是等价的）.
        # 由于completion_mask 是从completion ids里面生成的，因此也仅有t4、t5两个位置的mask
        return completion_log_probs, completion_mask_for_loss

    def train(self):
        self.model.to(self.device)
        print("模型实际所在设备:", next(self.model.parameters()).device)
        # print("self.device:", self.device)
        # self.model.config.use_cache = False
        # 这个设置是全局生效的,会一直持续到训练结束
        # generate()正常情况下依赖KV cache做增量生成:生成第N个token时,只需要计算这一个新token的attention,前面N-1个token的key/value直接复用缓存,整体复杂度是线性的。
        # 一旦use_cache=False,模型每生成一个新token,都要把从头到尾的整个序列重新算一遍attention,生成256个token的总计算量从"线性"变成了"近似平方级别"
        t=[]


        global_step = 0
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"epoch {epoch}")
            for batch_idx, batch in progress_bar:
                # batch.keys() : dict_keys(['prompts', 'ground_truths'])
                # len(batch["prompts"]) = len(batch["ground_truths"]) = batch_size

                if self.show_time:
                    t.append(time())
                prompt_ids, prompt_mask, completion_ids, completions_text = self.generate_completions(
                    batch["prompts"]
                )
                if self.show_time:
                    t.append(time())
                    print('1.generate生成答案耗时：',t[1]-t[0])
                # prompt_ids.shape = (batch_size*num_generations, prompt_steps)
                # completion_ids.shape = (batch_size*num_generations, response_steps)
                # len(completions_text) = batch_size * num_generations

                advantages, raw_rewards = self.compute_advantages(completions_text, batch["ground_truths"])
                # advantages.shape = raw_rewards.shape = (batch_size*num_generations,)
                if self.show_time:
                    t.append(time())
                    print('2.计算advantage耗时：',t[2]-t[1])

                log_probs, completion_mask = self.compute_log_probs(prompt_ids, prompt_mask, completion_ids)
                # log_probs.shape = completion_mask.shape = (batch_size * num_generations, response_steps)
                if self.show_time:
                    t.append(time())
                    print('3.计算log_prob耗时：：',t[3]-t[2])

                # policy gradient loss: -advantage * log_prob,只在completion有效token上算,按token数做平均
                per_token_loss = -advantages.unsqueeze(1) * log_probs
                per_token_loss = per_token_loss * completion_mask  # 把pad位置的loss乘以0,直接清零。


                # completion_mask.sum()算的正是"这个batch里总共有多少个有效token
                # 用总loss除以有效token数,得到的是"平均每个token的loss",这样不管这次batch生成的内容长还是短,算出来的loss都是同一个量纲下可比较的数值,
                loss = per_token_loss.sum() / completion_mask.sum().clamp(min=1)
                loss = loss / self.gradient_accumulation_steps

                loss.backward()
                if self.show_time:
                    t.append(time())
                    print('4.loss耗时：',t[4]-t[3])
                    self.show_time=False


                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    mean_reward = raw_rewards.mean().item()
                    progress_bar.set_postfix(
                        step=global_step,
                        loss=f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                        reward=f"{mean_reward:.3f}",
                    )

                    if global_step % 10 == 0:
                        print(
                            f"epoch {epoch} step {global_step} "
                            f"loss {loss.item() * self.gradient_accumulation_steps:.4f} "
                            f"mean_reward {mean_reward:.3f}"
                        )

                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)

        self.save_checkpoint("final")


def load_model_from_checkpoint(checkpoint,merge=False,dir="checkpoints"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import AutoPeftModelForCausalLM

    adapter_path = f"./{dir}/checkpoint-{checkpoint}"

    # 之所以能省略base model路径,是因为adapter_config.json里其实已经记录了base_model_name_or_path这个字段
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)

    if merge:
        merged_model = model.merge_and_unload()
        return merged_model,tokenizer
    return model,tokenizer

from peft import LoraConfig, get_peft_model, TaskType
def rl_train_main():
    model, tokenizer = load_model_from_checkpoint('final',merge=True)

    GSM8K_RL_Dataset = import_things('1. data_preparation.py','GSM8K_RL_Dataset')
    # LLMModelLoader = import_things('2. LLM_preparation.py','LLMModelLoader')

    # loader = LLMModelLoader(model_id="Qwen/Qwen3-0.6B", local_root="./LLM_models")
    # model, tokenizer = loader.get_model_and_tokenizer()

    train_data = GSM8K_RL_Dataset(split="train", max_samples=20)
    # - prompt: 应用chat template后的文本字符串
    # - ground_truth: 正确答案
    # - question: 原始问题
    # - full_answer: 完整答案



    # 加载模型之后,传进trainer之前包一层
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    #     GRPO通常参数更新信号很弱。

    )
    model = get_peft_model(model, lora_config)


    trainer = SimpleGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        reward_func=reward_func,
        batch_size=1,
        num_generations=4,
        max_new_tokens=612,
        gradient_checkpointing=True,
        save_steps=100,
        gradient_accumulation_steps=1,
        show_time=True
    )
    trainer.train()

class LLM_wrapper:
    def __init__(self, model, tokenizer):
        model.to("cuda")
        self.model = model.eval()
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat(self, query, max_new_tokens=612, do_sample=False, temperature=1.0, top_p=0.95):
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


def test():
    model,tokenizer = load_model_from_checkpoint(checkpoint='final',dir='grpo_checkpoints')

    agent = LLM_wrapper(model,tokenizer)

    # query = "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nLet's solve this step by step:\n"

    # query = "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nLet's solve this step by step:\n"

    query = "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nLet's solve this step by step:\n"
    response = agent.chat(query)
    print(response)

    print('======================')
    # LLMModelLoader = import_things('2. LLM_preparation.py','LLMModelLoader')
    # loader = LLMModelLoader(model_id="Qwen/Qwen3-0.6B", local_root="./LLM_models")
    # model2, tokenizer2 = loader.get_model_and_tokenizer()
    # agent2 = LLM_wrapper(model2,tokenizer2)
    #
    # response = agent2.chat(query)
    # print(response)

    model2,tokenizer2 = load_model_from_checkpoint(checkpoint='final')

    agent2 = LLM_wrapper(model2,tokenizer2)
    response = agent2.chat(query)
    print(response)


if __name__ == '__main__':
    rl_train_main()

    # test()












