import re
from typing import List


def extract_answer(text: str):
    """
    优先提取:
        Final Answer: 5

    如果没有 Final Answer:
    则退化为提取最后一个数字
    """

    match = re.search(
        r"Final\s*Answer\s*:\s*\$?\s*(-?\d+\.?\d*)",
        text,
        re.IGNORECASE,
    )

    if match:
        try:
            return float(match.group(1))
        except:
            pass

    numbers = re.findall(r"-?\d+\.?\d*", text)

    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass

    return None


def accuracy_reward(completion: str, truth: str):
    """
    主奖励（最重要）

    完全正确: 2.0
    接近正确: 1.5
    大致正确: 1.0
    有一定接近: 0.3
    错误: 0
    """

    pred = extract_answer(completion)

    if pred is None:
        return 0.0

    try:
        truth = float(truth)
    except:
        return 0.0

    error = abs(pred - truth)

    if error < 0.01:
        return 2.0

    if error < 1:
        return 1.5

    if error < 5:
        return 1.0

    if error < 20:
        return 0.3

    return 0.0


def format_reward(completion: str):
    """
    格式奖励

    鼓励:
    - think
    - Final Answer
    """

    score = 0.0

    if "<think>" in completion:
        score += 0.1

    if "Final Answer:" in completion:
        score += 0.3

    return score


def reasoning_reward(completion: str):
    """
    推理奖励

    不强制检查算式
    只奖励明显的推理过程
    """

    score = 0.0

    lines = [
        line.strip()
        for line in completion.split("\n")
        if line.strip()
    ]

    if len(lines) >= 3:
        score += 0.1

    if len(lines) >= 6:
        score += 0.1

    keywords = [
        "therefore",
        "because",
        "first",
        "then",
        "so",
        "thus",
        "因此",
        "首先",
        "然后",
        "所以",
    ]

    completion_lower = completion.lower()

    if any(keyword.lower() in completion_lower for keyword in keywords):
        score += 0.1

    return score


def length_reward(completion: str):
    """
    长度控制

    防止:
    - 太短直接摆烂
    - 无限思维链
    """

    length = len(completion)

    if length < 20:
        return -0.2

    if length < 400:
        return 0.2
    if length < 500:
        return 0.1

    # if length < 600:
    #     return -0.3

    return -0.3


def reward_func(completions: List[str], **kwargs):
    """
    SimpleGRPOTrainer 调用接口

    reward_func(
        completions,
        ground_truth=repeated_truths
    )
    """

    ground_truths = kwargs.get("ground_truth", [])

    rewards = []

    for completion, truth in zip(completions, ground_truths):

        reward = 0.0

        # 正确性（主奖励）
        reward += accuracy_reward(
            completion,
            truth,
        )

        # 格式奖励
        reward += format_reward(
            completion,
        )

        # 推理奖励
        reward += reasoning_reward(
            completion,
        )

        # 长度控制
        reward += length_reward(
            completion,
        )

        rewards.append(float(reward))

    return rewards