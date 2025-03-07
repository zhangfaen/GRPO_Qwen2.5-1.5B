# 从零开始写GRPO算法 | 强化学习15亿参数小模型，复现DeepSeek-R1-Zero的“啊哈”时刻 | 推理能力从14%提升到86%

## 快速开始
```bash
%conda create -n grpo  python=3.12 -y
%conda activate grpo
%pip install torch transformers datasets accelerate
%pip install flash-attn --no-build-isolation
%python grpo.py
```

近来，在AI领域，强化学习（Reinforcement Learning, RL）强势崛起。特别是在大模型中，强化学习通过试错和奖励机制，能够帮助大模型在复杂任务中表现出更智能的行为。3月6日，2024年图灵奖也颁给了强化学习。

DeepSeek-R1-Zero展示了强化学习算法帮助大模型的推理方面出现“啊哈”时刻。所谓“啊哈”时刻，指的是模型在训练过程中突然“开窍”，不仅能给出正确答案，还能展示出清晰的推理过程。

千行代码一个脚本，在小模型上，尝试复现DeepSeek R1 Zero的“啊哈”时刻。原理类似于QwQ-32B，无需使用CoT数据微调，模型通过强化学习自己“悟出来”：若要推理，需要思考。简单来说：蒸馏是学，左右互搏式的强化学习是悟。

## 强化学习 vs. 监督学习：为什么强化学习更“聪明”？

在深入源代码之前，咱们先聊聊强化学习和监督学习的核心区别，以及为什么强化学习在某些场景下能训练出更智能的模型。

### 监督学习：依赖标记数据的“被动学习者”

监督学习是传统的机器学习方法，依赖于大量标记好的数据。例如，在数学问题求解任务中，咱们需要准备许多问题及其正确答案，模型通过学习这些数据来预测答案。这种方法的优点是简单直接，但也有明显的局限性：

- **数据需求高**：需要大量高质量的标记数据，标注成本高昂。
- **泛化能力有限**：模型可能过度拟合训练数据，面对新问题时表现不佳。
- **缺乏探索性**：监督学习本质上是“被动”接收信息，难以应对需要多步推理或动态调整的复杂任务。

### 强化学习：试错中的“主动智者”

强化学习则完全不同。它不依赖预先标记的数据，而是通过与环境的交互来学习最优策略。模型（称为代理或者策略）在环境中执行动作，根据结果获得奖励信号，并通过优化累积奖励来改进行为。在语言模型微调中，这意味着模型可以通过生成答案、评估正确性并调整策略，逐步提升性能。

强化学习的优势在于：

- **试错能力**：通过探索和利用，模型能发现监督学习难以捕捉的复杂模式。
- **长期规划**：适合需要多步决策的任务，例如数学推理或游戏策略。
- **自适应性**：模型可以根据奖励信号动态调整，适应多样化的任务需求。


本文将从零开始，展示从零实现GRPO（Group Relative Policy Optimization）算法——一种由DeepSeek团队开发的高效强化学习方法——来训练一个15亿参数的小模型，复现DeepSeek-R1-Zero的“啊哈”时刻。

我写了一个Python脚本（https://github.com/zhangfaen/GRPO_Qwen2.5-1.5B/blob/main/grpo.py ），实现GRPO算法，强化学习Qwen2.5-1.5B-Instruct这个15亿参数的模型。强化学习前，这个模型在GSM8K数据集上的准确度为14%左右，强化学习后，其准确度可以达到86%左右。
<img width="611" alt="image" src="https://github.com/user-attachments/assets/9a372723-1c6b-4158-a88f-8eeba8f13b06" />


注：GSM8K是一个由OpenAI发布的数据集，有8500个高质量数学问题组成。这些问题需要2到8个步骤来解决，解决方法主要是使用基本的算术运算(+-/*)进行一连串的基本计算，以得出最终答案。虽然看起来很简单，但很多大模型的表现都不太好。

接下来，咱们看看通过GRPO算法，如何让模型在数学任务中学会生成正确答案。这个过程有点老顽童的“左右互搏”提升武功，也有点“左脚踩右脚，就能飞起来”的感觉，哈哈。 实际上，运行我写的脚步，完成强化学习的过程是这样的：针对任何一个数学题，不需要标注任何长CoT（思维链）数据，只需要让模型尽情的探索生成多个答案（我的代码中是8个），一个奖励函数判断模型生成的各个答案是否正确，正确就奖励模型，否则就惩罚模型，模型的推理能力就慢慢的变强了。

训练过程使用了1个80GB的A800 GPU，运行了10个小时左右。

## GRPO算法简介

GRPO（Group Relative Policy Optimization）是一种强化学习算法，专门用于训练大模型，使其在数学、逻辑和编程等需要精确答案的任务上表现出色。其核心思想是通过生成多个候选答案（completions），与标准答案（ground truth）比较，并根据匹配程度分配奖励，从而优化模型的生成策略。

### GRPO的基本原理

GRPO的运行机制可以概括为以下步骤：

1. **生成多个样本**：对于每个输入问题，模型生成若干个可能的答案。
2. **奖励评估**：将生成的答案与正确答案对比，根据正确性（correctness）和格式（format）等指标计算奖励。
3. **策略优化**：利用奖励信号，通过GRPO算法更新模型参数，增加生成正确答案的概率。

GRPO的关键创新在于“群体相对”（group relative）优化，它通过比较多个生成结果的相对优劣，结合KL散度惩罚（防止模型偏离原始行为过远），实现高效的策略改进。

## 脚本概览

脚本的目标是将Qwen2.5-1.5B-Instruct模型从通用语言模型转变为数学问题求解专家。以下是脚本的主要组成部分：

- **基本设置和导入**
- **数据格式化和答案提取**
- **数据集准备**
- **评估函数**
- **奖励函数**
- **训练设置和执行**
- **模型加载和测试**

接下来，我将逐一剖析这些部分，结合代码，剖析GRPO的实现过程。

## 基本设置和导入

脚本设置随机种子以确保结果可重复：

```python
# 设置随机种子
def set_random_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_random_seed(42)
```

## 数据格式化和答案提取

为了让模型生成结构化的输出，我定义了一个系统提示，要求答案以`<reasoning>`和`<answer>`标签包裹：

```python
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
```

接着，定义两个函数提取答案：

- **模型输出答案提取**：从生成文本中提取`<answer>`标签内的内容。
- **数据集答案提取**：从GSM8K数据集中提取标准答案（以`####`分隔）。

```python
def extract_answer_from_model_output(text):
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    answer = parts[-1].split("</answer>")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
```

这些函数确保模型输出和数据集答案能够统一比较，为后续奖励计算奠定基础。

## 数据集准备

咱们使用GSM8K数据集，一个包含8500个数学问题的集合。脚本将其格式化为包含提示和答案的示例：

```python
def prepare_dataset(split="train"):
    data = load_dataset('openai/gsm8k', 'main')[split]
    ......
    return formatted_data

def build_prompt(messages):
    return "\n".join([msg["content"].strip() for msg in messages])
```

训练前咱们从GSM8K中抽取了50个样本用于评估，其余用于训练。

## 评估函数

评估函数用于衡量模型在训练前后的性能。它生成答案，提取结果，并与标准答案比较，计算准确率：

```python
def evaluate_model(model, tokenizer, eval_examples, device):
    model.eval()
    correct = 0
    total = len(eval_examples)
    for example in eval_examples:
        .....
    accuracy = (correct / total) * 100
    return accuracy
```

## 奖励函数：强化学习的灵魂

奖励函数是强化学习的核心，决定模型优化的方向。咱们定义了两个奖励函数，并将其组合使用：

1. **正确性奖励（correctness_reward）**：
   - 完全匹配标准答案：2.0分
   - 数值相等但格式不同：1.5分
   - 错误：0分

```python
def correctness_reward(prompts, completions, answer):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        .....
    return rewards
```

2. **格式奖励（format_reward）**：
   - 鼓励模型遵循`<reasoning>`和`<answer>`格式，每出现一个标签加0.05分。

```python
def format_reward(completions):
    ......
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.05
        if "</reasoning>" in response: score += 0.05
        if "<answer>" in response: score += 0.05
        if "</answer>" in response: score += 0.05
        rewards.append(score)
    return rewards
```

3. **组合奖励**：
   - 将正确性和格式奖励相加，综合引导模型行为。

```python
def combined_reward(prompts, completions, answer):
    correctness_scores = correctness_reward(prompts, completions, answer)
    format_scores = format_reward(completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]
```

这种设计既强调答案的准确性，又确保输出格式规范，符合实际应用需求。这个也参考了DeepSeek R1的奖励函数设计：答案正确性 + 格式规范性。

## 训练设置和执行：GRPO的实现

训练部分是脚本的核心，实现了GRPO算法的完整流程。以下是关键函数：

- **生成样本**：`generate_completions`为每个问题生成多个答案。
- **计算对数概率**：`compute_log_probs`计算模型生成答案的概率。
- **GRPO损失**：`grpo_loss`结合奖励和KL散度，计算优化目标。
- **训练循环**：`train_with_grpo`执行多轮策略更新。

训练配置中，我设置了`num_generations=8`（每问题生成8个答案），`max_completion_length=400`（允许较长的推理过程），体现了GRPO的多样本优化特性。

## 模型加载和测试

训练完成后，模型被保存并测试：

```python
def test_fine_tuned_model():
    .....
    for prompt in prompts:
        test_prompt = build_prompt([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        inputs = tokenizer.encode(test_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=400)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
```

## 下一步

我展示了GRPO的基本实现，但后面仍有改进空间：

- **奖励设计**：增加停止生成（EOS）的奖励，避免冗长输出。
- **训练规模**：增加迭代次数或样本量，提升准确率。
- **混合训练**：结合监督学习和强化学习，先用标注数据预训练，再用GRPO微调。

## 总结

这篇文章，我展示了从零开始实现了GRPO算法，成功微调了一个15亿参数的语言模型。强化学习相较于监督学习的优势在于其探索性和自适应性，而GRPO通过多样本优化和奖励机制，为语言模型注入了更强的推理能力。训练结果显示准确率从14%提升到86%，测试用例表明模型已初步具备数学推理的“啊哈”时刻。

希望这篇文章能给大家带来一点启发。无论是复现脚本还是进一步优化GRPO，欢迎在评论区分享实验结果或问题。
