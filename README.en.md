# Writing the GRPO Algorithm from Scratch | Reinforcement Learning with a 1.5 Billion Parameter Small Model, Replicating DeepSeek-R1-Zero's 'Aha' Moment | Reasoning Ability Improves from 14% to 86%

**Quick Start**

```bash
%conda create -n grpo  python=3.12 -y
%conda activate grpo
%pip install torch transformers datasets accelerate
%pip install flash-attn --no-build-isolation
%python grpo.py
```

## Main Text Begins

Recently, in the field of AI, reinforcement learning (Reinforcement Learning, RL) has been on the rise. Especially in large models, reinforcement learning, through trial and error and reward mechanisms, can help large models exhibit more intelligent behavior in complex tasks. On March 6, 2024, the Turing Award was also given to reinforcement learning.

DeepSeek-R1-Zero demonstrated how reinforcement learning algorithms can lead to an 'aha' moment in the reasoning aspect of large models. The so-called 'aha' moment refers to the model suddenly 'getting it' during training, not only providing correct answers but also showcasing a clear reasoning process.

With a thousand lines of code in a single script, we attempt to replicate DeepSeek R1 Zero's 'aha' moment on a small model. The principle is similar to QwQ-32B, where without fine-tuning on CoT (Chain of Thought) data, the model 'figures it out' through reinforcement learning: to reason, one needs to think. Simply put: distillation is learning, while the self-play style of reinforcement learning is enlightenment.

## Reinforcement Learning vs. Supervised Learning: Why is RL 'Smarter'?

Before diving into the source code, let’s first discuss the core differences between reinforcement learning and supervised learning, and why reinforcement learning can train smarter models in certain scenarios.

### Supervised Learning: A 'Passive Learner' Reliant on Labeled Data

Supervised learning is a traditional machine learning method that relies on a large amount of labeled data. For example, in mathematical problem-solving tasks, we need to prepare many problems along with their correct answers, and the model learns from these data to predict answers. The advantages of this method are simplicity and directness, but it also has obvious limitations:

- **High Data Requirements**: Requires a large amount of high-quality labeled data, which is costly to annotate.
- **Limited Generalization**: The model may overfit the training data and perform poorly on new problems.
- **Lack of Exploration**: Supervised learning is essentially 'passive' in receiving information, making it difficult to handle complex tasks that require multi-step reasoning or dynamic adjustments.

### Reinforcement Learning: An 'Active Sage' in Trial and Error

Reinforcement learning is entirely different. It does not rely on pre-labeled data but learns optimal strategies through interaction with the environment. The model (called an agent or policy) performs actions in the environment, receives reward signals based on the outcomes, and improves its behavior by optimizing cumulative rewards. In language model fine-tuning, this means the model can generate answers, evaluate their correctness, and adjust its strategy to gradually improve performance.

The advantages of reinforcement learning include:

- **Trial and Error Capability**: Through exploration and exploitation, the model can discover complex patterns that are hard for supervised learning to capture.
- **Long-term Planning**: Suitable for tasks requiring multi-step decision-making, such as mathematical reasoning or game strategies.
- **Adaptability**: The model can dynamically adjust based on reward signals, adapting to diverse task requirements.

This article will demonstrate, from scratch, how to implement the GRPO (Group Relative Policy Optimization) algorithm—a highly efficient reinforcement learning method developed by the DeepSeek team—to train a 1.5 billion parameter small model and replicate DeepSeek-R1-Zero's 'aha' moment.

I have written a Python script (https://github.com/zhangfaen/GRPO_Qwen2.5-1.5B/blob/main/grpo.py) that implements the GRPO algorithm to reinforce learn the Qwen2.5-1.5B-Instruct model with 1.5 billion parameters. Before reinforcement learning, this model achieved about 14% accuracy on the GSM8K dataset; after reinforcement learning, its accuracy can reach around 86%.

<img width="611" alt="image" src="https://github.com/user-attachments/assets/9a372723-1c6b-4158-a88f-8eeba8f13b06" />

**Note**: GSM8K is a dataset released by OpenAI, consisting of 8,500 high-quality math problems. These problems require 2 to 8 steps to solve, mainly using basic arithmetic operations (+-/*) in a series of calculations to reach the final answer. Although it seems simple, many large models do not perform well on it.

Next, let’s see how, through the GRPO algorithm, the model learns to generate correct answers in mathematical tasks. This process is somewhat like the "self-play" of the old master improving martial arts skills, or like "stepping on the left foot with the right foot to fly up," haha. In reality, running my script, the reinforcement learning process is as follows: for any math problem, without labeling any long CoT (Chain of Thought) data, just let the model freely explore and generate multiple answers (8 in my code), a reward function judges whether each generated answer is correct, rewarding the model if correct, punishing if not, and the model's reasoning ability gradually strengthens.

The training process used one 80GB A800 GPU and ran for about 10 hours.

## Introduction to the GRPO Algorithm

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm specifically designed to train large models to excel in tasks requiring precise answers, such as mathematics, logic, and programming. Its core idea is to generate multiple candidate answers (completions), compare them with the ground truth, and assign rewards based on the degree of matching, thereby optimizing the model's generation strategy.

### Basic Principles of GRPO

The operation of GRPO can be summarized in the following steps:

1. **Generate Multiple Samples**: For each input question, the model generates several possible answers.
2. **Reward Evaluation**: Compare the generated answers with the correct answer and calculate rewards based on metrics such as correctness and format.
3. **Policy Optimization**: Use the reward signals to update the model parameters through the GRPO algorithm, increasing the probability of generating correct answers.

The key innovation of GRPO lies in "group relative" optimization, which compares the relative merits of multiple generated results and combines it with a KL divergence penalty (to prevent the model from deviating too far from the original behavior), achieving efficient policy improvement.

## Script Overview

The goal of the script is to transform the Qwen2.5-1.5B-Instruct model from a general language model into a math problem-solving expert. Below are the main components of the script:

- **Basic Settings and Imports**
- **Data Formatting and Answer Extraction**
- **Dataset Preparation**
- **Evaluation Functions**
- **Reward Functions**
- **Training Setup and Execution**
- **Model Loading and Testing**

Next, I will dissect each part, combining the code to analyze the implementation process of GRPO.

## Basic Settings and Imports

The script sets a random seed to ensure reproducibility:

```python
# Set random seed
def set_random_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_random_seed(42)
```

## Data Formatting and Answer Extraction

To make the model generate structured output, I defined a system prompt requiring answers to be wrapped in `<reasoning>` and `<answer>` tags:

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

Then, two functions are defined to extract answers:

- **Model Output Answer Extraction**: Extracts the content within the `<answer>` tags from the generated text.
- **Dataset Answer Extraction**: Extracts the standard answer from the GSM8K dataset (delimited by `####`).

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

These functions ensure that the model output and dataset answers can be uniformly compared, laying the foundation for subsequent reward calculations.

## Dataset Preparation

We use the GSM8K dataset, a collection of 8,500 math problems. The script formats it into examples containing prompts and answers:

```python
def prepare_dataset(split="train"):
    data = load_dataset('openai/gsm8k', 'main')[split]
    ......
    return formatted_data

def build_prompt(messages):
    return "\n".join([msg["content"].strip() for msg in messages])
```

Before training, we extracted 50 samples from GSM8K for evaluation, and the rest for training.

## Evaluation Functions

The evaluation function is used to measure the model's performance before and after training. It generates answers, extracts the results, compares them with the standard answers, and calculates the accuracy:

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

## Reward Functions: The Soul of Reinforcement Learning

The reward function is the core of reinforcement learning, determining the direction of model optimization. We defined two reward functions and combined them:

1. **Correctness Reward**:
   - Exact match with the standard answer: 2.0 points
   - Numerically equal but different format: 1.5 points
   - Incorrect: 0 points

```python
def correctness_reward(prompts, completions, answer):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        .....
    return rewards
```

2. **Format Reward**:
   - Encourages the model to follow the `<reasoning>` and `<answer>` format, adding 0.05 points for each tag present.

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

3. **Combined Reward**:
   - Adds the correctness and format rewards to comprehensively guide the model's behavior.

```python
def combined_reward(prompts, completions, answer):
    correctness_scores = correctness_reward(prompts, completions, answer)
    format_scores = format_reward(completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]
```

This design emphasizes both the accuracy of the answers and the standardization of the output format, meeting practical application needs. This also references DeepSeek R1's reward function design: answer correctness + format standardization.

## Training Setup and Execution: Implementing GRPO

The training part is the core of the script, implementing the complete process of the GRPO algorithm. Below are the key functions:

- **Generating Samples**: `generate_completions` generates multiple answers for each question.
- **Computing Log Probabilities**: `compute_log_probs` calculates the probability of the model generating each answer.
- **GRPO Loss**: `grpo_loss` combines rewards and KL divergence to calculate the optimization objective.
- **Training Loop**: `train_with_grpo` executes multiple rounds of policy updates.

In the training configuration, I set `num_generations=8` (generating 8 answers per question), `max_completion_length=400` (allowing longer reasoning processes), reflecting GRPO's multi-sample optimization characteristic.

## Model Loading and Testing

After training, the model is saved and tested:

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

## Next Steps

I have demonstrated the basic implementation of GRPO, but there is still room for improvement:

- **Reward Design**: Add rewards for stopping generation (EOS) to avoid lengthy outputs.
- **Training Scale**: Increase the number of iterations or samples to improve accuracy.
- **Mixed Training**: Combine supervised learning and reinforcement learning, pretraining with labeled data followed by GRPO fine-tuning.

## Summary

This article demonstrated implementing the GRPO algorithm from scratch to successfully fine-tune a 1.5 billion parameter language model. Reinforcement learning, compared to supervised learning, has advantages in exploration and adaptability, and GRPO, through multi-sample optimization and reward mechanisms, injects stronger reasoning abilities into the language model. The training results show that accuracy improved from 14% to 86%, and test cases indicate that the model has initially achieved the 'aha' moment in mathematical reasoning.

I hope this article can bring some inspiration to everyone. Whether replicating the script or further optimizing GRPO, feel free to share your experimental results or questions in the comments.

---

This translation retains the full Markdown structure, including headings, code blocks, lists, and emphasis, while keeping all code sections unchanged as per the user's instructions. The English text is designed to be clear, fluent, and faithful to the original meaning.
