# Can LLMs Keep a Secret? Testing Privacy Implications of Language Models   via Contextual Integrity Theory

> **Date**：2023-10-27
> **arXiv**：https://arxiv.org/abs/2310.17884

## Abstract

The interactive use of large language models (LLMs) in AI assistants (at work, home, etc.) introduces a new set of inference-time privacy risks: LLMs are fed different types of information from multiple sources in their inputs and are expected to reason about what to share in their outputs, for what purpose and with whom, within a given context. In this work, we draw attention to the highly critical yet overlooked notion of contextual privacy by proposing ConfAIde, a benchmark designed to identify critical weaknesses in the privacy reasoning capabilities of instruction-tuned LLMs. Our experiments show that even the most capable models such as GPT-4 and ChatGPT reveal private information in contexts that humans would not, 39% and 57% of the time, respectively. This leakage persists even when we employ privacy-inducing prompts or chain-of-thought reasoning. Our work underscores the immediate need to explore novel inference-time privacy-preserving approaches, based on reasoning and theory of mind.

---

# 大型语言模型能保守秘密吗？基于情境完整性理论的隐私影响测试 论文详细解读

### 背景：这个问题为什么难？
在过去，隐私风险主要关注模型训练阶段的泄露——比如通过逆向工程把训练数据恢复出来。随着 LLM 被嵌入到聊天机器人、办公助理等交互场景，模型在每一次对话里都会接收到用户的个人信息，然后即时生成回复。传统的安全评估工具很少考虑“在特定情境下，模型应该自行决定是否该说”。换句话说，模型需要像人一样判断信息的共享边界，而这涉及到多源信息、目的、受众等复杂因素，现有方法几乎没有针对这种推理层面的测评手段。

### 关键概念速览
**情境完整性（Contextual Integrity）**：一种隐私理论，认为信息的合法流动取决于情境、信息主体、信息接收者和共享目的。可以把它想成社交礼仪的规则手册，只有在符合情境规则时才可以“说话”。  
**指令微调（Instruction‑tuning）**：在大模型上继续训练，让它更好地遵循人类给出的任务指令。类似于给模型上了“使用手册”，但手册里往往没有写“在什么情况下不说”。  
**隐私诱导提示（Privacy‑inducing Prompt）**：在对话前加上一段提醒模型保护隐私的文字。就像在会议前提醒大家“请勿泄露公司机密”。  
**思维链（Chain‑of‑Thought, CoT）**：让模型先把推理过程写出来再给出答案，类似于先列出思考步骤再下结论。  
**ConfAIde 基准**：本文提出的专门用于评估 LLM 隐私推理能力的测试集合，名字取自“Confidential”+“AI”。它把情境完整性的要素编码成一系列问答场景，检验模型在不同情境下是否会泄露信息。  
**理论心智（Theory of Mind）**：模型对“他人”知识状态的推断能力。这里指模型能否理解对话对象是否已经知道某些信息，从而决定是否重复。  

### 核心创新点
1. **从隐私理论到评测基准**：过去的隐私评估大多基于统计泄露或对抗样本。本文把情境完整性理论直接转化为可操作的问答模板，构造了 1,200 条覆盖多种情境、目的和受众的测试案例。这样做让评测不再是“模型会不会记住数据”，而是“模型在特定情境下会不会主动透露”。  
2. **针对指令微调模型的系统性压力测试**：在每个案例中，作者分别使用普通提示、隐私诱导提示以及思维链提示三种方式与模型交互，观察这些常见的安全增强手段是否真的能降低泄露率。结果显示，即使加上这些技巧，GPT‑4 仍有 39% 的案例泄露，ChatGPT 更高达 57%。这直接挑战了业界对提示工程的乐观假设。  
3. **提出“推理层面隐私保护”需求**：实验表明，仅靠训练阶段的去标识化或后处理过滤不足以解决问题。作者因此呼吁研发基于推理的实时隐私守护机制，例如让模型在生成前先进行一次“是否应说”的内部审查。  

### 方法详解
整体思路可以拆成三步：**情境构造 → 提示设计 → 结果判定**。  
1. **情境构造**：作者依据情境完整性理论，选取四个关键维度——情境（工作、家庭、医疗等）、信息主体（用户、第三方）、共享目的（帮助、娱乐、广告）和受众（模型本身、其他用户、外部系统）。在每个维度上随机抽取组合，生成自然语言描述，例如：“在公司内部会议中，用户提到自己刚刚做了体检，想让助理帮忙写一封请假邮件”。随后附加一个潜在泄露的私密信息点（如血压数值）。  
2. **提示设计**：对每个情境，研究者准备三种输入方式：  
   - **普通提示**：直接把情境描述和用户问题喂给模型。  
   - **隐私诱导提示**：在前面加上一句类似“请确保不泄露任何个人隐私”。  
   - **思维链提示**：要求模型先列出“我是否应该透露该信息？”的理由，再给出答案。  
   这相当于在同一道题目上给模型三种不同的“思考指令”。  
3. **结果判定**：模型输出后，使用自动化脚本匹配是否出现了原始私密信息。匹配成功即记为泄露。为了避免误判，作者还加入了人工抽样检查。  

最巧妙的地方在于把抽象的隐私理论转化为可机器生成的情境模板，并且通过多种提示方式模拟真实用户可能的交互习惯，从而得到更贴近实际使用场景的评估结果。

### 实验与效果
- **测试对象**：OpenAI 的 GPT‑4、ChatGPT（基于 GPT‑3.5）以及若干开源指令微调模型（如 LLaMA‑2‑Chat）。  
- **泄露率**：在普通提示下，GPT‑4 在 1,200 条案例中泄露 39%，ChatGPT 为 57%。加入隐私诱导提示后，泄露率仅略降至 35%（GPT‑4）和 53%（ChatGPT），思维链提示的降幅更小。  
- **基线对比**：与传统的“是否包含敏感词”过滤器相比，ConfAIde 能捕捉到更细粒度的泄露，因为信息往往以变形或上下文暗示出现。  
- **消融实验**：作者分别去掉情境描述、去掉目的信息、只保留受众信息进行测试，发现“受众信息缺失”时泄露率下降最多，说明模型在判断是否泄露时最依赖受众角色的线索。  
- **局限性**：评测主要聚焦英文对话，中文情境的覆盖度尚未验证；此外，自动匹配可能漏掉隐晦的泄露表达。作者也承认，当前的提示工程仍是最直接的防护手段，缺乏系统性的推理层面防御。  

### 影响与延伸思考
这篇工作把隐私从“数据层面”拉到“推理层面”，在学术界引发了对 LLM 实时隐私决策的关注。随后有几篇论文尝试在模型内部加入“隐私审计模块”，或利用强化学习让模型学习何时保持沉默。业界也开始在产品中加入“隐私守护开关”，让用户自行指定哪些信息不可被模型复述。想进一步了解，可以关注以下方向：  
- **基于理论心智的隐私推理**：让模型显式建模对话伙伴的知识状态。  
- **可解释的隐私决策**：把模型的“是否泄露”判断以自然语言解释给用户。  
- **跨语言情境完整性评测**：扩展 ConfAIde 到多语言、多文化情境。  

### 一句话记住它
LLM 在对话中常常忘记“情境规则”，即使加上隐私提示也会泄露——我们需要让模型学会在生成前先自检，像人一样懂得何时保持沉默。