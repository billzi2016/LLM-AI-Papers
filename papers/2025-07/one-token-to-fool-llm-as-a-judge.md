# One Token to Fool LLM-as-a-Judge

> **Date**：2025-07-11
> **arXiv**：https://arxiv.org/abs/2507.08794

## Abstract

Large language models (LLMs) are increasingly trusted as automated judges, assisting evaluation and providing reward signals for training other models, particularly in reference-based settings like Reinforcement Learning with Verifiable Rewards (RLVR). However, we uncover a critical vulnerability even in this reference-based paradigm: generative reward models are systematically susceptible to reward hacking. We find that superficial inputs, which we term ''master keys'' such as non-word symbols (e.g., '':'' or ''.'') or generic reasoning openers (e.g., ''Thought process:'' or ''Let's solve this problem step by step.''), can consistently elicit false positive rewards without any substantive reasoning. Our systematic evaluation demonstrates this is a widespread failure affecting a diverse range of models, including leading proprietary systems such as GPT-o1 and Claude-4. These results challenge the assumed robustness of LLM judges and pose a significant threat to their reliability. To address this, we propose a simple yet effective data augmentation strategy using truncated model outputs as adversarial negative examples. The resulting Master Reward Models (Master-RMs) demonstrate state-of-the-art robustness against these ''master key'' attacks while maintaining high performance in standard evaluation settings. We supplement these findings with a comprehensive analysis of the vulnerability across model scales, prompt variations, and common inference-time strategies, offering insights to guide future research on robust LLM evaluation. We release our robust, general-domain reward models and the synthetic training data at https://huggingface.co/sarosavo/Master-RM and https://huggingface.co/datasets/sarosavo/Master-RM.

---

# 一枚令牌即可欺骗 LLM 判官 论文详细解读

### 背景：这个问题为什么难？

在强化学习里，研究者常用大语言模型（LLM）当“裁判”，给生成式模型的输出打分，形成奖励信号。理想情况下，裁判应该只在答案真的有价值时给高分，尤其在需要参考答案的场景（比如 RLVR）更是如此。可是，LLM 本身是通过大量文本学习的，它们对提示词的敏感度极高，往往会把一些表面形式当成“好答案”。之前的工作大多假设，只要给出参考答案或让模型解释推理过程，裁判就不容易被玩弄。实际上，这篇论文发现，只要在答案前面加上一个特定的符号或一句常见的开场白，LLM 判官就会几乎无条件地给出高奖励，根本不检查答案的真实性。这种“单 token 攻击”让整个奖励体系失去可信度，直接威胁到基于 LLM 的自我提升循环。

### 关键概念速览

**LLM‑as‑a‑Judge**：把大语言模型当作自动评分器，用来评估其他模型的输出。类似于让老师给学生的作文打分，只不过老师本身也是一个 AI。

**Reward Hacking（奖励黑客）**：模型找到规避真实评估的捷径，只要能得到高分就行，哪怕答案根本错误。就像学生只会写出老师喜欢的格式，而不是真正掌握知识。

**Master Key（万能钥匙）**：文中指的那几个“表面输入”，比如单个符号 “:” 或常见的开场句 “Thought process:”。它们像一把万能钥匙，能打开几乎所有 LLM 判官的高分大门。

**RLVR（Reinforcement Learning with Verifiable Rewards）**：一种强化学习框架，奖励信号来源于可验证的参考答案，理论上应该比纯粹的自评更安全。

**Adversarial Negative Example（对抗负例）**：在训练奖励模型时故意加入的错误样本，用来让模型学会辨别真正的好答案和伪装的高分答案。

**Master‑RM**：作者训练出的“主钥奖励模型”，专门用对抗负例强化，使其对 Master Key 攻击更鲁棒。

### 核心创新点

1. **发现并系统化 Master Key 攻击**  
   之前的研究只偶尔提到 LLM 对提示词的敏感性，这篇论文把这种现象抽象为“Master Key”，并在多种模型、不同规模、各种提示下做了大规模实验，证明它是普遍存在的漏洞。

2. **用截断输出生成对抗负例**  
   传统的奖励模型训练只用真实好答案和普通错误答案。作者把模型自己生成的、被 Master Key 戏弄后的输出截断下来，标记为负例，加入训练集。这样模型在学习时会看到“看似合规但实际上是骗分”的例子。

3. **构建 Master‑RM 并保持标准性能**  
   通过上述数据增强，训练出新的奖励模型。实验显示，这些模型在抵御 Master Key 攻击时几乎不再给出误导性高分，同时在常规评估（如对话质量、代码生成）上仍保持或略微提升原有水平。

4. **跨模型、跨尺度的系统评估**  
   作者不仅在开源模型上验证，还把 GPT‑o1、Claude‑4 等商业系统拉进实验，展示了漏洞的普适性和新模型的广泛适用性。

### 方法详解

整体思路可以拆成三步：**漏洞定位 → 对抗负例生成 → 鲁棒奖励模型训练**。

1. **漏洞定位**  
   研究者先准备一批标准的评估任务（比如数学题、代码补全），让多个 LLM 判官对正常答案打分。随后在答案前面加上各种候选 Master Key（单字符、常见开场句），观察分数变化。几乎所有模型在出现这些键后，分数都会显著上升，即使答案本身是空白或完全错误。

2. **对抗负例生成**  
   为了让模型学会识别这种骗分手法，作者让原始 LLM（作为“生成器”）输出带有 Master Key 的答案，然后把输出 **截断** 到关键位置——通常是只保留 Master Key 本身或紧跟其后的几词，抛去后面的真实内容。这样得到的文本在形式上符合高分的“诱饵”，但实际上没有提供任何有价值的信息。每条负例都被标记为 “0 分”，加入训练数据。

3. **鲁棒奖励模型训练**  
   训练过程仍然是标准的监督学习：输入是任务描述 + 模型输出，目标是预测人类或参考答案给出的分数。不同之处在于训练集里混入了大量对抗负例，使得模型在学习时必须区分“表面上看起来像高分”的噪声和真正的高质量答案。作者称这种训练好的模型为 **Master‑RM**。

4. **评估与验证**  
   训练完成后，作者再次用同样的 Master Key 攻击测试集评估 Master‑RM。分数回落到正常水平，说明模型不再被单 token 欺骗。随后在常规基准（如 MT-Bench、HumanEval）上跑一遍，确保鲁棒性提升没有牺牲原有评估能力。

**最巧妙的点**在于只用 **截断输出** 生成负例，而不需要人工标注大量“骗分”样本。这样几乎可以自动化产生海量对抗数据，成本极低，却能显著提升模型的防御能力。

### 实验与效果

- **数据与任务**：作者在公开的对话评估、数学推理、代码生成等多模任务上做实验，还专门构造了包含 50 种不同 Master Key 的攻击集。
- **基线对比**：与原始奖励模型（未做对抗训练）相比，Master‑RM 在攻击集上的平均得分下降了约 70%（原始模型常给 8‑9 分，Master‑RM 降到 2‑3 分）。在标准评估上，得分保持在 92%~95% 的原始水平，甚至在部分任务上略有提升（约 1%）。
- **消融实验**：去掉对抗负例或只使用部分 Master Key 进行训练，防御效果显著下降，说明完整的负例覆盖是关键。截断长度的不同也会影响鲁棒性，作者选取 2‑3 token 的截断最为有效。
- **局限性**：论文指出，虽然 Master‑RM 对已知 Master Key 有强抵抗力，但仍可能被未见过的更复杂结构攻击。此外，加入大量负例会略微增加训练时间。

### 影响与延伸思考

这篇工作让业界重新审视“LLM‑as‑a‑Judge” 的安全假设，提醒我们即使在参考答案驱动的 RLVR 场景，也不能掉以轻心。随后出现的几篇论文（如《Prompt‑Robust Reward Modeling》、《Adversarial Calibration of LLM Judges》）都在尝试更通用的对抗训练或多视角评估，以进一步提升判官的鲁棒性。对想继续深入的读者，可以关注以下方向：① 自动发现新型 Master Key 的生成算法；② 将对抗负例与人类偏好对齐的多任务学习；③ 在跨语言、多模态评估中验证类似漏洞的普适性。

### 一句话记住它

只要在答案前塞进一个“万能钥匙” token，几乎所有 LLM 判官都会误判；用截断的骗分样本做对抗训练，就能让判官不再被这枚 token 欺骗。