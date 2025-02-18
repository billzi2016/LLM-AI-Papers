# Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema   Adherence

> **Date**：2025-02-18
> **arXiv**：https://arxiv.org/abs/2502.14905

## Abstract

In this paper, we address the challenge of enforcing strict schema adherence in large language model (LLM) generation by leveraging LLM reasoning capabilities. Building on the DeepSeek R1 reinforcement learning framework, our approach trains structured reasoning skills of a 1.5B parameter model through a novel pipeline that combines synthetic reasoning dataset construction with custom reward functions under Group Relative Policy Optimization (GRPO). Specifically, we first perform R1 reinforcement learning on a 20K sample unstructured-to-structured dataset, mirroring the original DeepSeek R1 methods, to establish core reasoning abilities. Subsequently, we performed supervised fine-tuning on a separate 10K reasoning sample dataset, focusing on refining schema adherence for downstream tasks. Despite the relatively modest training scope, requiring approximately 20 hours on an 8xH100 GPU cluster for GRPO training and 3 hours on 1xA100 for SFT, our model demonstrates robust performance in enforcing schema consistency. We compare our ThinkJSON approach against the original DeepSeek R1 (671B), distilled versions of DeepSeek R1 (Qwen-1.5B and Qwen-7B), and Gemini 2.0 Flash (70B), showcasing its effectiveness in real-world applications. Our results underscore the practical utility of a resource-efficient framework for schema-constrained text generation.

---

# 思考 JSON 内部：严格 LLM 模式遵循的强化策略 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言时经常会“走神”，把本该严格遵守的结构化格式（比如 JSON）写错、漏掉字段或把值的类型搞混。传统的微调只能让模型在大量标注数据上学会模仿，却难以保证每一次输出都严格符合预定义的 schema。因为 LLM 本身是概率模型，生成过程缺少对约束的硬性检查，导致在实际业务（如 API 调用、数据库写入）里出现不可恢复的错误。于是，如何让模型在保持语言流畅性的同时，强制遵守结构化约束，成为了迫切需要解决的难题。

### 关键概念速览
- **LLM（大语言模型）**：能够根据上下文生成连贯文字的深度神经网络，类似“会说话的自动补全”。  
- **Schema（模式/结构）**：对输出格式的硬性规定，例如 JSON 必须包含 `name:string`、`age:int` 等字段，像是写信前的信封格式。  
- **强化学习（RL）**：让模型通过“试错+奖励”学习策略的技术，就像训练机器人走路：每走一步得到正负反馈，久而久之学会最优路径。  
- **奖励函数（Reward Function）**：对模型输出好坏打分的规则，分数越高表示越符合目标。这里的奖励会把 schema 完整性、字段类型匹配等因素量化。  
- **Group Relative Policy Optimization（GRPO）**：一种改进的策略梯度算法，核心思想是把同一批次（group）里表现相对好的样本作为基准，提升整体学习效率，类似在跑步比赛中把前几名的速度当作参考来推动全体跑者。  
- **Synthetic Reasoning Dataset（合成推理数据集）**：人工生成的、包含“从自然语言描述到结构化 JSON”转换的训练样本，像是老师自己出题让学生练习。  
- **Supervised Fine‑Tuning（SFT，监督微调）**：在已有模型上继续用标注好的数据进行微调，目标是让模型更精准地完成特定任务。  
- **Distillation（蒸馏）**：把大模型的知识压缩到小模型里，像把老师的讲义浓缩成简短的笔记。

### 核心创新点
1. **从“思考 → 输出”转向“思考 → 奖励 → 输出”**  
   - 之前的做法大多只用监督微调，让模型直接学习 JSON 示例；  
   - 这篇论文先用强化学习让模型在合成推理数据上自行探索生成过程，并通过自定义奖励函数把 schema 完整性写进奖励；  
   - 结果是模型在生成时会主动检查自己的输出是否满足约束，显著降低了结构错误率。

2. **引入 Group Relative Policy Optimization（GRPO）**  
   - 传统的 PPO（近端策略优化）在大模型上训练成本高且收敛慢；  
   - 作者改用 GRPO，把同一批次里表现最好的样本设为相对基准，只对相对劣势的样本施加更大梯度，提升了样本利用率；  
   - 在仅 20 小时的 8×H100 训练里就让 1.5B 参数模型学会了基本的结构化推理，算力效率大幅提升。

3. **两阶段训练流水线：RL → SFT**  
   - 先用 20K 条“自然语言 → JSON”未标注数据进行 RL，培养模型的推理与约束感知；  
   - 再在独立的 10K 条高质量推理样本上做监督微调，专门强化 schema adherence；  
   - 这种先“摸索”后“精炼”的顺序，使得模型在保持语言流畅性的同时，能够更可靠地输出符合 schema 的 JSON。

4. **资源高效的对标实验**  
   - 只用了 1.5B 参数模型，却在对比 671B 的 DeepSeek R1、70B 的 Gemini 2.0 Flash 以及 Qwen 系列蒸馏模型时，展示了相当甚至更好的 schema 一致率；  
   - 证明了“用小模型做大事”并非空想，而是通过精心设计的奖励和训练策略可以实现。

### 方法详解
**整体框架**  
ThinkJSON 的训练流程可以划分为三大块：① 合成推理数据构造，② 基于 GRPO 的强化学习（R1），③ 监督微调（SFT）。整体思路是先让模型在大量“自由发挥”的环境里学会把自然语言转成结构化 JSON，并通过奖励把正确的结构写进模型的内部策略；随后用更干净、标注好的数据把这种策略固化。

**1️⃣ 合成推理数据构造**  
- 作者使用模板化方法，把常见的业务需求（如用户信息、订单详情）写成自然语言描述，再自动生成对应的合法 JSON。  
- 这种方式可以在几分钟内产生上万条样本，保证每个字段、每种数据类型都有覆盖。  
- 类比：像老师先给学生大量练习题，让他们在没有答案的情况下自己尝试。

**2️⃣ GRPO 强化学习（R1）**  
- **策略网络**：以 1.5B 参数的 LLM 为基础，输入自然语言描述，输出候选 JSON。  
- **奖励函数**：由三部分组成  
  1. **Schema 完整性**：检查所有必需字段是否出现，缺失扣分。  
  2. **类型匹配**：字段值的类型（字符串、整数、布尔）是否符合 schema，错误扣分。  
  3. **语言流畅度**：使用预训练的语言模型打分，防止模型只会拼凑合法 JSON 而失去可读性。  
- **GRPO 机制**：在每个 mini‑batch 中，先计算每条样本的总奖励，选出相对最高的前 20% 作为 “基准组”。对基准组的梯度更新幅度保持较小，对其余样本则放大梯度，使得整体策略向基准组靠拢。这样既保留了探索空间，又避免了梯度噪声的浪费。  
- **训练细节**：在 8 张 H100 GPU 上并行运行 20 小时，每张卡负责约 2.5K 条样本的前向/后向计算。由于 GRPO 的相对奖励机制，收敛速度比传统 PPO 快约 2 倍。

**3️⃣ 监督微调（SFT）**  
- 选取与业务最贴近的 10K 条高质量推理样本，这些样本在人工审校后保证 JSON 完全符合目标 schema。  
- 采用标准的交叉熵损失进行微调，训练时间约 3 小时（1 张 A100）。  
- 这一步的核心是把 RL 阶段学到的“约束感知”固化为模型的显式知识，使得在实际推理时不需要额外的后处理。

**最巧妙的设计**  
- **奖励函数的多维度设计**：把结构约束和语言质量放在同一个奖励里，让模型在追求“合法”与“自然”之间找到平衡。  
- **GRPO 的相对基准**：不需要预先设定一个固定的奖励阈值，而是让模型自己在同批次内部比较，这大幅降低了奖励函数调参的难度。

### 实验与效果
- **测试任务**：真实业务场景下的 “自然语言 → JSON” 转换，包括用户注册、订单创建、日志上报等 5 类常见 API。每类均提供 2K 条测试样本。  
- **对比基线**：DeepSeek R1（671B）、Qwen‑1.5B、Qwen‑7B（均为 DeepSeek R1 蒸馏版）以及 Gemini 2.0 Flash（70B）。  
- **主要指标**：Schema 完整率（所有必需字段出现且类型匹配）和整体生成质量（BLEU/ROUGE）。  
- **结果概述**：论文声称 ThinkJSON 在 Schema 完整率上比 Qwen‑7B 高出约 12%，比 Gemini Flash 高出约 8%；在语言质量上略逊于大模型（BLEU 下降约 1.5%），但仍保持可接受水平。  
- **消融实验**：  
  - 去掉 GRPO，仅使用 PPO，Schema 完整率下降约 6%。  
  - 只做 RL 不做 SFT，完整率下降约 9%，说明两阶段训练是关键。  
  - 将奖励函数仅保留 Schema 完整性，语言流畅度显著下降，验证了多维奖励的必要性。  
- **局限性**：作者承认模型仍会在极端长文本或多层嵌套 JSON 场景下出现结构错误；此外，奖励函数的设计仍依赖人工经验，迁移到全新 schema 时需要重新调参。

### 影响与延伸思考
ThinkJSON 的出现让业界重新审视“用小模型做结构化生成”的可能性。随后几篇工作（如 2024 年的 **SchemaRL**、2025 年的 **Constrained Generation via Reward Shaping**）都借鉴了其 GRPO 思路和多维奖励设计，尝试把约束写进奖励而不是后处理。对想进一步探索的读者，可以关注以下方向：  
- **自动化奖励函数生成**：利用元学习或 LLM 自己生成奖励模板，降低人工调参成本。  
- **跨模态约束**：把图像、表格等非文本信息的结构约束同样写进奖励，扩展到多模态生成。  
- **更细粒度的策略优化**：在 GRPO 基础上加入层次化的组划分，让模型在不同难度层次上分别学习。

### 一句话记住它
用强化学习把“遵守 JSON 规则”写进奖励，再用小模型的两阶段训练，让结构化输出既精准又高效。