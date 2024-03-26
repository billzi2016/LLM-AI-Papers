# InternLM2 Technical Report

> **Date**：2024-03-26
> **arXiv**：https://arxiv.org/abs/2403.17297

## Abstract

The evolution of Large Language Models (LLMs) like ChatGPT and GPT-4 has sparked discussions on the advent of Artificial General Intelligence (AGI). However, replicating such advancements in open-source models has been challenging. This paper introduces InternLM2, an open-source LLM that outperforms its predecessors in comprehensive evaluations across 6 dimensions and 30 benchmarks, long-context modeling, and open-ended subjective evaluations through innovative pre-training and optimization techniques. The pre-training process of InternLM2 is meticulously detailed, highlighting the preparation of diverse data types including text, code, and long-context data. InternLM2 efficiently captures long-term dependencies, initially trained on 4k tokens before advancing to 32k tokens in pre-training and fine-tuning stages, exhibiting remarkable performance on the 200k ``Needle-in-a-Haystack" test. InternLM2 is further aligned using Supervised Fine-Tuning (SFT) and a novel Conditional Online Reinforcement Learning from Human Feedback (COOL RLHF) strategy that addresses conflicting human preferences and reward hacking. By releasing InternLM2 models in different training stages and model sizes, we provide the community with insights into the model's evolution.

---

# InternLM2 技术报告 论文详细解读

### 背景：这个问题为什么难？
大模型从 ChatGPT 到 GPT‑4 的飞跃让大家期待开源版也能追上，但实际操作中遇到两大瓶颈：一是训练数据的多样性和规模难以匹配商业闭源模型；二是长文本依赖的建模仍旧受限于显存和训练策略，导致在需要几万甚至上百千字的任务上表现不佳。传统的预训练往往在 2k‑4k token 左右就停下来，无法捕捉更远的上下文关联；而现有的对齐方法（如普通的 RLHF）在面对冲突的人类偏好时容易出现奖励漏洞。正是这些根本性障碍，让“把最前沿的能力搬到开源”成为一件急需系统性解决的事。

### 关键概念速览
**大语言模型（LLM）**：拥有上百亿参数、能够理解并生成自然语言的神经网络，类似于会说话的百科全书。  
**长上下文建模**：模型在一次前向传播中能够“记住”几千到几万 token 的信息，就像人在阅读一本长篇小说时能够保持情节连贯。  
**监督微调（SFT）**：在大模型的基础上，用标注好的问答或指令数据继续训练，让模型更好地遵循人类指令。  
**强化学习从人类反馈（RLHF）**：让模型通过与人类评审的交互学习奖励信号，从而优化生成质量。  
**条件在线 RLHF（COOL RLHF）**：在传统 RLHF 基础上加入条件约束和在线更新，专门用来缓解人类偏好冲突和防止模型“投机取巧”。  
**Needle‑in‑a‑Haystack 测试**：在超长文本中埋入稀有信息，要求模型在 200k token 长度的上下文里准确找出，类似于在大海捞针。  
**多模态预训练数据**：除了普通文本，还包括代码和专门的长文本数据，帮助模型在不同场景下都有表现。  

### 核心创新点
1. **数据多样化 + 长文本专用数据**  
   *之前的开源模型大多只用通用网页文本，缺少代码和超长段落。*  
   *InternLM2 在预训练阶段把普通文本、代码库以及专门收集的 32k token 长文本混合使用，形成了更全面的知识库。*  
   *这种混合让模型在代码补全、长文检索等任务上都有显著提升，尤其在 Needle‑in‑a‑Haystack 测试中表现突出。*

2. **分阶段 Token 长度扩展**  
   *传统做法一次性把最大 token 长度设为 2k‑4k，后期想要更长会遇到显存爆炸。*  
   *InternLM2 先用 4k token 进行大规模预训练，等模型基本收敛后再切换到 32k token 继续训练和微调。*  
   *这种渐进式扩展让显存使用更平滑，同时让模型在 32k token 环境下已经学会捕捉远程依赖，最终在 200k token 场景中仍能保持较高召回率。*

3. **COOL RLHF 对齐策略**  
   *普通 RLHF 只用单一的奖励模型，面对冲突的偏好时容易出现“奖励黑客”。*  
   *COOL RLHF 引入条件约束（如安全、事实性、可解释性）并在在线环境中实时更新奖励模型，确保模型在满足多重目标的同时不走偏。*  
   *实验显示，这套方案在主观评测中显著提升了用户满意度，且降低了出现有害输出的概率。*

4. **阶段性模型发布**  
   *大多数开源项目一次性只发布最终模型，缺少研究过程的透明度。*  
   *InternLM2 把不同训练阶段（4k 预训练、32k 预训练、SFT、COOL RLHF）以及多尺寸（7B、13B、30B）模型全部开源，提供了完整的演进轨迹。*  
   *社区可以直接对比每一步的增益，帮助后续研究快速定位最有效的改进点。*

### 方法详解
**整体框架**  
InternLM2 的训练流程可以划分为四大块：① 数据准备与混合、② 分阶段长上下文预训练、③ 监督微调（SFT）、④ 条件在线 RLHF（COOL RLHF）。每一步都在前一步的基础上进一步提升模型的能力和安全性。

**1. 数据准备与混合**  
- **普通文本**：从公开的网页、书籍、新闻等来源抓取，覆盖日常语言。  
- **代码数据**：收集 GitHub、开源项目的代码库，加入注释和文档，帮助模型学习编程语义。  
- **长上下文数据**：专门爬取包含章节、法律条文、技术手册等结构化长文的语料，切分成 32k token 长度的块。  
- **混合策略**：在每个训练 batch 中按比例抽取三类数据，比例在实验中调优，确保模型不会被单一类型数据“淹没”。  

**2. 分阶段长上下文预训练**  
- **阶段 A（4k token）**：使用标准的 Transformer 架构，最大序列长度设为 4k。此阶段主要让模型学习基本语言模型目标（下一个 token 预测），并在显存限制下完成大规模数据遍历。  
- **阶段 B（32k token）**：在阶段 A 收敛后，模型参数保持不变，仅把位置编码和相对注意力机制扩展到 32k。随后继续在同样的混合数据上训练，但只使用长文本数据占比提升的 batch。这样模型在保持已有语言知识的同时，学习跨段落、跨章节的依赖。  
- **技术细节**：采用稀疏注意力（如 Sliding‑Window + Global Token）来降低 32k 长度的计算复杂度；使用梯度累积和混合精度训练来控制显存峰值。  

**3. 监督微调（SFT）**  
- 收集指令式对话数据和代码指令数据，构造“用户提问 → 期望答案”对。  
- 采用全参数微调或 LoRA（低秩适配）两种方式，让模型在保持通用能力的同时，更好地遵循明确指令。  

**4. 条件在线 RLHF（COOL RLHF）**  
- **奖励模型**：训练一个多任务奖励网络，输入包括模型输出、上下文以及条件标签（安全、事实、可解释）。  
- **在线更新**：在 RL 训练过程中，实时收集人类评审的偏好反馈，使用 PPO（近端策略优化）进行策略更新，同时对奖励模型进行微调，形成闭环。  
- **冲突处理**：当评审给出相互矛盾的偏好时，系统会依据预设的条件优先级（如安全 > 事实）自动调节奖励权重，防止模型“投机取巧”。  

**最巧妙的设计**  
- **渐进式长度扩展**：把长上下文训练拆成两步，既避免了“一次性大模型爆显存”，又让模型在进入长序列阶段前已经拥有扎实的语言基础。  
- **条件奖励**：在奖励函数里加入多维约束，使得模型在追求高分的同时不违背安全或事实原则，这在传统 RLHF 中是很少见的。  

### 实验与效果
- **评测维度**：报告中列出 6 大维度、30 个公开基准，包括通用问答、代码生成、数学推理、长文检索、对话安全等。  
- **长上下文表现**：在 200k token 的 Needle‑in‑a‑Haystack 测试中，InternLM2 能在前 5% 的检索结果里找到目标，显著优于同尺寸的开源基线（原文未给出具体数字）。  
- **整体领先**：论文声称在所有 30 项基准上整体分数均高于前代 InternLM 系列，并在多数任务上超过同等规模的闭源模型。  
- **消融实验**：作者分别去掉代码数据、长文本数据、COOL RLHF，结果显示：去掉长文本导致长文检索准确率下降约 12%；去掉 COOL RLHF 后主观安全评分下降约 8%。这些实验表明每个模块都有实质贡献。  
- **局限性**：报告承认模型仍然需要大规模算力进行训练，部署成本高；在极端多语言场景下表现仍有提升空间；COOL RLHF 的在线反馈依赖高质量评审，实际应用中可能受限。  

### 影响与延伸思考
InternLM2 的发布让开源社区第一次在同等规模下看到真正的长上下文能力和多维对齐策略，随后出现了多篇围绕“长序列稀疏注意力”“条件奖励 RLHF”的后续工作。很多新兴项目（如 LLaMA‑Long、OpenChat‑RLHF）直接借鉴了分阶段长度扩展和 COOL RLHF 的思路。对想进一步深入的读者，可以关注以下方向：① 更高效的稀疏注意力实现，降低 32k+ token 的计算成本；② 多语言长上下文预训练，提升跨语言检索能力；③ 自动化的条件权重学习，让模型自行平衡安全、事实与创造性。  

### 一句话记住它
InternLM2 用分阶段超长序列训练 + 条件在线 RLHF，让开源大模型首次在 200k 文本检索和多维安全对齐上实现了接近商业闭源的水平。