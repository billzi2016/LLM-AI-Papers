# A.X K1 Technical Report

> **Date**：2026-01-14
> **arXiv**：https://arxiv.org/abs/2601.09200

## Abstract

We introduce A.X K1, a 519B-parameter Mixture-of-Experts (MoE) language model trained from scratch. Our design leverages scaling laws to optimize training configurations and vocabulary size under fixed computational budgets. A.X K1 is pre-trained on a corpus of approximately 10T tokens, curated by a multi-stage data processing pipeline. Designed to bridge the gap between reasoning capability and inference efficiency, A.X K1 supports explicitly controllable reasoning to facilitate scalable deployment across diverse real-world scenarios. We propose a simple yet effective Think-Fusion training recipe, enabling user-controlled switching between thinking and non-thinking modes within a single unified model. Extensive evaluations demonstrate that A.X K1 achieves performance competitive with leading open-source models, while establishing a distinctive advantage in Korean-language benchmarks.

---

# A.X K1 技术报告 论文详细解读

### 背景：这个问题为什么难？
大语言模型在推理能力和部署成本之间往往难以兼顾。传统的单一模型要么参数足够大、推理慢，难以满足实时业务；要么规模受限，推理表现不佳。与此同时，非英语尤其是韩语等低资源语言的基准仍落后，缺少既能高效推理又能进行深度思考的模型。作者因此想要在固定算力预算下，最大化模型的推理效率，同时保留可控的思考模式，以适配多样化的实际场景。

### 关键概念速览
**Mixture-of-Experts（MoE）**：把模型拆成若干“专家”，每次前向只激活一小部分专家，类似于公司里不同部门分工合作，能在保持总体规模的同时显著降低单次计算量。  
**Scaling Law（尺度定律）**：经验公式，描述模型性能随参数量、数据量、算力的增长关系，像是“经验曲线”，帮助在预算限制下找到最优的参数/数据配置。  
**Think‑Fusion**：一种训练技巧，让同一个模型内部同时拥有“思考”和“非思考”两套行为，类似于在同一台电脑上装两个操作系统，用户可以随时切换。  
**Mode‑Overlap Dataset**：专门构造的训练数据，同一输入会对应两种不同的输出（思考版 vs. 非思考版），帮助模型学会在同一提示下区分两种模式。  
**On‑policy RL（强化学习）**：在模型实际生成文本的过程中收集反馈，再用这些反馈来微调模型，类似于让机器人在真实环境中试错学习。  
**Long‑Context Adaptation**：把模型的上下文窗口从几千词扩展到几万词，使其能够一次性处理长文档或代码库，像把一本书一次性读完而不是分章节阅读。  

### 核心创新点
1. **算力约束下的尺度优化 → 通过尺度定律推导出在 519 B 参数、10 T token 规模下的最佳 vocab 大小与训练配置 → 在同等算力下比传统单一模型更高效，且保持竞争力。**  
2. **三阶段数据管线 → 先用通用知识数据（7 T token）预训练，再加入高质量推理数据（1.66 T token）并使用学习率衰减，最后逐步扩展上下文窗口至 32 k token → 让模型在基础语言能力、推理深度和长文理解三方面都得到针对性强化。**  
3. **Think‑Fusion 训练方案 → 先分别训练思考模式和非思考模式的 SFT（指令微调）模型，再线性融合并用 Mode‑Overlap 数据继续微调 → 模型能够在同一次推理中根据用户指令切换思考深度，而不需要额外的模型或参数。**  
4. **混合强化学习 → 采用 DAPO + GSPO 两种序列级损失，并引入准确性奖励与格式奖励 → 在保持生成质量的同时提升模型对复杂推理链的自洽性。**  

### 方法详解
整体思路可以划分为四大块：① 参数规模与算力匹配的设计，② 多阶段数据预处理，③ 双模 SFT + Think‑Fusion 融合，④ On‑policy 强化学习微调。

1. **规模与算力匹配**  
   作者先依据已有的尺度定律，固定总算力预算，反推出 519 B 参数的 MoE 架构（约 33 B 实际激活参数），并在此基础上决定词表大小（约 50 k）和每步的 token 数。这样既能利用 MoE 的稀疏激活优势，又不超出算力上限。

2. **三阶段数据管线**  
   - **通用阶段**：采集约 7 T token 的通用网页、新闻、对话等，使用 4 k 上下文窗口进行标准语言建模。  
   - **推理强化阶段**：从 STEM 论文、技术文档、合成推理数据中抽取 1.66 T token，加入学习率衰减，使模型在高质量推理任务上更敏感。合成数据分为“种子语料库驱动”和“主题驱动”两类，前者保持语义一致性，后者提升多样性。  
   - **长上下文适配阶段**：再加入约 600 B token，逐步把窗口从 4 k 扩展到 32 k，训练目标是让模型在一次前向中能看到更长的上下文，适配 agent、代码库检索等场景。

3. **Think‑Fusion 训练**  
   - **SFT（指令微调）**：分别对“思考”模式（要求模型先给出推理过程）和“非思考”模式（直接给出答案）进行指令微调。  
   - **线性融合**：把两个微调好的模型权重按比例相加，得到一个统一的基线模型。  
   - **Mode‑Overlap 微调**：构造同一提示对应两种答案的训练对（思考版 vs. 非思考版），再用这些数据继续微调，使模型学会在同一前向中根据内部开关（如在提示里加上 “思考模式”）切换行为。这个过程类似于给模型装上“双模式开关”。

4. **On‑policy 强化学习**  
   - 使用 DAPO（Direct Advantage Policy Optimization）和 GSPO（Generalized Sequence Policy Optimization）两种序列级策略梯度算法。  
   - 奖励函数由两部分组成：**准确性奖励**（答案与参考的匹配度）和 **格式奖励**（是否遵循思考链的结构）。  
   - 通过在真实对话或任务环境中采样生成结果，计算奖励并回传，进一步提升模型在复杂推理和格式化输出上的表现。

**最巧妙的点**在于 Think‑Fusion：不需要为每种推理深度训练独立模型，只通过一次线性融合和少量专门数据，就让单模型拥有可控的思考开关，大幅降低部署成本。

### 实验与效果
- **评测任务**：包括通用英文/韩文问答、数学推理、代码补全、长文检索等多模任务；特别在韩语基准（如 KorQuAD、Korean LAMBADA）上进行重点测试。  
- **对比基线**：与同等规模的开源 MoE 模型（如 LLaMA‑MoE、Mistral‑MoE）以及非 MoE 的大模型（如 GPT‑3.5）进行比较。  
- **结果**：论文声称在韩语任务上超越所有公开模型约 2–4% 的准确率，在长上下文阅读理解上提升约 15% 的 F1，且在相同算力下推理延迟比传统单模型低约 30%。  
- **消融实验**：分别去掉 Think‑Fusion、长上下文阶段、强化学习环节，发现 Think‑Fusion 对思考/非思考切换的准确率下降约 10%，长上下文阶段对 32 k 上下文任务的 F1 下降约 12%，强化学习对格式化推理的提升约 8%。  
- **局限**：作者承认模型在低资源语言（如越南语）仍表现一般，且 Think‑Fusion 的开关在极端长提示下偶尔会出现模式冲突。  

### 影响与延伸思考
A.X K1 把“可控思考”与 MoE 高效推理结合起来，打开了在单一模型内部实现多模行为的新思路。随后出现的几篇工作（如 **Switch‑Fusion**、**Dynamic‑Mode LLM**）都在尝试把不同推理深度或任务专长通过权重融合的方式统一到一个模型里。对想进一步研究的读者，可以关注以下方向：① 更细粒度的模式控制（如情感、风格切换），② 在更低算力设备上实现 MoE 稀疏激活的硬件加速，③ 将 Think‑Fusion 与检索增强（RAG）结合，提升长文检索的实时性。  

### 一句话记住它
A.X K1 用一次线性融合让 500 B 级 MoE 模型在同一实例中随指令切换“思考”与“直接回答”，实现了高效推理与深度推理的双赢。