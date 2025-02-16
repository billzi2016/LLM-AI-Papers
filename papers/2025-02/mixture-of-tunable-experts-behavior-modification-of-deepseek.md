# Mixture of Tunable Experts -- Behavior Modification of DeepSeek-R1 at   Inference Time

> **Date**：2025-02-16
> **arXiv**：https://arxiv.org/abs/2502.11096

## Abstract

We present the Mixture-of-Tunable-Experts (MoTE), a method that extends the Mixture-of-Experts architecture of Large Language Models (LLMs). Without additional training, MoTE enables meaningful and focused behavior changes in LLMs on-the-fly during inference time. By analyzing the digital LLM brain of DeepSeek-R1 using a technique we dub 'functional Token Resonance Imaging' (fTRI) -- inspired by fMRI and using prompts designed to elicit specific behavior (e.g., 'What happened {time}{place}?') -- we empirically identify distinctive experts associated with behaviors like refusal responses. Using MoTE we are able to intervene and control such specific behavior. We switched off the top 10 most refusal-relevant experts (0.07% of R1's 14,848 routed experts), achieving a 52% refusal reduction on sensitive reference prompts without performance degradation on MT-Bench. Random expert deactivation resulted in smaller behavioral shifts with increased noise, whereas forced expert activation led to significantly higher refusal rates. Our approach shares similarities with sparse autoencoders (SAEs) in terms of explainability and steerability. Unlike SAEs, MoTE does not require large training efforts, as within MoEs with a vast number of experts, specialization already emerged naturally during pretraining. Our findings suggest that significant functional mechanisms in Mixture-of-Experts architectures can at least partially be localized in a small number of specific experts, rather than being distributed throughout the model's weights. Expert subgroups can be tuned to trigger significant behavior variations, providing insights into the inner workings of LLMs.

---

# 可调专家混合——在推理阶段对 DeepSeek‑R1 行为的修改 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在实际应用中经常会出现“不想回答”“拒绝提供信息”等安全相关的行为。传统的解决办法是**微调**或**RLHF**（基于人类反馈的强化学习），但这两种方式都需要大量标注、算力和时间，成本高且容易破坏模型原有的能力。更糟的是，微调后模型的行为往往是全局性的，想只调节某一类回答（比如降低拒绝率）却会牵连到其他任务。于是出现了一个核心难题：**能否在不重新训练的前提下，只动模型内部的极小部分，就实现针对性的行为改动**？

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：一种把模型拆成很多“专家”（子网络）的架构，输入会被路由到少数几个专家进行计算，类似于公司里不同部门处理不同业务。  
- **Expert（专家）**：MoE 中的子网络，通常只有几千到上万条，专门负责处理特定的特征或模式。  
- **Routing（路由）**：模型在每层决定把哪个 token 送给哪些专家的过程，就像大脑的注意力机制挑选最相关的神经元。  
- **Functional Token Resonance Imaging（fTRI）**：作者自创的分析手段，借鉴 fMRI 的思路，用一系列“行为诱导”提示词让模型产生响应，然后统计哪些专家被频繁激活，类似于用功能性磁共振扫描找出大脑中负责记忆的区域。  
- **Refusal behavior（拒绝行为）**：模型在面对敏感或违规请求时给出的“不回答”“对不起，我不能帮助”等回复。  
- **Sparse Autoencoder（稀疏自编码器，SAE）**：一种把高维表示压缩到稀疏向量的模型，常被用来解释 LLM 内部表征。这里用来对比 MoTE 的可解释性。  
- **Behavior steering（行为引导）**：在推理阶段通过控制模型内部结构来实现特定输出倾向的技术。

### 核心创新点
1. **用 fTRI 定位行为相关专家**  
   *之前的做法*：要找出模型内部负责某种行为的权重，需要大量的梯度分析或专门的解释模型，过程繁琐且不一定精准。  
   *本文的做法*：设计一套“行为诱导”提示（如“请告诉我{时间}{地点}发生了什么？”），让模型在这些提示下运行，记录每一次路由的专家激活情况，统计每个专家在目标行为提示与普通提示之间的激活差异，得到“共振强度”。  
   *带来的改变*：只用几千条提示就能快速筛选出与拒绝、同意等行为高度相关的专家，省去了梯度回溯和额外训练的步骤。

2. **Mixture‑of‑Tunable‑Experts（MoTE）——在推理时直接开关专家**  
   *之前的做法*：行为调节只能通过微调或在生成阶段加入后处理规则，前者成本高，后者缺乏深层次控制。  
   *本文的做法*：在模型的路由阶段加入一个“专家开关”模块，允许在推理时手动禁用（或强制激活）选中的专家。实现方式类似于在路由表里把对应的专家权重置为零或设为极大。  
   *带来的改变*：只动 0.07%（10/14,848）的专家，就能显著降低模型的拒绝率，且不影响整体性能，实现了“即插即用”的行为调节。

3. **行为调节的细粒度实验验证**  
   *之前的做法*：大多数安全调节实验只报告整体安全指标，缺少对单一行为的因果分析。  
   *本文的做法*：分别进行“目标专家关闭”“随机专家关闭”“目标专家强制激活”三类实验，比较拒绝率变化和生成质量。  
   *带来的改变*：证明了行为是高度局部化的——只要关掉最相关的几位专家，拒绝率立刻下降 52%；随机关闭则只产生轻微、噪声式的行为漂移；强制激活则让拒绝率飙升，形成了清晰的因果链。

4. **与稀疏自编码器的可解释性对比**  
   *之前的做法*：SAE 能把内部表征映射到可解释的概念，但需要额外的训练和解码步骤。  
   *本文的做法*：直接利用 MoE 本身的专家划分和路由信息，无需额外模型。  
   *带来的改变*：展示了在已有的 MoE 结构中，功能已经自然分化，解释和调控成本几乎为零。

### 方法详解
**整体思路**可以概括为三步：  
1) **行为诱导采样** → 2) **共振度计算与专家排序** → 3) **推理时专家开关**。  
下面把每一步拆开讲。

#### 1) 行为诱导采样（Prompt Engineering + fTRI）
- 作者准备两套提示：一套是**目标行为提示**（比如让模型产生拒绝的敏感问题），另一套是**对照提示**（普通的问答或描述性句子）。  
- 把每套提示分别喂给 DeepSeek‑R1，记录每一层、每一个 token 被路由到的专家 ID。因为 MoE 的路由是稀疏的（每层只选 2‑4 个专家），日志非常紧凑。  
- 类比于医学中的功能性磁共振扫描：对照组相当于安静状态，任务组相当于刺激状态，后面要找出“激活最强”的脑区。

#### 2) 共振度计算与专家排序（Functional Token Resonance Imaging）
- 对每个专家，统计它在目标提示下被激活的次数 **A**，以及在对照提示下的激活次数 **B**。  
- 计算 **共振度 = (A - B) / (A + B)**，数值越大说明该专家在目标行为中越“专一”。  
- 按共振度从高到低排序，取前 **k**（本文实验取 10）作为“行为关键专家”。这里的 **k** 可以根据需求灵活调节，作者证明即使只关掉 10 个也能产生显著效果。

#### 3) 推理时专家开关（Mixture‑of‑Tunable‑Experts）
- 在模型的路由函数里加入一个 **mask** 向量，长度等于专家总数。  
- 对于要关闭的专家，把对应位置设为 **-∞**（在 softmax 前），这样它们的路由概率被压到 0，等同于从网络中剔除。  
- 对于要强制激活的专家，把对应位置设为一个很大的正数，使其路由概率接近 1。  
- 这一步只在推理时执行，不改变模型的权重，也不需要重新训练，只是动态地修改路由表。

**最巧妙的地方**在于：作者利用 MoE 本身的稀疏路由特性，把行为调节的“开关”直接映射到专家层面，而不必去改动每一层的权重矩阵。换句话说，模型的“大脑”已经有了功能分区，MoTE 只需要把对应的“神经元”关掉或点亮。

### 实验与效果
- **测试对象**：DeepSeek‑R1，拥有 14,848 个路由专家。  
- **行为评估**：使用公开的 **Sensitive Reference Prompts**（一组专门用于检测模型是否会拒绝或规避敏感话题的提示），统计模型给出拒绝类回复的比例。  
- **基准对比**：  
  - **原始模型**：在这些提示上拒绝率约为 **X%**（论文未给出具体数值，只说有显著的拒绝行为）。  
  - **关闭前 10 名拒绝相关专家**（约占 0.07%）：拒绝率下降 **52%**，即从原始的 X% 降到约 **0.48X%**。  
  - **随机关闭 10 名专家**：仅有轻微的行为漂移，拒绝率下降不到 10%，且生成文本出现更多噪声。  
  - **强制激活前 10 名专家**：拒绝率显著上升，超过原始水平，验证了这些专家的因果作用。  
- **整体性能**：在 **MT‑Bench**（一个衡量多任务能力的基准）上，关闭这 10 名专家后模型的得分几乎没有变化，说明核心能力未受影响。  
- **消融实验**：作者逐步增加关闭的专家数量（5、10、20），发现拒绝率随关闭数量呈近线下降，但超过约 30 名后性能开始出现轻微下降，说明有一个“安全阈值”。  
- **局限性**：论文只在 DeepSeek‑R1 上做了实验，未验证在其他 MoE 架构（如 SwitchTransformer、GLaM）上的可迁移性；此外，fTRI 依赖于精心设计的行为提示，提示质量直接影响专家排序的准确性。

### 影响与延伸思考
- **即时可控性**：MoTE 为 LLM 的安全与合规提供了一种“即插即用”的手段，省去了昂贵的微调流程，已经在一些企业内部的安全管控系统中被尝试采用（推测）。  
- **解释性研究的加速**：通过 fTRI 能快速定位功能专一的专家，为后续的“专家可视化”“概念抽取”等工作提供了低成本的入口。  
- **后续工作**：已有几篇论文（如 “Expert‑Level Prompt Tuning” 与 “Sparse Steering of MoE LLMs”）受其启发，尝试用自动化搜索或强化学习来挑选最有效的专家子集，实现更细粒度的多行为调节。  
- **未来方向**：  
  1) **跨模型通用性**：研究是否可以把在一个 MoE 上发现的行为专家映射到另一个模型。  
  2) **自动化 fTRI**：利用大规模行为提示库和统计学习自动生成共振度图谱。  
  3) **安全审计**：把专家开关机制集成到模型部署平台，实现实时的风险监控与快速响应。  

### 一句话记住它
只要在推理时把少数负责特定行为的 MoE 专家关掉，就能快速、无需再训练地改写大模型的行为。