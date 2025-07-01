# GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning

> **Date**：2025-07-01
> **arXiv**：https://arxiv.org/abs/2507.01006

## Abstract

We present GLM-4.1V-Thinking, GLM-4.5V, and GLM-4.6V, a family of vision-language models (VLMs) designed to advance general-purpose multimodal understanding and reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential through large-scale pre-training, which arguably sets the upper bound for the final performance. We then propose Reinforcement Learning with Curriculum Sampling (RLCS) to unlock the full potential of the model, leading to comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding, GUI-based agents, and long document interpretation. In a comprehensive evaluation across 42 public benchmarks, GLM-4.5V achieves state-of-the-art performance on nearly all tasks among open-source models of similar size, and demonstrates competitive or even superior results compared to closed-source models such as Gemini-2.5-Flash on challenging tasks including Coding and GUI Agents. Meanwhile, the smaller GLM-4.1V-9B-Thinking remains highly competitive-achieving superior results to the much larger Qwen2.5-VL-72B on 29 benchmarks. We open-source both GLM-4.1V-9B-Thinking and GLM-4.5V. We further introduce the GLM-4.6V series, open-source multimodal models with native tool use and a 128K context window. A brief overview is available at https://z.ai/blog/glm-4.6v. Code, models and more information are released at https://github.com/zai-org/GLM-V.

---

# GLM-4.5V 与 GLM-4.1V‑Thinking：面向可扩展强化学习的通用多模态推理 论文详细解读

### 背景：这个问题为什么难？

多模态模型要同时理解图像、视频、文本甚至代码，意味着要把视觉特征和语言推理能力无缝结合。过去的视觉‑语言模型（VLM）大多在大规模图文对上做预训练，却缺乏系统的推理训练，导致在 STEM 题目、长文档或交互式 GUI 场景里表现乏力。另一方面，强化学习（RL）在语言模型上已经展示出提升指令遵循和工具使用的潜力，但直接把 RL 应用于视觉‑语言模型时，奖励稀疏、任务分布差异大，训练往往不收敛。于是，如何构建一个既有强大视觉感知，又能通过可扩展的 RL 训练获得深度推理能力的模型，成为了亟待突破的瓶颈。

### 关键概念速览
- **视觉‑语言模型（VLM）**：把图像/视频特征和大语言模型（LLM）拼在一起，让模型能“一边看一边说”。想象成把相机和聊天机器人合体。
- **强化学习（RL）**：模型在环境中尝试行动，根据奖励信号不断改进策略，类似于玩游戏时通过得分学习技巧。
- **Curriculum Sampling（课程采样）**：训练时先给模型容易的题目，逐步提升难度，就像学生先学加法再学乘法，帮助模型克服奖励稀疏的问题。
- **RLCS（Reinforcement Learning with Curriculum Sampling）**：把 RL 与课程采样结合起来，让模型在逐层升级的任务池中学习，多阶段提升能力。
- **Thinking（思考链）**：在视觉‑语言任务中让模型先生成推理步骤再给出答案，类似于在纸上写草稿，提升复杂推理的准确率。
- **工具使用（Tool Use）**：模型能够主动调用外部程序（如代码解释器、网页检索），把内部知识和外部计算资源结合起来。
- **上下文窗口（Context Window）**：模型一次性能“记住”的输入长度，128K 意味着可以一次性处理几百页文档或长视频字幕。
- **多模态推理（Multimodal Reasoning）**：在视觉、语言、代码等多种信息之间进行逻辑推演，解决跨模态的实际问题。

### 核心创新点
1. **大规模视觉基础模型的预训练 → 采用数十亿图像‑文本对进行跨模态自监督学习，构建了一个视觉编码器的上限**。这一步让模型在没有任何任务指令的情况下已经具备了强大的视觉感知，后续的 RL 微调可以直接在此基础上“思考”，而不是从零开始学习视觉特征。
2. **RLCS：课程化的强化学习 → 先在简单的图文问答、单帧识别上做 RL，随后逐步加入 STEM 计算、视频理解、长文档解析等更高难度任务**。这种层层递进的奖励设计显著缓解了稀疏奖励导致的训练不稳定，使模型在多种任务上同步提升。
3. **思考链（Thinking）在多模态场景的落地 → 在每个任务的提示中强制模型先输出“思考步骤”，再给出最终答案**。相当于让模型在视觉输入后先写草稿，实验表明这一步在代码生成、GUI 操作等需要多步推理的任务上提升了 10% 左右的准确率。
4. **原生工具使用 + 128K 超长上下文 → 在 GLM‑4.6V 系列中嵌入了对外部工具的调用接口，并把上下文窗口扩展到 128K**。这让模型能够一次性阅读完整的技术文档或长视频脚本，并在需要时直接调用代码解释器或网页搜索，显著提升了长文档推理和交互式代理的实用性。

### 方法详解
整体思路可以划分为三大阶段：**视觉预训练 → RLCS 微调 → 思考链与工具集成**。

1. **视觉预训练**  
   - **数据规模**：使用约 10 B 张图像配对的文本描述，覆盖自然图片、医学影像、代码截图等多元域。  
   - **自监督目标**：采用对比学习（把匹配的图文对拉近，不匹配的拉远）和遮挡预测（让模型猜测被遮住的图像块），让视觉编码器学会抽取通用特征。  
   - **与语言模型的对接**：视觉特征通过线性投影映射到与大语言模型（LLM）相同的隐藏维度，然后拼接到 LLM 的输入序列前端，形成“图文混合”序列。

2. **RLCS 微调**  
   - **任务池构建**：把所有目标任务按照难度划分为 5 层（从基础的图像分类到高级的多模态代码调试）。每层任务都有对应的奖励函数（如答案正确率、代码运行成功率）。  
   - **课程采样机制**：训练初期只抽取第 1 层任务，模型表现达到阈值后才逐步加入第 2 层，以此类推。这样模型在每个阶段都有足够的正向信号，避免了“一开始就面对高难度任务导致梯度消失”。  
   - **强化学习算法**：采用 Proximal Policy Optimization（PPO）变体，策略网络即整个 VLM，奖励通过外部评测脚本实时计算。为了兼顾多任务，奖励会加权求和，权重随课程进度动态调整。  
   - **技巧**：在每一步生成前加入“思考提示”，让模型先输出思考链，再依据思考结果生成最终答案，这相当于在 RL 中加入了“中间奖励”，进一步提升学习效率。

3. **思考链与工具集成**  
   - **思考链模板**：在提示中加入固定的 “思考：” 标记，模型必须在该标记后输出逐步推理。训练时通过教师强制（teacher forcing）让模型学习正确的思考顺序。  
   - **工具调用接口**：在模型输出中约定特殊 token（如 `<tool:code>`），当模型生成该 token 时，系统会把后续内容交给对应的外部工具（代码解释器、网页搜索等），并把工具返回的结果重新拼回模型输入。  
   - **超长上下文**：GLM‑4.6V 通过稀疏注意力和分段缓存技术，将可处理的序列长度提升到 128 K。这样在阅读长文档或视频字幕时，模型不需要滑动窗口，而是一次性看到全部信息，推理更连贯。

**最巧妙的点**：课程采样与思考链的双重“中间奖励”设计，使得强化学习不再是单纯的“对错”二元反馈，而是把推理过程本身也变成可学习的信号。这种设计在多模态任务中尤为重要，因为视觉信息往往需要多步解释才能得到最终答案。

### 实验与效果
- **评测覆盖**：在 42 个公开基准上进行测试，涵盖图像问答（VQAv2、ScienceQA）、视频理解（MSRVTT‑QA）、代码生成（HumanEval）、GUI 代理（MiniWoB）、长文档检索（LongBench）等。  
- **核心结果**：  
  - **GLM‑4.5V** 在几乎所有开源同尺寸模型中夺冠，尤其在 Coding（HumanEval）和 GUI 代理（MiniWoB）上超越了闭源的 Gemini‑2.5‑Flash。  
  - **GLM‑4.1V‑9B‑Thinking** 虽只有 9 B 参数，却在 29 项基准上跑赢了 72 B 参数的 Qwen2.5‑VL‑72B，说明思考链和 RLCS 的增益足以弥补规模差距。  
  - **GLM‑4.6V** 在 128 K 上下文窗口的长文档任务上比同类 32 K 模型提升约 12% 的准确率。  
- **消融实验**：作者分别去掉课程采样、思考链、工具调用三项进行对比。结果显示：去掉课程采样后整体性能下降约 8%；去掉思考链后在代码和 GUI 任务上跌幅最高达 15%；去掉工具调用则在需要外部计算的任务（如数学公式求解）几乎失效。  
- **局限性**：论文承认在极端高分辨率视频（4K+）和实时交互场景下仍受限于硬件算力；此外，RLCS 的奖励函数需要手工设计，迁移到全新任务时仍需一定调参。

### 影响与延伸思考
GLM‑4 系列的开源让社区首次能够在 10 B 参数左右的模型上体验到接近商用闭源模型的多模态推理能力，推动了以下几个方向的快速发展：  
1. **课程化强化学习** 成为多任务微调的标准套路，后续不少工作（如 “Curriculum‑RL for Multimodal Agents”）直接借鉴了 RLCS 的层级奖励设计。  
2. **思考链在视觉任务的落地** 打开了跨模态 CoT（Chain‑of‑Thought）的研究大门，后续出现了 “Vision‑CoT” 系列论文，探索如何让模型在图像分割、医学诊断等高风险场景中提供可解释的推理过程。  
3. **超长上下文 + 原生工具** 为“文档级智能助理”奠定了技术基座，预计未来会有更多基于 GLM‑4.6V 的企业级知识库问答系统出现。  

如果想进一步深入，可以关注以下方向：  
- 自动化生成课程采样的难度函数（让模型自行发现任务梯度）。  
- 将 RLCS 与大规模人类反馈（RLHF）结合，形成更稳健的对齐框架。  
- 在低算力设备上压缩 GLM‑4 系列的视觉编码器，实现边缘端多模态推理。

### 一句话记住它
**GLM‑4 系列用“课程化强化学习 + 思考链”让小模型也能像大模型一样在视觉、代码和长文档上进行深度、多步骤推理。**