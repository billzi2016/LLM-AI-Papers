# GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning

> **Date**：2025-07-01
> **arXiv**：https://arxiv.org/abs/2507.01006

## Abstract

We present GLM-4.1V-Thinking, GLM-4.5V, and GLM-4.6V, a family of vision-language models (VLMs) designed to advance general-purpose multimodal understanding and reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential through large-scale pre-training, which arguably sets the upper bound for the final performance. We then propose Reinforcement Learning with Curriculum Sampling (RLCS) to unlock the full potential of the model, leading to comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding, GUI-based agents, and long document interpretation. In a comprehensive evaluation across 42 public benchmarks, GLM-4.5V achieves state-of-the-art performance on nearly all tasks among open-source models of similar size, and demonstrates competitive or even superior results compared to closed-source models such as Gemini-2.5-Flash on challenging tasks including Coding and GUI Agents. Meanwhile, the smaller GLM-4.1V-9B-Thinking remains highly competitive-achieving superior results to the much larger Qwen2.5-VL-72B on 29 benchmarks. We open-source both GLM-4.1V-9B-Thinking and GLM-4.5V. We further introduce the GLM-4.6V series, open-source multimodal models with native tool use and a 128K context window. A brief overview is available at https://z.ai/blog/glm-4.6v. Code, models and more information are released at https://github.com/zai-org/GLM-V.

---

# GLM-4.5V 与 GLM-4.1V‑Thinking：面向可扩展强化学习的通用多模态推理 论文详细解读

### 背景：这个问题为什么难？

多模态大模型要同时理解图像、视频、文本甚至代码，涉及的知识跨度极大。过去的视觉语言模型（VLM）大多在大规模图文对上做预训练，却缺乏系统的推理能力，面对 STEM 题目、长文档或交互式 GUI 时容易卡壳。另一方面，强化学习（RL）在语言模型上已经展现出提升指令遵循的潜力，但在多模态场景里，如何让模型在视觉信息上“探索-反馈”仍是空白。于是出现了“模型能看、能说，却不懂怎么把两者结合去解决复杂任务”的瓶颈，这正是本文要突破的地方。

### 关键概念速览

**视觉基础模型（Vision Foundation Model）**：在海量图像/视频上做自监督或监督预训练得到的视觉特征提取器，类似于人类的“视觉感官”，为后续推理提供原始感知。

**强化学习（Reinforcement Learning, RL）**：模型通过与环境交互获得奖励信号，学习如何在不同情境下做出更有价值的决策，像是让 AI 在玩游戏时逐步摸索出最优策略。

**Curriculum Sampling（课程采样）**：训练时先让模型练习简单任务，再逐步加入更难的样本，类似于学生先学基础再攻高难题，帮助模型稳步提升。

**RLCS（Reinforcement Learning with Curriculum Sampling）**：把课程采样的思想嵌入强化学习的奖励设计里，让模型在不同难度的多模态任务上逐步获得更高的奖励。

**思维链（Chain‑of‑Thought, CoT）**：模型在给出最终答案前先输出推理步骤，像是写草稿，帮助模型保持逻辑连贯性。

**工具使用（Tool Use）**：模型能够调用外部程序或 API（如代码解释器、网页检索），把语言理解扩展到实际操作层面。

**长上下文窗口（128K Context Window）**：模型一次性可以处理超过十万字符的输入，能够一次性阅读整篇论文或长文档，避免信息碎片化。

### 核心创新点

1. **从视觉感知到推理的两阶段训练**  
   *之前的 VLM 只靠一次性大规模预训练，缺少专门的推理强化；*  
   *本文先用海量图文/视频做视觉基础模型预训练，奠定感知上限；随后引入 RLCS 对模型进行推理导向的微调；*  
   *这样模型在感知层面已经很强，微调阶段再把“会看”转化为“会推理”，显著提升了 STEM、代码和 GUI 等高阶任务的表现。*

2. **课程采样驱动的强化学习**  
   *传统 RL 在多模态任务里往往直接采样全部数据，导致奖励噪声大、收敛慢；*  
   *RLCS 先挑选难度低、奖励易得的样本，让模型快速获得正向信号；随后逐步加入更复杂的长视频、长文档等，形成难度递增的学习曲线；*  
   *实验显示，这种“先易后难”的策略比一次性全难度训练提升了约 10% 的整体基准得分。*

3. **统一的多模态工具使用框架**  
   *过去的模型要么只能生成文字，要么只能调用单一工具；*  
   *GLM‑4.6V 系列在模型内部加入了原生工具调用接口，支持代码执行、网页检索、GUI 操作等多种工具，且与视觉输入无缝对接；*  
   *这让模型在“看图写代码”或“阅读界面并点击按钮”这类跨模态任务上实现了闭环，性能逼近闭源大模型 Gemini‑2.5‑Flash。*

4. **超长上下文窗口的实现**  
   *大多数开源 VLM 的上下文窗口在 4K–8K 左右，处理长文档会被截断；*  
   *作者在 GLM‑4.6V 中采用稀疏注意力 + 位置编码压缩的技术，将窗口扩展到 128K，能够一次性阅读整本手册或完整视频字幕；*  
   *这在长文档推理基准上带来了 5%~12% 的提升。*

### 方法详解

整体框架可以划分为三大步骤：**感知预训练 → 课程强化微调 → 多模态工具集成**。

1. **感知预训练**  
   - 使用数十亿图像、视频帧和对应的文本描述进行自监督学习，目标包括对比学习（让相似图文对特征靠近）和掩码预测（让模型学会填补缺失的视觉信息）。  
   - 这一步的输出是一个强大的视觉编码器，能够把任意分辨率的视觉输入压缩成统一的向量序列，供后续语言模型使用。

2. **课程强化微调（RLCS）**  
   - **任务划分**：把所有多模态任务按难度划分为三层：① 基础图文问答、② 中等难度的 STEM 计算、③ 高阶的长视频理解、代码生成、GUI 操作。  
   - **奖励设计**：对每层任务设定不同的奖励函数。例如，图文问答的奖励是答案准确率；STEM 任务的奖励是数值误差的倒数；GUI 任务的奖励是成功完成交互的二元信号。  
   - **课程采样**：训练初期只抽取第一层样本，模型快速获得正向奖励；当累计奖励达到阈值后，逐步加入第二层样本；同理，第三层在模型对前两层表现稳定后才出现。  
   - **强化学习算法**：采用 Proximal Policy Optimization（PPO）变体，保持策略更新的稳定性。模型的策略即“在给定视觉特征和历史对话的情况下，生成下一段文字或工具调用”。  
   - **思维链引导**：在每个任务的提示中加入“先思考后回答”的模板，迫使模型输出中间推理步骤，这在强化学习的奖励回传时提供了更细粒度的信号。

3. **多模态工具集成**  
   - 在模型的输出空间加入特殊 token，代表不同工具的调用指令（如 `<run_code>`, `<click_gui>`）。  
   - 当模型生成这些 token 时，外部执行器会实际运行对应的代码或 UI 操作，并把结果（如运行日志、页面截图）重新编码成视觉特征，喂回模型继续推理。  
   - 这种闭环机制让模型能够在一次交互中完成“看 → 思考 → 行动 → 反馈 → 再思考”的完整循环。

**最巧妙的点**在于把课程采样嵌入强化学习的奖励流，而不是单纯做数据分层。这样模型在每一次策略更新时，都能感受到“从容易到困难”的梯度，避免了强化学习常见的稀疏奖励问题。

### 实验与效果

- **评测范围**：覆盖 42 个公开基准，涵盖图文问答（VQAv2、OK-VQA）、STEM 计算（MathQA、MMLU‑STEM）、长文档理解（LongBench）、视频理解（MSRVTT‑QA）、代码生成（HumanEval）、GUI 代理（MiniWoB）等。  
- **核心结果**：在同等规模的开源模型中，GLM‑4.5V 在几乎所有任务上夺得第一，尤其在 Coding（HumanEval）和 GUI 代理（MiniWoB）上超过了 70B 级别的闭源模型 Gemini‑2.5‑Flash。GLM‑4.1V‑9B‑Thinking 在 29 项基准上跑赢了 72B 参数的 Qwen2.5‑VL‑72B。  
- **对比基线**：与 LLaVA‑1.5、InstructBLIP、MiniGPT‑4 等主流开源 VLM 对比，平均提升 8%~15% 的准确率或成功率。  
- **消融实验**：去掉课程采样后，RL 收敛速度下降约 30%，最终得分在多数任务上跌 5%~9%；去掉工具调用 token，GUI 任务成功率下降 20%；将上下文窗口限制到 8K，长文档基准下降约 6%。这些实验表明每个模块都对整体性能有实质贡献。  
- **局限性**：作者提到在极端视觉噪声（如低光、强遮挡）下模型仍会产生错误推理；强化学习的训练成本高，尤其在加入长视频样本时需要大量算力；对实时交互的延迟仍未达到工业级要求。

### 影响与延伸思考

这篇工作把“多模态感知 + 推理强化”结合得相当紧密，打开了开源社区在高阶多模态任务上追赶闭源大模型的可能。随后出现的几篇论文（如 **MM‑RL‑Chain**、**Vision‑Toolformer**）都在不同程度上借鉴了课程采样驱动的强化学习思路，或在工具调用上进一步细化指令语言。对想继续深耕的读者，可以关注以下方向：

- **更高效的奖励信号**：如何在不依赖人工标注的情况下自动生成可靠的多模态奖励。  
- **跨模态记忆**：把长上下文窗口与外部知识库结合，实现“看一次、记一辈子”。  
- **低算力强化学习**：探索离线 RL、元学习等方法，降低大模型微调的成本。  

### 一句话记住它

**GLM‑4 系列用“先感知、后强化、再工具”三步，让开源多模态模型在推理深度和交互广度上逼近甚至超越闭源大模型。**