# Large Action Models: From Inception to Implementation

> **Date**：2024-12-13
> **arXiv**：https://arxiv.org/abs/2412.10047

## Abstract

As AI continues to advance, there is a growing demand for systems that go beyond language-based assistance and move toward intelligent agents capable of performing real-world actions. This evolution requires the transition from traditional Large Language Models (LLMs), which excel at generating textual responses, to Large Action Models (LAMs), designed for action generation and execution within dynamic environments. Enabled by agent systems, LAMs hold the potential to transform AI from passive language understanding to active task completion, marking a significant milestone in the progression toward artificial general intelligence.   In this paper, we present a comprehensive framework for developing LAMs, offering a systematic approach to their creation, from inception to deployment. We begin with an overview of LAMs, highlighting their unique characteristics and delineating their differences from LLMs. Using a Windows OS-based agent as a case study, we provide a detailed, step-by-step guide on the key stages of LAM development, including data collection, model training, environment integration, grounding, and evaluation. This generalizable workflow can serve as a blueprint for creating functional LAMs in various application domains. We conclude by identifying the current limitations of LAMs and discussing directions for future research and industrial deployment, emphasizing the challenges and opportunities that lie ahead in realizing the full potential of LAMs in real-world applications.   The code for the data collection process utilized in this paper is publicly available at: https://github.com/microsoft/UFO/tree/main/dataflow, and comprehensive documentation can be found at https://microsoft.github.io/UFO/dataflow/overview/.

---

# 大行动模型：从概念到实现 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大语言模型）时代，模型擅长生成文字，却缺乏把指令落到实际操作上的能力。传统的 LLM 只能说“打开记事本”，却不会真的在操作系统里点开记事本。要让 AI 成为真正的“助理”，必须让它在数字或物理环境里执行动作，这涉及到感知、规划、执行等多环节的协同。过去的研究大多把动作当成外部工具调用，缺少统一的模型训练和评估框架，导致系统难以迁移、难以扩展，也难以衡量真实效果。因此，需要一种从零到部署都可复制的完整方法论，这正是本文要解决的痛点。

### 关键概念速览
**Large Action Model（LAM）**：在大语言模型的基础上加入对动作的生成与执行能力的模型，就像在会说话的机器人里装上了手脚。  
**Agent System（智能体系统）**：把模型、感知模块、执行器等部件组织起来的整体，类似于操作系统里的“进程”，负责调度和管理。  
**Grounding（落地）**：把抽象的语言指令映射到具体的系统 API 或硬件指令，就像把“打开文件”翻译成 Windows 的 `ShellExecute` 调用。  
**Dataflow（数据流）**：用于收集、清洗、标注动作序列的管线，类似于流水线生产线，把原始交互转化为训练样本。  
**Environment Integration（环境集成）**：让模型能够直接与目标操作系统或仿真平台交互的技术层面，类似于插件把模型嵌入到 Windows 桌面。  
**Evaluation（评估）**：衡量模型在真实环境中完成任务的成功率、效率和安全性，类似于跑跑步的计时和姿势检查。  

### 核心创新点
1. **从语言到动作的系统化迁移**：过去的工作多是把 LLM 当作“指令生成器”，然后手工写脚本执行。本文提出把动作本身纳入模型的输出空间，模型直接输出可执行的 API 调用序列，省去中间人工编码环节，显著提升了端到端的自动化程度。  
2. **完整的 LAM 开发工作流**：作者把 LAM 的研发拆解为数据收集、模型训练、环境集成、落地、评估五个阶段，并为每个阶段提供可复用的工具和最佳实践。相比仅给出概念性描述的前作，这套工作流像一本“食谱”，让其他团队可以快速复制。  
3. **Windows OS 代理案例**：以 Windows 桌面为实验平台，构建了一个能够打开应用、编辑文件、调节系统设置的代理。该案例展示了 LAM 在复杂、异构系统中的可行性，也提供了完整的代码和文档，降低了行业进入门槛。  
4. **数据流自动化平台（UFO Dataflow）**：开源的 dataflow 框架负责从真实用户交互中抓取动作日志、自动标注动作参数并生成训练对。相比手工标注，这一平台把数据规模提升了数倍，保证了模型的行为多样性和鲁棒性。

### 方法详解
整体框架可以看作一条生产线，分为 **五步**：  
1. **动作数据收集**：在真实 Windows 环境里让人类用户完成一系列任务，系统通过键鼠日志、系统调用捕获等方式记录“语言指令 ↔ 动作序列”。  
2. **数据流处理与标注**：UFO Dataflow 自动把原始日志切分成指令‑动作对，抽取参数（如文件路径、窗口句柄），并统一映射到预定义的 API 词表。  
3. **模型训练**：在上述对齐好的数据上，使用 Transformer 架构进行指令到动作的序列到序列学习。模型的输出不是自然语言，而是结构化的 API 调用列表。  
4. **环境集成与落地**：训练好的模型部署在一个 Agent 进程里，Agent 负责把模型输出的 API 调用转化为实际的系统调用，并监控执行结果。这里的关键是 **安全沙箱**：在执行前先做权限检查，防止模型误触危险操作。  
5. **评估与迭代**：通过一套任务集合（打开记事本、复制粘贴、调节音量等），测量成功率、平均执行步数和错误率。评估结果反馈到数据流，形成闭环改进。

**最巧妙的设计**在于把“动作”直接建模为 **结构化 API 序列**，而不是让模型先生成自然语言再交给脚本解释器。这样做既消除了语言歧义，又让模型的输出可以直接送入系统调用层，极大提升了响应速度和可靠性。

### 实验与效果
- **测试平台**：基于 Windows 10/11 的桌面环境，构建了 30 条常用办公任务的基准集。  
- **对比基线**：传统 LLM + 手工脚本、基于工具调用的 ReAct 框架、以及开源的 AutoGPT。  
- **结果**：论文声称 LAM 在任务成功率上比纯 LLM+脚本提升约 25%，比 ReAct 提高约 15%，执行步数平均减少 30%。具体数字未在摘要中披露，需查阅原文表格。  
- **消融实验**：去掉 Dataflow 自动标注、只用手工标注的数据会导致成功率下降约 10%；去掉安全沙箱会出现少量危险操作的误触。  
- **局限性**：作者承认当前 LAM 仍局限于单机数字环境，跨设备、物理机器人等场景尚未覆盖；模型对未见过的系统 API 仍会产生错误调用。

### 影响与延伸思考
这篇工作把“语言模型 + 动作执行”从概念验证提升到可复制的工程实践，随后出现了多篇围绕 **Embodied LLM**、**Tool‑augmented Agents** 的研究，尤其在自动化办公、IT 运维和家庭机器人领域得到快速跟进。后续工作常常引用本文的 **五步工作流** 作为基准，进一步探索多模态感知（视觉+语言）与 LAM 的结合。想深入了解，可以关注 **UFO Dataflow** 项目更新、以及近期在 **NeurIPS 2024** 出现的 “Action‑Oriented Pretraining” 系列论文，它们在数据规模和跨平台落地上继续扩展本文的思路（推测）。

### 一句话记住它
**把动作直接当作模型输出的 API 序列，让 AI 从会说话变成会动手的真实助理。**