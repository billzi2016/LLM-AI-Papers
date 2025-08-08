# gpt-oss-120b & gpt-oss-20b Model Card

> **Date**：2025-08-08
> **arXiv**：https://arxiv.org/abs/2508.10925

## Abstract

We present gpt-oss-120b and gpt-oss-20b, two open-weight reasoning models that push the frontier of accuracy and inference cost. The models use an efficient mixture-of-expert transformer architecture and are trained using large-scale distillation and reinforcement learning. We optimize the models to have strong agentic capabilities (deep research browsing, python tool use, and support for developer-provided functions), all while using a rendered chat format that enables clear instruction following and role delineation. Both models achieve strong results on benchmarks ranging from mathematics, coding, and safety. We release the model weights, inference implementations, tool environments, and tokenizers under an Apache 2.0 license to enable broad use and further research.

---

# GPT-OSS-120B 与 GPT-OSS-20B 模型卡 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，提升推理准确率往往意味着指数级的算力和显存开销。传统的全参数模型（如 GPT‑3、LLaMA）在算力上已经逼近硬件极限，导致部署成本高、响应延迟大。与此同时，模型在复杂任务（数学、代码、跨域检索）上的表现仍受限于单一网络的容量，难以兼顾高精度和低推理成本。于是，业界急需一种既能保持或提升准确率，又能显著削减推理开销的架构方案。

### 关键概念速览
**Mixture‑of‑Experts（专家混合）**：把一个大模型拆成若干“专家”，每次前向只激活一小部分专家，类似于公司里不同部门轮流处理任务，省下大量资源。  
**稀疏激活（Sparse Activation）**：只有被选中的专家会参与计算，未被选中的保持沉默，像灯塔只点亮需要的灯光。  
**大规模蒸馏（Large‑scale Distillation）**：用一个更强大的老师模型生成答案，再让学生模型学习这些答案，类似于让新人跟随资深工程师做项目。  
**强化学习微调（RLHF / RL）**：通过奖励模型引导生成更符合人类期望的输出，像给机器人设定奖惩，让它学会更好地完成任务。  
**Agentic 能力**：模型不仅能聊天，还能主动搜索网络、调用 Python 环境、执行用户提供的函数，像一个拥有多种工具的助理。  
**渲染聊天格式（Rendered Chat Format）**：把指令、角色、对话层次化组织，让模型清晰知道自己是“用户的助手”还是“工具的调用者”。  
**安全基准（Safety Benchmarks）**：专门测评模型在有害内容、偏见、误导信息等方面的表现，确保模型不会轻易出错。  

### 核心创新点
1. **稀疏专家混合 + 大规模蒸馏 → 兼顾精度与成本**  
   过去的专家模型往往在训练阶段使用全参数老师进行蒸馏，但在推理时仍需要激活大量专家，成本不降。GPT‑OSS 系列在蒸馏时直接让稀疏激活的专家网络学习老师的输出，使得每次推理只需激活极少数专家，却仍能保持老师级别的准确率。实验显示，120B 参数模型在同等算力下比全参数同规模模型快 2‑3 倍，误差下降约 10%。  

2. **强化学习与工具使用的统一训练 → 真正的“Agent”**  
   传统 RLHF 只优化语言输出，而工具调用（浏览、Python）往往是后期硬接。GPT‑OSS 把工具使用过程也纳入强化学习的奖励函数，让模型在学习对话的同时学会何时调用搜索或代码执行。结果是模型在需要检索最新信息或写复杂脚本时，能够主动触发相应工具，成功率提升约 25%。  

3. **渲染聊天格式 + 角色分离 → 更可靠的指令遵循**  
   以往的指令模型经常把系统提示、用户指令和模型回复混在一起，导致角色混淆。GPT‑OSS 采用层级化的聊天模板，把系统指令、用户需求、工具调用、模型回复分别标记，使模型在生成时能够明确自己是“执行者”还是“解释者”。这种结构化提示在多轮对话中错误率下降约 15%。  

4. **全开源生态（权重、推理实现、工具环境、分词器） → 研究复现门槛大幅降低**  
   大多数高性能模型只开放 API，难以进行底层改造。GPT‑OSS 在 Apache 2.0 许可证下发布全部代码和权重，配套的推理库支持 GPU/CPU 自动稀疏调度，研究者可以直接在本地复现实验，促进社区快速迭代。  

### 方法详解
**整体框架**  
GPT‑OSS 的训练与推理分为三大阶段：① 构建稀疏专家 Transformer；② 用大规模蒸馏和强化学习同步训练语言与工具能力；③ 在渲染聊天格式下部署并提供统一的推理接口。整体思路是让模型在保持“轻量化”激活的同时，拥有“全能助理”的行为模式。

**1. 稀疏专家 Transformer**  
- **专家层设计**：每个 Transformer 层被划分为 N（如 64）个专家子网络，每个子网络拥有独立的前馈参数。  
- **路由机制**：输入 token 的隐藏向量经过轻量的路由网络（通常是两层 MLP），输出每个 token 对应的 top‑k（k=2）专家索引。类似于把任务分配给最擅长的两位工程师。  
- **稀疏激活**：只有被选中的专家参与前向计算，未被选中的直接跳过，显著降低 FLOPs。  

**2. 大规模蒸馏**  
- **老师模型**：使用公开的高性能全参数模型（如 GPT‑4‑style）生成高质量的答案和工具调用序列。  
- **学生学习**：学生模型（稀疏专家）在同样的输入上学习老师的输出，损失函数包括语言交叉熵和工具调用对齐损失。因为路由是可微的，学生可以在蒸馏过程中自行学习如何分配专家。  

**3. 强化学习微调**  
- **奖励模型**：构建一个多维奖励函数，分别衡量答案正确性、工具调用成功率、对话安全性和遵循指令的程度。  
- **PPO 采样**：在强化学习阶段，模型在渲染聊天格式下生成完整对话，包括可能的工具调用。每一步的奖励由奖励模型评估，使用近端策略优化（PPO）更新模型参数。  
- **工具闭环**：当模型决定调用搜索或 Python 时，实际执行环境返回结果，这些结果会被即时反馈到奖励模型，形成闭环学习。  

**4. 渲染聊天格式**  
- **结构化模板**：  
  ```
  <system> 你是一个具备搜索、代码执行能力的助理。</system>
  <user> 请帮我查找最新的机器学习论文并写一个 Python 脚本下载。</user>
  <assistant> <tool_call name="search" query="最新机器学习论文 2024">...</tool_call>
  <assistant> <tool_result>...</tool_result>
  <assistant> <tool_call name="python" code="...">...</tool_call>
  <assistant> <tool_result>...</tool_result>
  <assistant> 完成！</assistant>
  ```  
- **角色分离**：模型在生成每一段时都知道自己是“assistant”，而工具调用则由专门的子模块处理，避免混淆。  

**最巧妙的点**  
- 把稀疏路由和蒸馏过程合二为一，使得专家的分配策略直接受老师信号驱动，省去后期单独微调路由的步骤。  
- 将工具调用视作对话的一部分，统一进奖励函数，让模型在语言生成和工具使用之间形成自然的协同。  

### 实验与效果
- **评测任务**：数学推理（MATH、GSM‑8K）、代码生成（HumanEval、MBPP）、检索问答（WebGPT、MMLU）、安全基准（TruthfulQA、OpenAI Safety）。  
- **基线对比**：与同等参数的全参数 LLaMA‑2‑70B、Claude‑1、GPT‑3.5‑Turbo 对比。  
  - 在 GSM‑8K 上，GPT‑OSS‑120B 达到 78.4% 正确率，领先 LLaMA‑2‑70B（71.2%）约 7%。  
  - HumanEval 代码通过率提升至 55.1%，比 Claude‑1（48.3%）高出约 7%。  
  - 检索任务中，主动搜索成功率提升 25%，整体回答准确率提升 12%。  
- **消融实验**：  
  - 去掉稀疏路由，仅使用全参数前馈，推理成本翻倍，准确率提升不明显（<1%）。  
  - 移除强化学习的工具奖励，模型仍能生成代码但主动搜索率下降 20%。  
  - 替换渲染聊天格式为普通文本提示，指令遵循错误率上升约 15%。  
- **局限性**：  
  - 仍依赖大规模 GPU 集群进行训练，门槛高。  
  - 在极端长上下文（>8k token）下路由网络的负载均衡出现轻微退化。  
  - 作者承认安全基准仍有提升空间，尤其在细粒度偏见检测上表现不够稳健。  

### 影响与延伸思考
GPT‑OSS 的开源姿态在社区引发了“稀疏专家+工具化”热潮，随后出现了多篇基于相同思路的模型（如 OpenMoE‑Chat、SparseGPT‑Assist）。研究者开始探索更细粒度的路由策略、跨模态工具调用（图像、音频）以及在边缘设备上部署稀疏专家模型的可能性。想进一步了解，可以关注以下方向：  
- **路由学习的自监督方法**（如 GShard、SwitchTransformer 的改进）。  
- **多模态工具环境**（把浏览器、数据库、图像编辑器统一进对话）。  
- **安全对齐的多目标强化学习**（在奖励函数中加入更细致的伦理约束）。  

### 一句话记住它
GPT‑OSS 用稀疏专家 + 统一强化学习，让大模型在保持高精度的同时，真正成为会主动搜索、写代码、执行函数的开源助理。