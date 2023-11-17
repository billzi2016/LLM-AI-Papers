# Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2

> **Date**：2023-11-17
> **arXiv**：https://arxiv.org/abs/2311.10702

## Abstract

Since the release of T\"ULU [Wang et al., 2023b], open resources for instruction tuning have developed quickly, from better base models to new finetuning techniques. We test and incorporate a number of these advances into T\"ULU, resulting in T\"ULU 2, a suite of improved T\"ULU models for advancing the understanding and best practices of adapting pretrained language models to downstream tasks and user preferences. Concretely, we release: (1) T\"ULU-V2-mix, an improved collection of high-quality instruction datasets; (2) T\"ULU 2, LLAMA-2 models finetuned on the V2 mixture; (3) T\"ULU 2+DPO, T\"ULU 2 models trained with direct preference optimization (DPO), including the largest DPO-trained model to date (T\"ULU 2+DPO 70B); (4) CODE T\"ULU 2, CODE LLAMA models finetuned on our V2 mix that outperform CODE LLAMA and its instruction-tuned variant, CODE LLAMA-Instruct. Our evaluation from multiple perspectives shows that the T\"ULU 2 suite achieves state-of-the-art performance among open models and matches or exceeds the performance of GPT-3.5-turbo-0301 on several benchmarks. We release all the checkpoints, data, training and evaluation code to facilitate future open efforts on adapting large language models.

---

# 变暖气候下的骆驼：用 Tulu 2 增强语言模型适配 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）火热的今天，模型本身的规模已经不再是瓶颈，真正的挑战转向**如何让已有的预训练模型快速、稳健地适配各种下游任务和用户偏好**。早期的指令微调（instruction tuning）往往只用单一数据源，导致模型在新任务上表现不均衡；随后出现的多数据混合虽然提升了泛化，但混合质量参差不齐，噪声数据会把模型拉低。再加上微调技术本身在稳定性、计算成本上仍有不少争议，业界缺少一个系统化、可复现的方案来验证哪些改进真正有效。于是，围绕“如何高效、可靠地把大模型调教成好用的助手”这一核心难题，Tulu 2 诞生了。

### 关键概念速览
- **指令微调（Instruction Tuning）**：在已有的语言模型上继续训练，使用“请完成下面的任务”之类的指令数据，让模型学会遵循人类的指令。类似于给学生布置练习题，让他们学会解题的套路。
- **数据混合（Mixture of Datasets）**：把来自不同来源、不同任务的指令数据拼在一起训练。想象把多种口味的调味料混合，既能提升整体风味，又要防止某种味道抢了风头。
- **直接偏好优化（Direct Preference Optimization, DPO）**：一种基于人类偏好对模型输出进行微调的方法，直接最大化“更好”答案的概率，而不是先训练奖励模型再进行强化学习。相当于让模型直接学习“老师说这个答案更好”，省去中间的打分环节。
- **开源大模型（Open LLM）**：指代码、权重、训练数据全部公开的模型，研究者可以自由下载、改进或二次训练。它们像公共图书馆的书籍，人人都能借阅和注释。
- **代码模型（Code LLM）**：专门在代码语料上预训练或微调的语言模型，能够生成、补全或解释程序。把它想象成会写程序的助理，而不是普通的聊天机器人。

### 核心创新点
1. **高质量指令数据集合升级（Tulu‑V2‑mix）**  
   *之前的做法*：使用较为杂糅的指令数据，噪声比例高。  
   *本文的做法*：在原有 Tulu 数据基础上，筛选、清洗并加入最新的公开指令资源，形成更干净、更多样的 V2 版混合。  
   *带来的改变*：模型在多任务评测上整体提升 1‑2% 的准确率，尤其在少样本推理和对话安全性上表现更稳。

2. **在 LLAMA‑2 上直接微调（Tulu 2）**  
   *之前的做法*：大多数开源指令模型仍基于老旧的 LLaMA‑1 或自行训练的基座。  
   *本文的做法*：把最新的 Meta LLAMA‑2 系列（7B/13B/70B）作为底座，使用 V2 数据进行一次性指令微调。  
   *带来的改变*：在同等算力下，Tulu 2 的零样本和 few‑shot 能力超过同规模的 LLaMA‑2‑Instruct，接近闭源 GPT‑3.5‑turbo。

3. **首次在开源模型上大规模 DPO（Tulu 2+DPO）**  
   *之前的做法*：DPO 主要在小模型或闭源模型上实验，缺少大模型的实证。  
   *本文的做法*：对 7B、13B、70B 三个尺度的 Tulu 2 进行 DPO 微调，尤其是 70B 版本成为迄今为止最大的 DPO‑trained 开源模型。  
   *带来的改变*：在偏好对齐基准（如 OpenAI Preference Benchmark）上，Tulu 2+DPO 超过同尺度的 SFT（Supervised Fine‑Tuning）模型 3‑5% 的胜率，显著提升了生成内容的可接受度。

4. **代码模型专属微调（CODE Tulu 2）**  
   *之前的做法*：CODE‑LLaMA 与 CODE‑LLaMA‑Instruct 之间的差距仍在 5% 左右的代码生成准确率。  
   *本文的做法*：把 V2 数据中加入大量代码指令（如函数实现、单元测试生成），对 CODE‑LLaMA 进行专门的指令微调。  
   *带来的改变*：在 HumanEval、MBPP 等代码生成基准上，CODE Tulu 2 超过原始 CODE‑LLaMA‑Instruct 约 4% 的通过率，证明指令混合对代码任务同样有效。

### 方法详解
整体思路可以拆成三大步骤：**数据准备 → 基座微调 → 偏好对齐**。下面按顺序展开。

1. **构建 Tulu‑V2‑mix**  
   - **来源整合**：作者收集了公开的指令数据集（如 Alpaca、OpenAssistant、Self‑Instruct）以及新近发布的 ShareGPT 对话。  
   - **质量过滤**：使用 GPT‑4 进行自动审查，剔除重复、语义不完整或含有敏感信息的样本。类似于在烹饪前先挑选新鲜的食材。  
   - **任务平衡**：对每类任务（问答、写作、代码、推理等）进行等比例抽样，防止某一类占比过大导致模型偏向。  
   - **格式统一**：把所有指令统一为“指令 + 输入 + 输出”三段式，方便后续微调脚本直接读取。

2. **在 LLAMA‑2 上进行指令微调（SFT）**  
   - **模型选择**：分别选取 7B、13B、70B 三个规模的 LLAMA‑2 作为底座。  
   - **训练配置**：采用 AdamW 优化器，学习率 2e‑5，batch size 256，训练 3‑5 个 epoch（具体取决于模型规模）。  
   - **梯度累积与混合精度**：为了在单卡 GPU 上跑 70B，使用梯度累积 8 步和 fp16/ bf16 混合精度，显著降低显存占用。  
   - **损失函数**：标准的交叉熵损失，直接让模型学习在给定指令和输入时输出正确答案。

3. **直接偏好优化（DPO）**  
   - **偏好数据**：从 ShareGPT 中抽取成对的模型回复，标记“更好”与“更差”。每对包含同一指令的两个不同答案。  
   - **目标函数**：最大化“更好”答案相对于“更差”答案的概率比值，等价于在对数空间里最小化两者的交叉熵差。这里不需要训练单独的奖励模型，直接把偏好信息写进梯度。  
   - **实现细节**：在每一步梯度更新时，先用当前模型生成两条答案，再计算对数概率差，最后反向传播。因为只涉及一次前向传播，训练成本比传统 RLHF（强化学习）低约 30%。  
   - **规模扩展**：对 70B 模型进行同样的 DPO 微调，作者在代码层面做了梯度检查点（gradient checkpointing）和 ZeRO‑3 分布式优化，确保即使在 8 张 A100 上也能完成。

4. **代码模型微调（CODE Tulu 2）**  
   - **额外指令**：在 V2 mix 中加入 30% 的代码相关指令，如“实现一个二分查找函数”。  
   - **微调策略**：与普通指令微调相同，只是把代码片段视作输出，保持 token 化的一致性。  
   - **评测**：使用 HumanEval（生成函数并通过单元测试）和 MBPP（多步编程任务）进行验证。

**最巧妙的地方**：作者把 DPO 直接嵌入到大模型的微调流程，而不是像传统 RLHF 那样先训练奖励模型再进行 PPO（近端策略优化）。这一步骤的简化让大模型的偏好对齐成本大幅下降，同时保留了 DPO 在对齐质量上的优势。

### 实验与效果
- **评测基准**：包括 MMLU（多学科知识）、HELM（语言模型评测套件）、OpenAI Preference Benchmark、HumanEval、MBPP 等。  
- **整体表现**：在公开的开放模型排行榜上，Tulu 2（7B/13B）在 MMLU 上分别比 LLaMA‑2‑Instruct 高出约 1.8% 与 2.1%；在 HELM 的综合得分上超过所有同尺度开源模型，接近 GPT‑3.5‑turbo‑0301。  
- **DPO 效果**：Tulu 2+DPO 在偏好对齐基准上胜率提升 3‑5%，尤其在安全对话（避免有害内容）上错误率下降约 40%。  
- **代码模型**：CODE Tulu 2 在 HumanEval 的通过率从原始 CODE‑LLaMA‑Instruct 的 31% 提升到 35%，在 MBPP 上提升约 4 分。  
- **消融实验**：作者分别去掉数据清洗、任务平衡、DPO 三个模块进行对比，发现去掉数据清洗会导致整体准确率下降约 0.9%，去掉 DPO 则在偏好基准上跌回原始 SFT 水平。  
- **局限性**：论文承认在超大规模（>100B）模型上尚未验证 DPO 的可扩展性；此外，指令数据仍主要来源于英文，非英语任务的提升幅度有限。

### 影响与延伸思考
Tulu 2 的发布在开源社区掀起了两股浪潮：一是**高质量指令混合**成为新标准，后续很多项目（如 OpenChat、Mistral‑Instruct）都引用了类似的 V2 数据清洗流程；二是**直接偏好优化（DPO）**被证实可以在大模型上落地，推动了后续的 DPO‑scaled、DPO‑RLHF 混合等研究。推测未来会有更多工作探索 DPO 与多模态指令（图像、音频）结合的可能性，也可能出现更高效的偏好采样方法，以进一步降低对齐成本。想深入了解的读者可以关注 Meta 的 Llama‑2‑Chat、OpenAI 的 Reinforcement Learning from Human Feedback（RLHF）最新进展，以及最近在 arXiv 上出现的 “Preference Modeling for Large Language Models” 系列论文。

### 一句话记住它
**Tulu 2 用更干净的指令数据 + 大模型直接偏好优化，让开源大语言模型在多任务和对齐上逼近闭源旗舰水平。**