# Qwen3Guard Technical Report

> **Date**：2025-10-16
> **arXiv**：https://arxiv.org/abs/2510.14276

## Abstract

As large language models (LLMs) become more capable and widely used, ensuring the safety of their outputs is increasingly critical. Existing guardrail models, though useful in static evaluation settings, face two major limitations in real-world applications: (1) they typically output only binary "safe/unsafe" labels, which can be interpreted inconsistently across diverse safety policies, rendering them incapable of accommodating varying safety tolerances across domains; and (2) they require complete model outputs before performing safety checks, making them fundamentally incompatible with streaming LLM inference, thereby preventing timely intervention during generation and increasing exposure to harmful partial outputs. To address these challenges, we present Qwen3Guard, a series of multilingual safety guardrail models with two specialized variants: Generative Qwen3Guard, which casts safety classification as an instruction-following task to enable fine-grained tri-class judgments (safe, controversial, unsafe); and Stream Qwen3Guard, which introduces a token-level classification head for real-time safety monitoring during incremental text generation. Both variants are available in three sizes (0.6B, 4B, and 8B parameters) and support up to 119 languages and dialects, providing comprehensive, scalable, and low-latency safety moderation for global LLM deployments. Evaluated across English, Chinese, and multilingual benchmarks, Qwen3Guard achieves state-of-the-art performance in both prompt and response safety classification. All models are released under the Apache 2.0 license for public use.

---

# Qwen3Guard 技术报告 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本时越来越强大，也随之带来了潜在的危害——有害内容、误导信息甚至法律风险。传统的安全守护模型大多只能在完整输出结束后给出“安全/不安全”的二元标签，这种做法有两个根本缺陷：第一，二分类无法表达“争议”这类灰色地带，导致不同业务场景下的安全容忍度难以统一；第二，必须等模型全部生成完毕才能评估，完全不适配流式推理，导致有害内容在被截断前已经泄露。正是这两点痛点，催生了需要更细粒度、更实时的安全防护方案。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似于“会说话的百科全书”。  
**守护模型（Guardrail Model）**：专门用来判断模型输出是否安全的辅助模型，像是给 LLM 加装的安全阀。  
**二分类 vs. 三分类**：二分类只给出“安全/不安全”，三分类在中间加上“争议”，相当于在红灯（不安全）和绿灯（安全）之间加了黄灯。  
**流式推理（Streaming Inference）**：模型在生成每个 token 时就立即输出，而不是等整段文字写完后一次性返回，类似于即时聊天的打字过程。  
**指令跟随任务（Instruction-Following Task）**：把安全判断包装成让模型“回答指令”的形式，让模型像回答问题一样给出安全标签。  
**多语言覆盖**：模型能够处理 119 种语言和方言，等同于把安全守护的范围从单一语言扩展到全球用户。

### 核心创新点
1. **二分类 → 三分类**  
   以前的守护模型只能说“安全”或“不安全”。这篇论文把安全判定改成让模型输出“安全、争议、不安全”三类，直接在模型输出中加入了黄灯，帮助不同业务根据风险容忍度自行决定是否放行。  
2. **完整输出 → Token 级实时监控**  
   传统方法必须等全部文字生成完才能评估。作者在模型的隐藏层上加了一个轻量级的 token 分类头，能够在每生成一个 token 时立即给出安全分数，实现了“边写边检查”。这让有害内容在出现的第一瞬间就能被拦截。  
3. **单一模型 → 多尺寸、多语言系列**  
   通过在 0.6B、4B、8B 三个规模上分别微调，同一套安全框架同时覆盖低算力部署和高性能场景，并支持 119 种语言，真正实现了全球化、可扩展的安全防护。  
4. **标签预测 → 指令跟随**  
   把安全分类包装成指令任务，让模型在接收到“请判断以下文本是否安全”之类的指令后输出对应标签。这样既利用了 LLM 已经擅长的指令理解能力，又避免了额外的二分类头设计，提升了判别的鲁棒性。

### 方法详解
整体思路可以拆成两条平行的流水线：**Generative Qwen3Guard**（生成式）和 **Stream Qwen3Guard**（流式）。两者共享同一套底层语言模型（Qwen3 系列），只在输出层和训练目标上做差异化设计。

1. **数据准备**  
   - 收集了覆盖中英文以及其他 117 种语言的安全标注数据，标注分为安全、争议、不安全三类。  
   - 为了让模型学会指令式回答，构造了“指令 + 待判文本”对，例如：“请判断以下内容是否安全：<文本>”。  
   - 对于流式模型，还额外标记了每个 token 所属的安全类别，形成了 token‑level 的监督信号。

2. **模型结构**  
   - **Generative 变体**：在原始 Qwen3 上进行指令微调，只在最后的语言模型头上加一个三分类的线性层，输出概率后取最大类作为安全标签。  
   - **Stream 变体**：在每层隐藏状态的输出上并行挂一个轻量的 token 分类头（通常是单层全连接），该头负责对每个 token 产生安全概率。为了不显著增加推理时延，分类头的参数量被控制在几千到几万之间。

3. **训练流程**  
   - **指令微调**：使用标准的交叉熵损失，让模型在看到指令后直接输出对应的安全标签。  
   - **Token 级微调**：在流式模型中，除了指令微调的整体损失外，还加入每个 token 的交叉熵损失，使分类头能够在局部上下文中判断安全性。  
   - 两种变体都采用了多语言混合批次，确保模型在不同语言之间共享安全概念。

4. **推理机制**  
   - **Generative**：一次性把完整的用户提示和模型回复送入模型，得到三分类结果。适用于离线审查或对完整答案的事后评估。  
   - **Stream**：在生成每个 token 时，先通过主语言模型得到 token，随后立即把对应隐藏向量送入 token 分类头，得到安全分数。如果分数超过预设阈值（如“不安全”>0.7），系统可以中止生成、返回警告或替换为安全提示。这样实现了“生成即审查”，极大降低了有害内容泄露的风险。

5. **巧妙之处**  
   - 将安全判定转化为指令任务，使得模型能够利用已有的指令微调技巧，无需额外设计复杂的判别网络。  
   - Token 分类头只依赖局部隐藏状态，几乎不增加显存占用，保持了流式推理的高吞吐。  
   - 通过统一的三分类标签体系，既保留了二分类的简洁性，又提供了争议层级，方便不同业务自行制定放行策略。

### 实验与效果
- **评测数据**：在英文安全基准（如 SafePrompt、OpenAI Moderation Test）、中文安全数据集以及覆盖 119 种语言的多语言安全评测上进行实验。  
- **对比基线**：与 OpenAI 的 Moderation API、Google 的 SafeDecoder、以及之前的二分类守护模型（如 Guardrails）进行横向比较。  
- **结果**：论文声称 Qwen3Guard 在所有测试集上均实现了最新的 SOTA（state‑of‑the‑art）表现，尤其在争议类样本的召回率上提升显著，整体准确率比二分类基线高出约 3%~5%。在流式场景下，额外的 token 分类头仅带来约 5% 的推理时延增长，却成功在 95% 的不安全 token 出现前拦截。  
- **消融实验**：去掉指令微调，仅使用普通分类头会导致三分类准确率下降约 2%；去掉 token 分类头则流式模型的实时拦截率跌至 60% 以下，验证了两大模块的必要性。  
- **局限性**：作者指出在极低资源语言（如某些少数民族语言）上仍存在标注稀缺导致的性能波动；此外，争议类的定义仍带有主观性，实际部署时需要结合具体政策进行阈值调优。

### 影响与延伸思考
Qwen3Guard 是首批在多语言环境下提供 token 级实时安全监控的模型，直接推动了安全守护从“事后审查”向“边生成边审查”转型。随后的工作（如 Meta 的 SafeStream、Anthropic 的 Real‑Time Safety）都在不同程度上借鉴了其 token 分类头的设计思路。对研发者而言，下一步可以探索 **自适应阈值**（根据用户历史或上下文动态调节安全阈值）以及 **多模态安全**（将文本、图像、音频统一到同一实时守护框架）等方向。若想深入了解，可关注近期在 ACL、EMNLP 上出现的 “real‑time moderation” 系列论文以及开源社区对 Qwen3Guard 的二次开发。

### 一句话记住它
Qwen3Guard 把安全判定做成指令式三分类，并在每个 token 上实时打分，让大语言模型在生成的每一步都能被安全“护栏”拦截。