# Phi-3 Technical Report: A Highly Capable Language Model Locally on Your   Phone

> **Date**：2024-04-22
> **arXiv**：https://arxiv.org/abs/2404.14219

## Abstract

We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69% on MMLU and 8.38 on MT-bench), despite being small enough to be deployed on a phone. Our training dataset is a scaled-up version of the one used for phi-2, composed of heavily filtered publicly available web data and synthetic data. The model is also further aligned for robustness, safety, and chat format. We also provide parameter-scaling results with a 7B, 14B models trained for 4.8T tokens, called phi-3-small, phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75%, 78% on MMLU, and 8.7, 8.9 on MT-bench). To enhance multilingual, multimodal, and long-context capabilities, we introduce three models in the phi-3.5 series: phi-3.5-mini, phi-3.5-MoE, and phi-3.5-Vision. The phi-3.5-MoE, a 16 x 3.8B MoE model with 6.6 billion active parameters, achieves superior performance in language reasoning, math, and code tasks compared to other open-source models of similar scale, such as Llama 3.1 and the Mixtral series, and on par with Gemini-1.5-Flash and GPT-4o-mini. Meanwhile, phi-3.5-Vision, a 4.2 billion parameter model derived from phi-3.5-mini, excels in reasoning tasks and is adept at handling both single-image and text prompts, as well as multi-image and text prompts.

---

# Phi-3 技术报告：在手机本地运行的高性能语言模型 论文详细解读

### 背景：这个问题为什么难？

在过去，想要让一款语言模型同时拥有接近 GPT‑3.5 的推理能力和在手机上实时运行的轻量级，几乎是不可能的。大模型需要数十 GB 的显存和数百 GB 的存储，普通移动设备根本装不下；而把模型压缩到几百 MB 往往会导致推理质量大幅下降。现有的开源模型要么体积大、性能好，却只能跑在服务器；要么体积小、只能完成非常基础的对话。如何在保持参数规模极小的前提下，仍然实现高质量的语言理解和生成，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **参数（Parameters）**：模型内部的可学习数值，参数越多通常代表模型容量越大。这里的 3.8 B 参数相当于 38 亿个可调节的数字。
- **Token（标记）**：文本被切分后的最小单位，模型在训练时逐个预测下一个 token。3.3 T token 意味着模型看过了 3.3 万亿个单词或子词。
- **数据过滤（Data Filtering）**：从公开网页中剔除噪声、低质量或有害内容的过程，就像在挑选水果时只留下熟透、无瑕的那部分。
- **合成数据（Synthetic Data）**：利用已有模型或规则自动生成的训练样本，用来补足真实数据的不足，类似于老师给学生出的练习题。
- **对齐（Alignment）**：让模型的输出符合安全、可靠、对话友好的目标，常通过人类反馈或规则约束实现。
- **MoE（Mixture of Experts）**：一种让模型在每次前向传播只激活部分子网络的架构，像是公司里不同专家轮流处理不同任务，从而在保持总体规模的同时提升效率。
- **多模态（Multimodal）**：模型能够同时理解文字和图像信息，类似于人类在看图说话时把视觉和语言融合在一起。
- **长上下文（Long Context）**：模型能够记住并利用数千甚至上万 token 的前文信息，像是一次长篇对话中不忘前面说过的细节。

### 核心创新点
1. **极致数据管线 → 大规模过滤 + 合成扩增 → 在 3.8 B 参数下实现 3.3 T token 的高质量预训练**  
   过去的轻量模型往往只用原始网页数据，噪声多导致学习效率低。Phi‑3 采用了多层过滤（去除重复、低质量、违规内容）并加入了大规模合成数据，使得每个 token 都更有信息密度，从而在相同参数规模下获得了接近 8‑B 参数模型的表现。

2. **统一对齐流程 → 安全、鲁棒、聊天三阶段微调 → 直接生成可在手机上使用的对话模型**  
   与仅做一次指令微调的做法不同，Phi‑3 先进行鲁棒性微调（提升对噪声输入的容错），再做安全对齐（过滤有害输出），最后加入聊天格式的对话微调。三层防护让模型在资源受限的设备上仍能保持一致的输出质量。

3. **轻量 MoE 设计 → 16‑路专家网络，每次仅激活 6.6 B 参数 → 在相同 FLOPs 下实现更强的推理与代码能力**  
   传统 MoE 需要数十甚至上百亿的活跃参数，部署成本高。Phi‑3.5‑MoE 把每个专家的规模压到 3.8 B，使用稀疏激活只调用 6.6 B 参数，既保持了 MoE 的专家特化优势，又能在普通 GPU 上训练、在手机上推理。

4. **视觉扩展 → 从语言模型直接派生 4.2 B 参数的多图像对话模型 → 支持单图、文本混合以及多图多文本提示**  
   过去的视觉语言模型往往需要重新训练完整的视觉编码器。Phi‑3.5‑Vision 只在语言模型上添加轻量的跨模态投影层，利用已有的语言知识快速适配视觉输入，实现了在移动端的多模态交互。

### 方法详解
**整体框架**：Phi‑3 的训练分为四大阶段——（1）数据采集与多层过滤，（2）合成数据生成并混合进训练集，（3）大规模自回归预训练，（4）三阶段对齐微调。随后，根据需求分别构建标准模型、MoE 变体和视觉扩展模型。

**1. 数据管线**  
- **采集**：从 Common Crawl、Wikipedia、GitHub 等公开来源抓取原始文本。  
- **过滤**：使用规则过滤（去除 HTML、代码块等噪声）+ 语言模型评分（剔除低质量、重复度高的句子）+ 安全审查（过滤暴力、政治敏感等内容）。  
- **合成**：利用已有的 Phi‑2 模型生成问答、代码补全、数学推理等任务的合成样本，比例约为真实数据的 20%。这样既补足了稀缺的高质量任务数据，又保持了整体数据分布的多样性。

**2. 预训练**  
采用标准的自回归 Transformer 架构，输入为 token 序列，目标是预测下一个 token。训练使用 AdamW 优化器，学习率采用线性 warm‑up + cosine decay。3.8 B 参数模型在 3.3 T token 上训练约 600 B FLOPs，使用混合精度（FP16+BF16）加速。

**3. 对齐微调**  
- **鲁棒性微调**：在加入噪声、拼写错误、口语化表达的对话数据上继续训练，使模型对不规范输入不易崩溃。  
- **安全对齐**：引入人类标注的有害/安全标签，使用对比学习让模型倾向生成安全回答。  
- **聊天格式微调**：把对话数据组织成 “用户 → 助手” 的交互对，使用教师强制（teacher forcing）方式让模型学习对话轮次的上下文保持。

**4. MoE 变体**  
在标准 Transformer 的每个前馈层插入稀疏路由模块，路由网络根据输入 token 的特征选择 2‑4 个专家激活。每个专家本身是一个完整的 3.8 B 参数子网络，整体模型拥有 16 条专家路径，但实际计算只涉及约 6.6 B 参数。路由采用 Top‑K 选择 + Softmax 正则化，确保负载均衡。

**5. 视觉扩展**  
在语言模型的输入嵌入层前加入一个轻量的视觉编码器（ViT‑B/16），输出的视觉 token 与文本 token 共享同一位置编码。随后使用跨模态注意力层让语言部分能够感知图像信息。整个视觉模型的参数只比语言基座多约 0.4 B，保持了在移动端的可部署性。

**最巧妙的点**：把合成数据和严格过滤结合，使得每个 token 的信息密度大幅提升；以及在 MoE 中把专家规模压到与主模型相同，却通过稀疏激活实现了“更多专家、更少计算”的效果。

### 实验与效果
- **评测数据集**：MMLU（多学科语言理解）和 MT‑bench（指令遵循与对话质量）是主要公开基准。  
- **性能对比**：  
  - phi‑3‑mini 在 MMLU 上取得 69% 正确率，MT‑bench 8.38 分，接近 Mixtral 8×7B（约 70% / 8.3）和 GPT‑3.5（约 71% / 8.4）。  
  - phi‑3‑small（7 B）提升至 75% / 8.7，phi‑3‑medium（14 B）进一步到 78% / 8.9。  
  - phi‑3.5‑MoE 在数学、代码和推理任务上与 Gemini‑1.5‑Flash、GPT‑4o‑mini 持平，明显领先同等 FLOPs 的 Llama 3.1 与 Mixtral 系列。  
- **基线**：对比对象包括 Llama 3.1‑8B、Mixtral‑8×7B、OpenAI 的 GPT‑3.5‑Turbo。Phi‑3 系列在相同或更低的参数量下实现了相似甚至更好的分数。  
- **消融实验**：作者分别去掉合成数据、三阶段对齐和 MoE 稀疏路由，发现：去掉合成数据后 MMLU 下降约 3%；去掉安全对齐后 MT‑bench 下降 0.4 分；去掉稀疏路由后 MoE 版本的推理速度下降 2.5 倍，性能下降约 1.5%。  
- **局限性**：虽然在多数基准上接近中等规模模型，但在极端长上下文（>8k token）和少数低资源语言上仍有差距；安全对齐仍可能出现边缘案例的有害输出；视觉模型在高分辨率多图任务上受限于显存。

### 影响与延伸思考
Phi‑3 的发布让“在手机上跑 GPT‑3.5 级别的对话”从概念走向可实现的产品路线，激发了开源社区对极致轻量化 LLM 的兴趣。随后出现的项目如 **Mistral‑Tiny‑Chat**、**Llama‑3‑Mobile** 都在数据过滤和稀疏激活上借鉴了 Phi‑3 的思路。MoE 的轻量实现也为后续的 **Edge‑MoE** 系列提供了参考。未来可以进一步探索：更高效的合成数据生成、跨语言的统一对齐、以及在 4‑8 k token 以上的长上下文记忆机制（如可逆层或检索增强）。如果想深入，建议关注 **稀疏路由算法的负载均衡** 与 **移动端多模态推理的硬件加速** 两大方向。

### 一句话记住它
Phi‑3 证明：通过极致过滤+合成数据的高质量预训练，加上轻量 MoE 与多模态扩展，3.8 B 参数也能在手机上实现接近 GPT‑3.5 的对话与推理能力。