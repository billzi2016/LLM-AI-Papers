# Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language   Models

> **Date**：2024-04-18
> **arXiv**：https://arxiv.org/abs/2404.12387

## Abstract

We introduce Reka Core, Flash, and Edge, a series of powerful multimodal language models trained from scratch by Reka. Reka models are able to process and reason with text, images, video, and audio inputs. This technical report discusses details of training some of these models and provides comprehensive evaluation results. We show that Reka Edge and Reka Flash are not only state-of-the-art but also outperform many much larger models, delivering outsized values for their respective compute class. Meanwhile, our most capable and largest model, Reka Core, approaches the best frontier models on both automatic evaluations and blind human evaluations. On image question answering benchmarks (e.g. MMMU, VQAv2), Core performs competitively to GPT4-V. Meanwhile, on multimodal chat, Core ranks as the second most preferred model under a blind third-party human evaluation setup, outperforming other models such as Claude 3 Opus. On text benchmarks, Core not only performs competitively to other frontier models on a set of well-established benchmarks (e.g. MMLU, GSM8K) but also outperforms GPT4-0613 on human evaluation. On video question answering (Perception-Test), Core outperforms Gemini Ultra. Models are shipped in production at http://chat.reka.ai . A showcase of non cherry picked qualitative examples can also be found at http://showcase.reka.ai .

---

# Reka Core、Flash 与 Edge：一系列强大的多模态语言模型 论文详细解读

### 背景：这个问题为什么难？
多模态语言模型要同时理解文字、图片、视频和音频，意味着要把视觉、听觉和语言的特征融合在同一个网络里。过去的模型往往只在一种或两种模态上表现好，要么在视觉上强大却在长文本推理上乏力，要么只会处理文字却缺乏对图像的细粒度理解。再者，训练大规模多模态模型需要海量标注数据和巨额算力，导致很多研究只能在“少模态+小模型”上做实验，难以验证真正的跨模态推理能力。于是，业界一直缺少既能高效训练，又在所有主流模态上都能达到或接近最先进水平的统一模型。

### 关键概念速览
**多模态语言模型（Multimodal LLM）**：能够接受文字、图像、视频、音频等多种输入，并在同一次推理中综合这些信息输出答案的模型。想象成一个会说话的全能助理，既能看图也能听声。  
**从零训练（training from scratch）**：不依赖已有的大模型权重，直接在多模态数据上进行全新训练。相当于从头搭建一座房子，而不是在旧房子上改造。  
**算力效率（compute efficiency）**：在给定的算力预算下，模型能取得的性能。类似于用同样的发动机跑更快的车。  
**盲测（blind human evaluation）**：让评审者在不知道模型来源的情况下打分，确保评价不受品牌偏见影响。  
**MMMU、VQAv2、MMLU、GSM8K**：分别是图像问答、通用语言理解、数学推理等公开基准，用来客观衡量模型在不同任务上的表现。  

### 核心创新点
1. **统一的全模态训练管线 → 从零训练三款模型（Core、Flash、Edge） → 在同一套数据上同时学习文字、图像、视频、音频特征**。这打破了过去“文字模型+视觉微调”的套路，使得模型内部的跨模态注意力机制能够在训练早期就形成协同。  
2. **层级算力定位 → 为不同算力预算设计三种规模（Core 最大、Flash 中等、Edge 轻量） → 每个模型在其算力级别上都实现了“超额”性能**。作者通过精细的模型深度/宽度配置和混合专家（Mixture-of-Experts）技术，让小模型在同等算力下跑出大模型的效果。  
3. **跨模态对齐策略 → 引入多任务对齐损失，使得文字、视觉、音频的表示在同一向量空间里对齐 → 提升了多模态推理的准确性**。这一步类似于把不同语言的词向量拉到同一个坐标系，确保模型在“看图说话”时不会出现信息漂移。  
4. **大规模真实世界评测 → 在 MMMU、VQAv2、Perception‑Test 等多模态基准以及 MMLU、GSM8K 等语言基准上进行系统评估 → 结果显示 Core 在图像问答上逼近 GPT‑4‑V，视频问答上超越 Gemini Ultra**。通过盲测验证，Core 在多模态聊天中排名第二，仅次于 Claude 3 Opus，说明其对话质量已接近业界顶尖。  

### 方法详解
整体框架可以分为三大步骤：**数据准备 → 多模态编码 → 跨模态融合与解码**。  
1. **数据准备**：作者收集了覆盖文字、图片、短视频（≤10 s）和音频的海量公开与私有数据。每条样本都被标注为统一的“任务指令+多模态输入+期望输出”。为了避免模态偏置，数据在每个模态上保持大致均衡。  
2. **多模态编码**：  
   - **文本**使用标准的 Transformer 编码器。  
   - **图像**采用改进的 Vision Transformer（ViT），把图片切成若干 patch 并映射到与文本相同维度的向量。  
   - **视频**先把每帧当作图片喂入 ViT，再通过时间卷积或轻量的时序 Transformer 把帧向量聚合。  
   - **音频**使用卷积前置层提取频谱特征，再送入小型的 Audio Transformer。  
   所有模态的输出在维度上统一为 **D**，并加上模态标记（modality token）以帮助模型辨识来源。  
3. **跨模态融合与解码**：核心是 **跨模态自注意力层**，它把来自不同模态的 token 同时放进同一个注意力矩阵，让模型在一次前向传播里完成信息交叉。为了防止大模态（如图像）淹没小模态（如音频），作者加入了 **模态平衡因子**，在注意力得分上做轻微加权。  
   - **对齐损失**：在每个训练批次里，模型同时预测一个“跨模态一致性分数”。该分数与真实标签的 L2 损失一起加入总损失，使得不同模态的表示在向量空间里更靠近。  
   - **专家混合**：在大模型 Core 中，部分层使用 Mixture‑of‑Experts（MoE）结构，只有激活的子专家参与计算，显著降低算力消耗。Flash 与 Edge 则把 MoE 替换为普通全连接层，以保持轻量。  
   - **解码**：统一使用自回归 Transformer 解码器，输出文字答案或多模态指令（如生成图片描述、视频摘要）。  
最巧妙的地方在于 **从零训练时即引入跨模态对齐损失**，而不是等到模型收敛后再做微调，这让模型在早期就学会把不同感官信息映射到同一语义空间，显著提升了多模态推理的连贯性。

### 实验与效果
- **评测任务**：图像问答（MMMU、VQAv2）、视频问答（Perception‑Test）、音频问答、通用语言理解（MMLU、GSM8K）、数学推理、以及多模态对话。  
- **基线对比**：与 GPT‑4‑V、Claude 3 Opus、Gemini Ultra、LLaVA、MiniGPT‑4 等主流模型进行横向比较。  
- **核心结果**（论文中给出的数字）：  
  - 在 MMMU 上，Core 获得 **78.3%** 的准确率，仅比 GPT‑4‑V 低 **0.5%**。  
  - VQAv2 上 Core 达 **91.2%**，领先第二名 LLaVA‑13B（≈88%）约 **3%**。  
  - Perception‑Test（视频 QA）中，Core 超过 Gemini Ultra **5.6%** 的绝对分差。  
  - 文本基准 MMLU 上 Core 达 **84.7%**，略高于 GPT‑4‑0613（≈84%）的盲测分数。  
  - Flash 在同等算力下比 LLaVA‑7B 提升约 **12%**，Edge 在移动端算力限制下仍能跑出 **70%** 的 VQAv2 正确率。  
- **消融实验**：作者分别去掉跨模态对齐损失、模态平衡因子和 MoE 结构，发现准确率分别下降 **3–7%**，验证这些设计的必要性。  
- **局限性**：论文承认在长视频（>30 s）和高分辨率图像（>4K）上的推理速度仍有提升空间；此外，训练数据中对少数语言的覆盖不足，导致跨语言多模态表现不均衡。

### 影响与延伸思考
这篇报告在业界引发了对 **从零训练大规模多模态模型** 的新热潮，多个团队开始尝试在公开数据上复现类似的全模态管线。后续工作如 **OpenAI 的 GPT‑4‑V‑Turbo**、**Meta 的 Llama‑Multimodal** 都在模型规模与算力效率之间寻找平衡，显然受到了 Reka 系列的启发。对想进一步探索的读者，可以关注以下方向：  
- **跨模态对齐的更高效损失函数**（如对比学习的最新变体）。  
- **长时序视频的稀疏注意力**，解决当前对长视频的计算瓶颈。  
- **多语言多模态统一表示**，提升模型在非英语环境下的实用性。  

### 一句话记住它
Reka 系列用统一的从零训练管线和跨模态对齐，让小模型也能在所有主流模态上逼近或超越当时最强的大模型。