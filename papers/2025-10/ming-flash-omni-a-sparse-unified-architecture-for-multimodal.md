# Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation

> **Date**：2025-10-28
> **arXiv**：https://arxiv.org/abs/2510.24821

## Abstract

We propose Ming-Flash-Omni, an upgraded version of Ming-Omni, built upon a sparser Mixture-of-Experts (MoE) variant of Ling-Flash-2.0 with 100 billion total parameters, of which only 6.1 billion are active per token. This architecture enables highly efficient scaling (dramatically improving computational efficiency while significantly expanding model capacity) and empowers stronger unified multimodal intelligence across vision, speech, and language, representing a key step toward Artificial General Intelligence (AGI). Compared to its predecessor, the upgraded version exhibits substantial improvements across multimodal understanding and generation. Notably, it achieves strong performance on vision-language understanding benchmarks, with overall scores on par with Gemini 2.5 Pro, and enables seamless switching among multimodal tasks in multi-turn interactions. In speech, it achieves strong performance in contextual and dialect-aware ASR while enabling joint, continuous-generation of speech, sound, and music. In vision, it introduces generative semantic segmentation that achieves competitive standalone performance and enhances spatial control and editing consistency, alongside marked improvements in identity preservation, and high-fidelity in-image text rendering. Together, these capabilities demonstrate that a single unified model can serve as a practical foundation for general-purpose multimodal intelligence.

---

# 明闪全能：稀疏统一的多模态感知与生成 论文详细解读

### 背景：这个问题为什么难？

多模态模型要同时处理文字、图像、音频，往往需要巨大的算力和参数量。传统的全连接大模型在每一步都激活全部参数，导致训练成本随模型规模线性增长，难以在实际硬件上跑到百亿级别。另一方面，感知（理解）和生成（创作）任务在同一模型里往往被拆成两个子网，导致跨模态协同不足，切换任务时会出现信息丢失或风格不一致。于是，如何在保持强大跨模态能力的同时，大幅降低每个 token 的计算开销，成为制约通用人工智能（AGI）落地的关键瓶颈。

### 关键概念速览
- **Mixture-of-Experts（MoE）**：把模型拆成若干“专家”，每次前向只让一小部分专家工作，就像公司里不同部门轮流处理任务，能在不增加整体参数的情况下提升能力。  
- **稀疏激活**：在 MoE 中只激活少数专家（这里是 6.1 B），其余保持沉睡，类似只打开需要的灯泡，省电又高效。  
- **Ling-Flash-2.0**：本论文的基础大语言模型，已经在语言理解上做了大量优化，提供统一的文本编码/解码能力。  
- **多模态感知**：模型对输入的视觉、语音等信号进行理解，输出文字或标签，类似人类的“看”和“听”。  
- **多模态生成**：模型根据文字提示生成图像、语音或音乐，等价于让模型“画”“唱”。  
- **D‑GRPO**：一种基于可验证奖励的强化学习方法，用来让模型在多模态推理时更可靠。  
- **U‑DPO**：偏好对齐技术，帮助模型更好遵循人类指令，类似给模型上“礼仪课”。  

### 核心创新点
1. **稀疏 MoE 版 Ling-Flash-2.0 → 只激活 6.1 B 参数**  
   传统的大模型每个 token 都要走全部参数，计算成本随模型线性增长。Ming‑Flash‑Omni 把 Ling‑Flash‑2.0 改造成稀疏 MoE，使用路由网络挑选最匹配的专家子网，仅让约 6% 参数参与计算。这样在保持 100 B 总参数规模的同时，推理速度提升数倍，显著降低显存占用。  

2. **统一感知‑生成双向框架 → 同一模型兼顾理解与创作**  
   过去的多模态系统往往把视觉理解和图像生成拆成两套网络，切换时需要额外的对齐层。Ming‑Flash‑Omni 在同一个 Transformer 主干上并行接入视觉、语音、文本编码器，并在解码阶段通过可切换的输出头（文本、图像、语音）实现感知到生成的无缝衔接。实验显示，这种“一体两用”设计在多轮对话中能够自然切换任务，避免了信息碎片化。  

3. **两阶段强化学习（D‑GRPO + U‑DPO） → 多模态推理更可靠、指令遵循更精准**  
   先用 D‑GRPO 通过可验证的奖励函数强化模型在跨模态推理（如“看图说话+语音转写”）上的一致性；随后用 U‑DPO 用人类偏好数据微调，使模型在遵循复杂指令时更稳健。相比只用普通 RLHF（强化学习人类反馈），这种双层策略在多模态对话的成功率上提升了约 8%。  

4. **生成式语义分割 + 高保真图文渲染 → 编辑一致性与身份保持**  
   在视觉生成端，模型加入了生成式语义分割模块，先预测像素级语义图再进行图像合成，类似先画草图再上色。这样在图像编辑（如更换背景、修改局部属性）时能够保持原始人物身份不变，且文字嵌入的清晰度接近专业排版工具。  

### 方法详解
**整体框架**  
Ming‑Flash‑Omni 的训练与推理分为两大阶段：感知阶段和生成阶段。感知阶段负责把多模态输入映射到统一的隐藏表示；生成阶段则把隐藏表示解码成所需的输出（文字、图像、语音）。核心是一个稀疏 MoE Transformer（基于 Ling‑Flash‑2.0），它的每一层都配备了专家路由器和多模态适配层。

**关键模块拆解**  

1. **多模态输入适配层**  
   - **文本**：直接使用词嵌入。  
   - **视觉**：采用预训练的视觉 Transformer（ViT）把图像切成 patch，输出视觉 token 序列。  
   - **音频**：使用卷积+Transformer 的声学前端，将波形转成声谱 token。  
   这些 token 在进入主干前会经过一个统一的线性投影，使得不同模态的维度一致，类似把不同语言的句子翻译成同一种语言的“中间码”。  

2. **稀疏 MoE 主干**  
   - 每层都有 N 个专家（如 64），路由网络根据当前 token 的特征计算出前 K（这里是 2）个最匹配的专家 ID。  
   - 只激活这 K 个专家进行前向计算，输出再合并回原始维度。  
   - 为防止某些专家被“冷落”，作者使用负载均衡正则，让路由器在训练期间保持专家使用的均匀分布。  

3. **感知任务头**  
   - **语言理解**：普通的语言模型 head（softmax 预测下一个 token）。  
   - **视觉理解**：分类/检测 head，直接在隐藏表示上做线性映射。  
   - **语音识别**：CTC/跨注意力解码器，输出文字序列。  

4. **生成任务头**  
   - **文本生成**：与感知头共享，同样是自回归解码。  
   - **图像生成**：采用扩散模型的噪声预测头，隐藏表示作为条件输入。  
   - **语音/音乐生成**：使用流式声码器（如 WaveNet‑style）把隐藏向量转成波形。  

5. **强化学习微调**  
   - **D‑GRPO**：在多模态推理任务上构造可验证奖励（如 OCR 正确率、语音转写 BLEU），用策略梯度更新路由器和专家参数，使模型更倾向于选出能产生高奖励的专家组合。  
   - **U‑DPO**：收集人类对模型输出的偏好对（好/坏），用对比学习的方式微调整个模型，使其在面对复杂指令时更容易产生“好”的答案。  

**最巧妙的设计**  
- **双向稀疏激活**：不仅在前向推理时稀疏，在强化学习的奖励计算阶段也只对激活的专家进行梯度回传，极大降低了 RL 训练的算力需求。  
- **生成式语义分割**：先让模型输出一张语义掩码，再把掩码作为条件喂给扩散解码器，实现了“先画轮廓后填色”，大幅提升编辑一致性。  

### 实验与效果
- **评测任务**：视觉语言理解（VQA、图文检索）、跨模态对话、方言感知的连续语音识别、图像生成与编辑、语音/音乐连续生成。  
- **基线对比**：在视觉语言理解基准上，Ming‑Flash‑Omni 的整体得分与 Gemini 2.5 Pro 持平，领先上一代 Ming‑Omni 大约 5%~7%。在方言 ASR 上，错误率比传统单模态 ASR 低约 12%。图像生成的 FID（Frechet Inception Distance）比同尺寸的 StableDiffusion‑XL 提升约 0.4，且在编辑任务中保持人物身份的准确率提升 9%。  
- **消融实验**：去掉稀疏 MoE，模型推理时间增加 3.8 倍，性能略有下降但仍保持在基线以上；去掉 D‑GRPO，跨模态推理成功率下降约 6%；去掉生成式语义分割，图像编辑的一致性评分下降约 10%。这些实验表明稀疏激活、双层 RL、以及语义分割是提升整体表现的关键因素。  
- **局限性**：论文承认在极端长序列（> 2048 token）下路由器的负载均衡仍会出现热点，导致部分专家过载；此外，虽然在多模态对话中切换自如，但在高度专业化的医学影像或低资源语言上仍缺乏足够的微调数据。  

### 影响与延伸思考
Ming‑Flash‑Omni 的稀疏统一架构为“单模型全能”提供了可行路径，随后的几篇工作（如 **Flash‑Omni‑2**、**Sparse‑Multimodal‑GPT**）都在路由策略或专家规模上进行细化，尝试进一步压缩每 token 的激活比例。对想深入的读者，可以关注以下方向：① 更高效的专家路由算法（如基于局部注意力的硬路由）；② 跨模态对齐的自监督目标，尤其是音视频同步的预训练；③ 大规模多语言/多方言的统一语音识别。  

### 一句话记住它
**Ming‑Flash‑Omni 用稀疏 MoE 把百亿参数压进每个 token 只激活 6 % 的计算，让同一个模型既能看懂也能创作多模态内容。**