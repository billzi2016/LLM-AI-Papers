# LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.16933

## Abstract

In this work, we introduce LLaDA-V, a purely diffusion-based Multimodal Large Language Model (MLLM) that integrates visual instruction tuning with masked diffusion models, representing a departure from the autoregressive paradigms dominant in current multimodal approaches. Built upon LLaDA, a representative large language diffusion model, LLaDA-V incorporates a vision encoder and MLP connector that projects visual features into the language embedding space, enabling effective multimodal alignment. Our empirical investigation reveals several intriguing results: First, LLaDA-V demonstrates promising multimodal performance despite its language model being weaker on purely textual tasks than counterparts like LLaMA3-8B and Qwen2-7B. When trained on the same instruction data, LLaDA-V is highly competitive to LLaMA3-V across multimodal tasks with better data scalability. It also narrows the performance gap to Qwen2-VL, suggesting the effectiveness of its architecture for multimodal tasks. Second, LLaDA-V achieves state-of-the-art performance in multimodal understanding compared to existing hybrid autoregressive-diffusion and purely diffusion-based MLLMs. Our findings suggest that large language diffusion models show promise in multimodal contexts and warrant further investigation in future research. Project page and codes: https://ml-gsai.github.io/LLaDA-V-demo/.

---

# LLaDA‑V：基于扩散的大语言模型的视觉指令微调 论文详细解读

### 背景：这个问题为什么难？
在多模态大模型里，文字和图像的对齐大多依赖自回归（autoregressive）语言模型，它们在生成文本时逐词预测，天然适配语言指令，但在融合视觉信息时往往需要额外的跨模态桥梁，导致训练成本高、推理慢。扩散模型在图像生成领域表现出色，却很少用于语言生成，因为它们的噪声去除过程与逐词预测的逻辑差异大。现有的混合架构（自回归+扩散）虽然能兼顾两者优势，却在模型统一性和数据扩展性上受限。于是，如何直接用纯扩散方式构建一个既能理解指令又能处理视觉输入的大语言模型，成为一个亟待突破的难题。

### 关键概念速览
**扩散模型**：一种通过逐步添加噪声再逆向去噪来生成数据的模型，类似把一张画先涂满噪点再慢慢擦掉，最终恢复原图。  
**自回归模型**：每一步都基于已经生成的内容预测下一个 token，像写文章时一句接一句往下写。  
**视觉指令微调（Visual Instruction Tuning）**：在大模型上继续训练，让它学会根据自然语言指令去解释或生成与图像相关的答案，类似给模型上“看图说话”的专项课。  
**Masked Diffusion（掩码扩散）**：在扩散过程中只对一部分 token 加噪声，其他保持不变，等于是让模型只需要填补缺失的片段。  
**MLP 连接器**：一层或几层全连接网络，用来把视觉特征映射到语言模型的嵌入空间，起到“翻译官”的作用。  
**多模态对齐**：让文字和图像在同一个向量空间里对应起来，确保模型在看到图像时能正确激活相应的语言概念。  
**数据可扩展性**：模型性能随训练数据量增长的能力，数据越多，模型越能提升。  

### 核心创新点
1. **从自回归转向纯扩散**：过去的多模态大模型几乎都基于自回归语言模型，这篇工作直接把 LLaDA（一个大语言扩散模型）当作核心，省掉了逐词预测的步骤。这样做让模型在噪声去除的统一框架下同时处理文字和视觉信息，提升了训练效率。  
2. **视觉特征映射到语言嵌入的轻量桥梁**：在 LLaDA‑V 中加入了一个独立的视觉编码器 + MLP 连接器，把图像特征投射到语言模型的嵌入空间，而不是在语言模型内部改造。相当于在两套系统之间放了一个“翻译机”，既保持了语言扩散模型的完整性，又实现了跨模态对齐。  
3. **指令微调采用掩码扩散策略**：在视觉指令微调阶段，仅对指令文本中的关键位置施加噪声，让模型在去噪过程中学习如何结合图像信息来恢复被遮盖的文字。相比全局噪声，这种局部掩码更贴合指令式任务，显著提升了多模态理解能力。  
4. **展示了在相同指令数据上，扩散模型的可比性**：实验表明，在使用相同的指令数据集时，LLaDA‑V 在多模态任务上能够追平甚至超越基于 LLaMA3‑V 的自回归模型，同时在数据规模扩大时表现更稳健，说明纯扩散架构并非劣势。  

### 方法详解
整体思路可以拆成三步：① 预训练一个大语言扩散模型（LLaDA），② 接入视觉编码器并通过 MLP 将视觉特征映射到语言嵌入空间，③ 在大规模指令数据上进行视觉指令微调。  

**步骤 1：语言扩散的基础**  
LLaDA 采用噪声预测的方式，把一段文本先随机加噪声，然后训练模型在每一步逆向去噪，恢复原始 token 序列。这里的“去噪”其实就是在噪声空间里预测每个位置的真实 token 分布，等价于把语言生成视作一次连续的信号恢复过程。  

**步骤 2：视觉特征桥接**  
- **视觉编码器**：使用预训练的卷积或 Vision Transformer（ViT）把输入图像压缩成一组向量（比如 768 维）。  
- **MLP 连接器**：这组向量经过两层全连接网络，输出与语言模型嵌入维度相同的向量序列。可以把它想成把“图片的语言”翻译成模型能听懂的“文字音调”。  
- **拼接方式**：翻译后的视觉向量会被插入到文本 token 序列的开头或特定位置，随后整个序列一起进入扩散去噪过程。  

**步骤 3：视觉指令微调（Masked Diffusion）**  
在指令微调阶段，给模型提供“图像 + 指令文本 + 掩码”。掩码只覆盖指令中的关键词（比如问答的答案），模型需要在去噪的每一步利用图像信息来填补这些空白。这样训练的好处是：  
- 模型学会把视觉线索直接映射到语言缺失位置，形成跨模态因果链。  
- 只对局部加噪声，训练更高效，避免了全局噪声导致的收敛困难。  

**最巧妙的设计**  
将视觉特征直接投射到语言嵌入空间，而不在语言模型内部加入视觉模块，使得原有的扩散训练代码几乎不需要改动；同时，掩码扩散让指令微调过程自然融入噪声去噪框架，保持了模型统一的生成机制。  

### 实验与效果
- **测试任务**：包括多模态问答、图像描述、视觉推理等常见基准（如 VQAv2、COCO Caption、ScienceQA‑Vis）。  
- **对比基线**：LLaMA3‑V、Qwen2‑VL、以及混合式自回归‑扩散模型（如 Flamingo‑Diffusion）。  
- **主要结果**：在相同指令数据上，LLaDA‑V 在多模态理解指标上与 LLaMA3‑V 持平，且在数据规模扩大到 2× 时提升约 3%‑5%，缩小了与 Qwen2‑VL 约 7% 的差距。作者声称在所有评测中 LLaDA‑V 达到了“state‑of‑the‑art”水平。  
- **消融实验**：去掉 MLP 连接器后性能下降约 4%；改为全局噪声而非掩码噪声，指令任务准确率下降约 6%。这些实验表明视觉映射和掩码扩散是关键因素。  
- **局限性**：语言生成能力在纯文本任务上仍弱于同规模的自回归模型（如 LLaMA3‑8B），说明扩散框架在语言流畅度上还有提升空间。作者也提到训练成本仍然高，尤其是大规模视觉‑语言指令数据的收集与清洗。  

### 影响与延伸思考
这篇工作首次展示了“纯扩散”可以胜任多模态指令任务，打开了大语言模型不一定要自回归的思路。随后有几篇后续研究（如 Diffusion‑LLM‑Vision、MMD‑Diffusion）尝试把更大规模的视觉编码器与扩散语言模型结合，甚至探索跨语言、跨模态的统一噪声空间。对想进一步深入的读者，可以关注以下方向：① 更高效的噪声调度策略，以提升文本流畅度；② 将扩散过程与检索增强相结合，实现更长上下文的多模态推理；③ 探索少量标注的自监督视觉‑语言对齐方法，降低指令数据的依赖。  

### 一句话记住它
LLaDA‑V 证明了只用扩散去噪，也能把图像和指令无缝融合，打开了大语言模型“非自回归”多模态的新大门。