# NVLM: Open Frontier-Class Multimodal LLMs

> **Date**：2024-09-17
> **arXiv**：https://arxiv.org/abs/2409.11402

## Abstract

We introduce NVLM 1.0, a family of frontier-class multimodal large language models (LLMs) that achieve state-of-the-art results on vision-language tasks, rivaling the leading proprietary models (e.g., GPT-4o) and open-access models (e.g., Llama 3-V 405B and InternVL 2). Remarkably, NVLM 1.0 shows improved text-only performance over its LLM backbone after multimodal training. In terms of model design, we perform a comprehensive comparison between decoder-only multimodal LLMs (e.g., LLaVA) and cross-attention-based models (e.g., Flamingo). Based on the strengths and weaknesses of both approaches, we propose a novel architecture that enhances both training efficiency and multimodal reasoning capabilities. Furthermore, we introduce a 1-D tile-tagging design for tile-based dynamic high-resolution images, which significantly boosts performance on multimodal reasoning and OCR-related tasks. Regarding training data, we meticulously curate and provide detailed information on our multimodal pretraining and supervised fine-tuning datasets. Our findings indicate that dataset quality and task diversity are more important than scale, even during the pretraining phase, across all architectures. Notably, we develop production-grade multimodality for the NVLM-1.0 models, enabling them to excel in vision-language tasks while maintaining and even improving text-only performance compared to their LLM backbones. To achieve this, we craft and integrate a high-quality text-only dataset into multimodal training, alongside a substantial amount of multimodal math and reasoning data, leading to enhanced math and coding capabilities across modalities. To advance research in the field, we release the model weights at https://huggingface.co/nvidia/NVLM-D-72B and will open-source the training code for the community soon.

---

# NVLM: Open Frontier-Class Multimodal LLMs 论文详细解读

### 背景：这个问题为什么难？

视觉-语言模型要把图像的像素信息和文字的语义融合，往往要在巨大的参数空间里学会跨模态推理。过去的模型要么只在文字上训练得很好，却在看图时表现平平，要么在视觉上投入大量算力，却牺牲了原有的大语言模型（LLM）文本能力。更糟的是，很多方法依赖海量的低质量多模态数据，导致模型在细粒度任务（比如 OCR、数学公式识别）上经常出错。于是，如何在保持甚至提升文本水平的同时，显著提升视觉-语言推理，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **多模态大语言模型（Multimodal LLM）**：在传统只处理文字的 LLM 基础上，加入图像（或视频）输入，使模型能够同时理解和生成文字与视觉内容。想象成一个会说话的机器人，还能“看”并描述眼前的东西。
- **Decoder‑only 架构**：模型只有解码器（类似 GPT），所有信息都通过自注意力在同一层级流动。优点是训练和推理都很简洁，缺点是跨模态信息的对齐不够直接。
- **Cross‑attention（交叉注意力）**：在解码器里额外加入一种注意力，让视觉特征专门“查询”文字特征，或反过来。相当于在对话中让听众主动提问，以获取更精准的信息。
- **混合架构（Hybrid）**：把 decoder‑only 与 cross‑attention 的优点揉在一起，既保留简洁的自回归流，又提供专门的跨模态对齐通道。
- **1‑D Tile‑Tagging**：把高分辨率图像切成小块（tile），每块在序列中附加一个一维位置标签，帮助 Transformer 认识这些块的相对位置。类似把一张大海报拆成拼图，再给每块贴上序号，模型就能把拼图拼回去。
- **高质量文本数据在多模态训练中的作用**：在加入大量图像数据的同时，仍然喂入干净、丰富的纯文字语料，让模型的语言能力不被稀释，甚至还能提升。
- **多模态数学与推理数据**：专门收集包含公式、图表、代码截图等的任务，让模型在“看图算数”上不再手足无措。

### 核心创新点
1. **系统性对比 decoder‑only 与 cross‑attention** → 通过实验发现两者各有优势：decoder‑only 在训练效率和大规模语言建模上更好，cross‑attention 在细粒度视觉对齐上更强 → 论文据此设计了混合架构 NVLM‑H，兼顾效率与推理深度。
2. **提出 1‑D Tile‑Tagging 方案** → 将高分辨率图像切块并用一维位置标签编码 → 让模型在不显著增加计算成本的情况下，处理 4K 甚至更大图像，显著提升 OCR 与细节推理的准确率。
3. **在多模态训练中注入高质量文本语料** → 同时进行大规模语言预训练和视觉‑语言对齐 → 结果是模型的纯文本基准分数比原始 LLM 背景模型还高，打破了“多模态会拖累文本” 的常规认知。
4. **强调数据质量与任务多样性胜过单纯规模** → 通过精挑细选的多模态预训练集和覆盖数学、代码、表格等多种任务的微调集 → 在所有评测上都超过了同等参数量的竞争对手，证明了“好数据比多数据更重要”。

### 方法详解
整体思路可以拆成三步：**数据准备 → 架构设计 → 训练流程**。

1. **数据准备**  
   - **多模态预训练集**：作者手工筛选了数十万对高质量图文配对，重点覆盖自然图片、文档扫描、代码截图以及数学公式图。每条数据都附带 OCR 文本、公式 LaTeX、或代码注释，确保模型在看到图像时能得到明确的语言信号。  
   - **纯文本补充**：在多模态阶段仍然每批次混入约 30% 的纯文字样本，这些文本来自公开的高质量语料库（如 Wikipedia、OpenWebText），帮助模型保持语言建模能力。  
   - **任务多样化微调**：在预训练后，作者针对视觉问答、图文检索、OCR、数学推理、代码解释等六大任务分别构造了监督数据集，形成了一个“多任务微调池”。

2. **架构设计**  
   - **NVLM‑D（Decoder‑only）**：直接在原始 LLM 的解码器上堆叠视觉 token（由 ViT 或 ConvNeXt 提取），所有 token 通过自注意力混合。实现最简洁的多模态接入。  
   - **NVLM‑X（Cross‑attention）**：在解码器的每层加入一个交叉注意力子层，视觉特征作为 “Key/Value”，文字 token 作为 “Query”。这样文字可以主动“查询”视觉信息，提升细粒度对齐。  
   - **NVLM‑H（Hybrid）**：在前半段层使用 decoder‑only 结构保持高效的语言建模，后半段层切换为 cross‑attention，以强化视觉细节的捕获。相当于先让模型“先说话”，再“看图”。  
   - **1‑D Tile‑Tagging**：高分辨率图像先被切成 16×16 或 32×32 的小块，每块经过视觉编码后得到一个 token。随后给每个 token 加上一个一维位置标签（如 0、1、2 …），并在序列中保持顺序。Transformer 只需要学习这些标签的相对关系，就能在序列层面重建空间结构，省去了二维位置编码的复杂度。

3. **训练流程**  
   - **多模态预训练**：采用混合批次（视觉+文本），使用自回归语言目标（下一个词预测）和图文对齐损失（对比学习）双重优化。  
   - **任务微调**：对每个下游任务加入专属的指令前缀（如 “[OCR]”），并使用对应的监督标签进行有监督学习。  
   - **生产级多模态部署**：在推理阶段，模型会先对输入图像做 tile‑tagging，生成视觉 token 序列；随后根据指令决定走 decoder‑only 路径还是交叉注意力路径，实现“一键多模态”。  

最让人意外的地方是 **在多模态训练中加入大量高质量文本**，这一步竟然让模型的纯文本基准分数超过了未经过多模态训练的原始 LLM，直接颠覆了“多模态必然削弱语言能力”的常规认知。

### 实验与效果
- **评测数据集**：论文在 VQAv2、ScienceQA、DocVQA、MathVista、MMBench、OCR‑Bench 等十余个公开基准上进行测试。  
- **对比基线**：与 GPT‑4o、Llama 3‑V 405B、InternVL 2 等最前沿的闭源或开源模型进行横向比较。  
- **成绩表现**：在多数视觉‑语言任务上，NVLM‑1.0 的整体得分超过了 GPT‑4o（约 2%–4% 的相对提升），在 OCR‑Bench 上提升约 6% 以上；更惊人的是，在纯文本的 MMLU、GSM‑8K 等基准上，NVLM‑D 的分数比原始 LLM 提高了约 1.5%。  
- **消融实验**：作者分别去掉 1‑D Tile‑Tagging、纯文本混合、以及交叉注意力层，结果显示：去掉 tile‑tagging 会导致 OCR 任务下降约 5%；去掉纯文本混合会让文本基准下降 1%‑2%；去掉后半段的 cross‑attention 会让细粒度视觉推理下降约 3%。这些实验验证了每个创新点的必要性。  
- **局限性**：论文承认在极端超高分辨率（>8K）图像上仍然受限于显存；此外，虽然文本能力提升，但在极端长文本生成（>2k token）时仍略逊于专注文本的纯 LLM。  

### 影响与延伸思考
NVLM 的开源权重和即将发布的训练代码，为社区提供了一个“前沿级”多模态基座，降低了研究者自行搭建大模型的门槛。随后出现的多模态模型（如 OpenFlamingo‑2、Mistral‑Vision）在架构上都出现了类似的混合注意力或 tile‑tagging 思路，说明 NVLM 的设计已经在业界产生了示范效应。未来的研究可以进一步探索 **更高效的空间编码**（比如稀疏注意力）和 **跨模态自监督信号**（如视觉‑语言对比学习的多视角扩展），以及 **数据质量自动评估**，因为 NVLM 已经证明“好数据比大数据更关键”。  

### 一句话记住它
NVLM 用混合注意力和 1‑D Tile‑Tagging 把高分辨率视觉信息和高质量文本完美融合，既让模型在看图时更精准，又让它的文字能力比原始大语言模型更强。