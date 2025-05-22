# LaViDa: A Large Diffusion Language Model for Multimodal Understanding

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.16839

## Abstract

Modern Vision-Language Models (VLMs) can solve a wide range of tasks requiring visual reasoning. In real-world scenarios, desirable properties for VLMs include fast inference and controllable generation (e.g., constraining outputs to adhere to a desired format). However, existing autoregressive (AR) VLMs like LLaVA struggle in these aspects. Discrete diffusion models (DMs) offer a promising alternative, enabling parallel decoding for faster inference and bidirectional context for controllable generation through text-infilling. While effective in language-only settings, DMs' potential for multimodal tasks is underexplored. We introduce LaViDa, a family of VLMs built on DMs. We build LaViDa by equipping DMs with a vision encoder and jointly fine-tune the combined parts for multimodal instruction following. To address challenges encountered, LaViDa incorporates novel techniques such as complementary masking for effective training, prefix KV cache for efficient inference, and timestep shifting for high-quality sampling. Experiments show that LaViDa achieves competitive or superior performance to AR VLMs on multi-modal benchmarks such as MMMU, while offering unique advantages of DMs, including flexible speed-quality tradeoff, controllability, and bidirectional reasoning. On COCO captioning, LaViDa surpasses Open-LLaVa-Next-8B by +4.1 CIDEr with 1.92x speedup. On bidirectional tasks, it achieves +59% improvement on Constrained Poem Completion. These results demonstrate LaViDa as a strong alternative to AR VLMs. Code and models will be released in the camera-ready version.

---

# LaViDa：用于多模态理解的大规模扩散语言模型 论文详细解读

### 背景：这个问题为什么难？

视觉‑语言模型（VLM）需要同时理解图像和文字，并在此基础上完成推理、描述或生成任务。传统的自回归（AR）VLM（如 LLaVA）在生成时只能一步步往后写，导致推理速度受限，而且生成过程只能从左到右，难以在中间插入约束或修改已有内容。实际应用常常需要快速响应（比如实时对话）和可控输出（比如要求答案符合特定格式），而 AR 结构天生缺乏并行解码和双向上下文的能力。于是，如何在保持多模态理解水平的同时，实现更快、更灵活的生成，成为了亟待突破的瓶颈。

### 关键概念速览
- **自回归模型（AR）**：每一步只能依据已经生成的前缀来预测下一个词，就像只能从左往右读书写句子，速度慢且难以回头改动。  
- **离散扩散模型（DM）**：先把完整的目标序列随机破坏成噪声，再一步步“去噪”恢复，整个过程可以并行处理，类似先把完整画作涂满噪点，再一次性擦掉噪点恢复原图。  
- **文本填充（text‑infilling）**：在已有上下文的两端提供信息，让模型在中间补全缺失的文字，等价于给出句子开头和结尾，让模型填空。  
- **视觉编码器**：把图片转成向量序列的网络，常用的有 CLIP、ViT 等，作用类似把照片翻译成机器能读的“文字”。  
- **KV 缓存（Key‑Value cache）**：在 Transformer 中保存已经计算好的注意力键值对，以免重复计算，类似把已经查好的字典条目存起来，下次直接用。  
- **时间步移位（timestep shifting）**：在采样时把噪声的时间步稍微提前或推后，以提升生成质量，像在烤箱里调节温度让食物更酥。  
- **互补遮蔽（complementary masking）**：训练时对输入进行两种互补的遮挡，让模型学会从不同角度恢复信息，类似给学生出两套不同的填空题，帮助其全面掌握句子结构。

### 核心创新点
1. **把离散扩散模型搬进多模态指令跟随**  
   - 之前的扩散模型主要用于纯文本或图像生成，缺少跨模态指令能力。  
   - 本文在扩散模型的解码器前接入视觉编码器，并在大规模多模态指令数据上联合微调，使模型既能看图也能听指令。  
   - 结果是模型在多模态基准（如 MMMU）上达到或超过 AR VLM 的水平，同时保留了扩散模型的并行解码优势。

2. **互补遮蔽提升训练效率**  
   - 直接在扩散框架下训练会出现信息缺失导致的收敛慢。  
   - 作者设计了两套互补的遮蔽策略：一种遮掉文本，另一种遮掉视觉特征，使模型在每一步都能从另一模态获得线索。  
   - 这种“相互补位”的方式显著加速了模型学习，尤其在需要跨模态推理的任务上表现更稳。

3. **前缀 KV 缓存实现高效推理**  
   - 扩散解码虽然可以并行，但每一步仍需对视觉特征做全量注意力，导致显存占用大。  
   - 通过在解码的前几步预先计算并缓存注意力键值对，后续步骤直接复用，类似把前菜的配料提前准备好。  
   - 该技巧让 LaViDa 在 COCO 图像描述任务上比同等规模的 Open‑LLaVA‑Next‑8B 快 1.92 倍。

4. **时间步移位提升生成质量**  
   - 直接使用标准扩散采样会出现细节缺失或不符合约束的情况。  
   - 作者在采样时把噪声的时间步稍微向前或向后平移，使模型在关键步骤拥有更丰富的上下文。  
   - 在受约束的诗歌补全任务上，这一技巧带来了约 59% 的性能提升，证明了双向推理的潜力。

### 方法详解
**整体框架**  
LaViDa 由三大块组成：视觉编码器、离散扩散解码器、以及指令对齐层。训练时，模型接受一张图片、一段指令（可能包含填空占位）以及对应的目标文本。首先，视觉编码器把图片映射成一系列向量；随后，这些向量与指令的文字嵌入一起进入扩散解码器。解码器在多个离散时间步上执行“去噪”操作，最终输出完整的文本答案。

**关键步骤拆解**  

1. **视觉特征提取**  
   - 使用预训练的 CLIP/Vit 大模型，将图片转成 $V = \{v_1,…,v_N\}$。  
   - 这些向量在后续的注意力层中充当“键值”，帮助文本去噪。

2. **离散噪声注入与互补遮蔽**  
   - 将目标文本 $T$ 先映射成离散 token 序列。  
   - 在每个训练时间步 $t$，随机把一部分 token 替换为噪声 token（离散噪声），并根据两套遮蔽策略分别遮掉文本或视觉特征。  
   - 互补遮蔽确保模型在任何一步都有至少一种模态提供完整信息，从而学习跨模态恢复。

3. **去噪解码（扩散逆过程）**  
   - 解码器是基于 Transformer 的自注意力网络。  
   - 在每个时间步，模型接受当前噪声序列、视觉特征以及指令的前缀，输出对噪声的修正建议。  
   - 通过多步迭代，噪声逐渐被消除，最终得到干净的文本。

4. **前缀 KV 缓存**  
   - 前 $k$ 步（如 2 步）完成后，所有注意力键值对被存入缓存。  
   - 后续步骤直接读取缓存，省去重复计算视觉‑文本注意力的开销。

5. **时间步移位采样**  
   - 推理时，作者在标准时间步序列上加入微小的偏移（例如把 $t$ 改为 $t+Δ$），让模型在关键位置拥有更长的上下文窗口。  
   - 这一步在不增加额外计算的情况下提升了文本的连贯性和约束符合度。

**最巧妙的设计**  
互补遮蔽把视觉和文本信息交叉“备份”，让模型在噪声极大时仍能靠另一模态恢复；而前缀 KV 缓存把本来需要每步重复的注意力计算一次性搞定，直接把扩散模型的并行优势转化为实际加速。

### 实验与效果
- **评测数据集**：MMMU（多模态理解大盘点）、COCO Caption（图像描述）、Constrained Poem Completion（受约束的诗歌补全）等。  
- **整体表现**：在 MMMU 上 LaViDa 与最强的 AR VLM 持平或略有超越，证明了跨模态指令跟随的可行性。  
- **速度与质量**：在 COCO Caption 上，LaViDa 的 CIDEr 分数比 Open‑LLaVA‑Next‑8B 高出 4.1，推理速度提升约 1.92 倍。  
- **可控生成**：在受约束的诗歌补全任务中，利用文本填充和时间步移位，性能提升约 59%，显示出双向推理的优势。  
- **消融实验**：论文分别去掉互补遮蔽、KV 缓存和时间步移位，发现每项技术都对最终分数有显著贡献，尤其是互补遮蔽对跨模态恢复最关键。  
- **局限性**：作者提到 LaViDa 仍然依赖大规模预训练的视觉编码器，若视觉特征质量不佳会直接拖累整体表现；此外，离散扩散的采样步骤仍比单步 AR 稍慢，实际加速幅度受硬件并行度限制。

### 影响与延伸思考
LaViDa 把离散扩散模型成功搬进多模态指令跟随，打开了“并行+双向”在视觉‑语言领域的可能性。后续工作已经开始探索更细粒度的跨模态噪声注入、基于连续扩散的视觉‑语言生成，以及把扩散框架与大规模检索结合以提升长文本一致性。对想进一步研究的读者，可以关注以下方向：  
1. **跨模态噪声建模**：如何让噪声本身也携带视觉信息，从而在去噪时自然融合图像特征。  
2. **高效采样策略**：利用自适应步数或混合 AR‑DM 解码，进一步压缩推理时间。  
3. **可控多模态生成**：结合结构化约束（如表格、代码）与扩散填空，实现更精准的输出格式控制。  
这些方向都有望在下一代 VLM 中实现更快、更灵活的交互。

### 一句话记住它
LaViDa 用离散扩散把多模态理解变成并行、双向的填空游戏，既快又能精准控制输出。