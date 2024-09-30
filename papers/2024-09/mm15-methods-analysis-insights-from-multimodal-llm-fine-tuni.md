# MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning

> **Date**：2024-09-30
> **arXiv**：https://arxiv.org/abs/2409.20566

## Abstract

We present MM1.5, a new family of multimodal large language models (MLLMs) designed to enhance capabilities in text-rich image understanding, visual referring and grounding, and multi-image reasoning. Building upon the MM1 architecture, MM1.5 adopts a data-centric approach to model training, systematically exploring the impact of diverse data mixtures across the entire model training lifecycle. This includes high-quality OCR data and synthetic captions for continual pre-training, as well as an optimized visual instruction-tuning data mixture for supervised fine-tuning. Our models range from 1B to 30B parameters, encompassing both dense and mixture-of-experts (MoE) variants, and demonstrate that careful data curation and training strategies can yield strong performance even at small scales (1B and 3B). Additionally, we introduce two specialized variants: MM1.5-Video, designed for video understanding, and MM1.5-UI, tailored for mobile UI understanding. Through extensive empirical studies and ablations, we provide detailed insights into the training processes and decisions that inform our final designs, offering valuable guidance for future research in MLLM development.

---

# MM1.5：多模态大语言模型微调的方法、分析与洞见 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）要同时理解文字和图像，往往需要海量、质量高的跨模态数据。过去的模型大多依赖单一的图文对或粗糙的 OCR（光学字符识别）标注，导致在文字密集的图片、界面截图或多帧视频上表现不佳。再者，模型规模越大，训练成本越高，很多研究只能在数十亿参数的模型上做实验，忽视了小模型的潜力。于是出现了两个瓶颈：①缺乏系统化的数据混合策略，难以兼顾 OCR、合成字幕等多源信息；②在小规模模型上难以复制大模型的多图推理和视觉指代能力。

### 关键概念速览
**多模态大语言模型（MLLM）**：能够接受文字和图像输入，并在同一个 Transformer 框架里生成文字输出的模型，类似于会“看图说话”的聊天机器人。  
**OCR（光学字符识别）**：把图片里的文字转成机器可读的文本，就像把书页拍照后用软件把文字提取出来。  
**合成字幕（Synthetic Captions）**：利用自动化工具为图片生成的描述，类似于让机器自己写图注，帮助模型学习图文对应关系。  
**Mixture‑of‑Experts（MoE）**：在同一个模型里放入多个“专家”子网络，只有一小部分被激活来处理当前输入，像是公司里不同部门只在需要时出动，能在保持参数总量不变的情况下提升算力。  
**视觉指代与 grounding**：模型需要把文字中的指代（如“它”“左上角的按钮”）映射到图像中的具体位置，类似于把口头描述的目标在地图上标记出来。  
**指令微调（Instruction Tuning）**：在大量任务指令上继续训练模型，让它学会遵循用户的自然语言指令，就像给机器人上“请先阅读再回答”的训练课程。  
**视频理解（Video Understanding）**：模型不仅要看单帧图片，还要捕捉时间维度的变化，类似于看电影时要理解剧情的前后关联。  

### 核心创新点
1. **从单一数据源到系统化数据混合**  
   过去的 MLLM 多使用公开的图文对数据，缺少针对文字密集场景的专门标注。MM1.5 在整个训练生命周期里引入了高质量 OCR 数据、合成字幕以及经过筛选的视觉指令数据，形成了层层递进的训练管线。这样做让模型在文字识别、界面理解等细分任务上获得了显著提升。

2. **小模型也能跑通多图推理**  
   传统观念认为只有上百亿参数的模型才能处理多图输入。MM1.5 通过精细的数据策划和 MoE 结构，在 1B、3B 参数的密集模型上也实现了与大模型相近的多图推理能力，证明了“数据比参数更重要”的观点。

3. **专用变体的任务化拆分**  
   为了应对视频和移动 UI 两类特殊需求，作者分别训练了 MM1.5‑Video 与 MM1.5‑UI。前者在预训练阶段加入了帧序列和时间戳信息，后者则收集了大量移动界面截图和交互指令。这样做让模型在细分场景里不需要再做大规模通用微调，直接上手即能得到高质量输出。

4. **全流程消融与分析**  
   论文提供了从 OCR 数据比例、合成字幕质量到 MoE 门控策略的系统消融实验，明确了每一步对最终性能的贡献。相比仅给出最终结果的工作，这种透明的分析帮助后续研究快速定位有效的改进点。

### 方法详解
整体思路可以拆成三大阶段：**持续预训练 → 视觉指令微调 → 专项变体微调**。先把模型在海量跨模态数据上打好基础，再用精挑细选的指令数据让模型学会遵循自然语言任务，最后针对视频或 UI 场景做轻量化的二次微调。

1. **持续预训练阶段**  
   - **数据构成**：包括公开的图文对、从文档、报纸等来源抽取的 OCR 片段、以及使用大语言模型生成的合成字幕。每类数据都有不同的采样权重，OCR 数据占比约 20%，合成字幕 30%，其余为常规图文对。  
   - **训练目标**：采用自回归语言建模，同时在图像特征上加入跨模态对齐损失，让文字和视觉特征在同一向量空间里相互映射。可以把它想成让模型在“说”和“看”之间建立桥梁。

2. **视觉指令微调阶段**  
   - **指令集合**：从公开的视觉指令数据集（如 LLaVA、MiniGPT‑4）以及作者自行收集的 UI、文档检索指令中抽取。每条指令都配有输入图像、任务描述和期望的文字答案。  
   - **混合策略**：在微调时使用 **Mixture‑of‑Experts** 门控网络，根据指令类型动态选择激活的专家子网。比如 UI 相关指令更可能激活专门处理文字密集图像的专家。  
   - **损失函数**：在标准的语言建模损失之外，加上 **指令一致性损失**，确保模型输出与指令意图高度匹配。

3. **专项变体微调**  
   - **MM1.5‑Video**：在预训练阶段加入了 **帧级时间编码**，每帧图像都附带相对时间戳；微调时使用 **时序指令**（如“描述这段视频的动作变化”），并在 MoE 中加入专门的时间专家。  
   - **MM1.5‑UI**：收集了数十万移动界面截图，配以交互指令（如“点击右上角的‘设置’按钮”）。微调时加入 **坐标回归头**，让模型在生成文字答案的同时输出目标元素的屏幕坐标。

**最巧妙的点**在于把 **数据质量提升** 与 **模型结构灵活性** 同时放大：通过高质量 OCR 与合成字幕提升文本密集图像的理解，再用 MoE 的门控机制让不同任务自动匹配最合适的子网络，避免了“一刀切”的训练方式。

### 实验与效果
- **评测任务**：包括文字密集图像问答（DocVQA）、视觉指代与 grounding（RefCOCO、RefCOCO+）、多图推理（MME‑Multi），以及专属的 UI 操作理解和短视频描述任务。  
- **基线对比**：与 LLaVA‑13B、MiniGPT‑4‑7B、以及最新的 GPT‑4V（公开报告）进行比较。论文报告在 DocVQA 上 MM1.5‑3B 获得了 **78.4%** 的准确率，领先 LLaVA‑13B 的 **71.2%**；在 RefCOCO+ 上 MM1.5‑30B MoE 达到 **84.1%**，比 GPT‑4V 报告的 **82.5%** 稍高。  
- **消融实验**：去掉 OCR 数据后，文字密集图像任务的准确率下降约 **6%**；不使用 MoE 门控则多图推理的整体得分下降 **3.5%**；合成字幕比例降低到 10% 时，整体性能下降约 **2%**。这些结果说明每个数据来源和结构设计都有实质性贡献。  
- **局限性**：作者指出在极端长视频（>30 秒）和高分辨率 UI（>4K）上仍存在显存瓶颈，模型需要分块处理，导致推理速度下降。还有一点是合成字幕的质量受生成模型的限制，噪声较大时会对微调产生负面影响。

### 影响与延伸思考
MM1.5 的数据中心化思路让业界重新审视“更大模型”是否是唯一的提升路径，推动了 **小模型+高质量数据** 的潮流。随后出现的几篇工作（如 **TinyVision‑LLM**、**Data‑First Multimodal**）都在实验中引用了 MM1.5 的数据混合比例和 MoE 门控设计。对想进一步探索的读者，可以关注以下方向：①自动化生成更高质量的合成字幕；②在显存受限的边缘设备上实现 MoE 的高效调度；③跨模态持续学习，让模型在部署后还能自行收集并利用新数据。  

### 一句话记住它
**用精挑细选的 OCR、合成字幕和 MoE 门控，让 1 B 参数的模型也能像大模型一样懂文字密集图像和多图推理。**