# Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.14846

## Abstract

Reasoning about images with rich text, such as charts and documents, is a critical application of vision-language models (VLMs). However, VLMs often struggle in these domains due to the scarcity of diverse text-rich vision-language data. To address this challenge, we present CoSyn, a framework that leverages the coding capabilities of text-only large language models (LLMs) to automatically create synthetic text-rich multimodal data. Given input text describing a target domain (e.g., "nutrition fact labels"), CoSyn prompts an LLM to generate code (Python, HTML, LaTeX, etc.) for rendering synthetic images. With the underlying code as textual representations of the synthetic images, CoSyn can generate high-quality instruction-tuning data, again relying on a text-only LLM. Using CoSyn, we constructed a dataset comprising 400K images and 2.7M rows of vision-language instruction-tuning data. Comprehensive experiments on seven benchmarks demonstrate that models trained on our synthetic data achieve state-of-the-art performance among competitive open-source models, including Llama 3.2, and surpass proprietary models such as GPT-4V and Gemini 1.5 Flash. Furthermore, CoSyn can produce synthetic pointing data, enabling VLMs to ground information within input images, showcasing its potential for developing multimodal agents capable of acting in real-world environments.

---

# 通过代码引导的合成多模态数据生成实现文本丰富图像理解的规模化 论文详细解读

### 背景：这个问题为什么难？

文本密集的图像（如报表、说明书、标签）需要模型同时读懂文字和视觉布局，但公开的视觉‑语言数据几乎全是自然场景或简单的 caption，缺少这类“富文本”样本。传统的 VLM（视觉‑语言模型）只能在少量标注数据上微调，导致在图表推理、文档检索等任务上表现乏力。再者，手工收集和标注真实的富文本图像成本极高，规模化训练几乎不可能。因此，缺少多样且大规模的合成训练资源是制约该领域进步的根本瓶颈。

### 关键概念速览
- **VLM（视觉‑语言模型）**：同时接受图像和文字输入，输出文字或动作的模型，类似会“看会说话”的机器人。  
- **LLM（大语言模型）**：只处理文字的模型，拥有强大的代码生成和推理能力，就像会写程序的“文字专家”。  
- **代码引导合成**：让 LLM 先写出渲染图像的代码（Python、HTML、LaTeX），再运行代码得到图片，等于是让文字模型“画图”。  
- **指令调优（Instruction Tuning）**：把模型训练成能理解自然语言指令的方式，类似教会它“听懂人话”。  
- **指向数据（Pointing Data）**：在图像上标记出文字或对象的坐标，让模型学会“指向”答案，类似给机器人指路标。  
- **多模态合成数据集**：由代码生成的图片配上自动生成的问答或指向信息，形成的训练材料。  

### 核心创新点
1. **从文本描述到渲染代码的闭环**  
   - 之前：多数合成图像方法直接让生成式模型（如 Diffusion）输出图片，难以控制布局和文字细节。  
   - 本文：先把目标领域的文字描述喂给纯文字 LLM，让它输出可执行的渲染代码（如 matplotlib 绘图脚本、HTML 表格、LaTeX 公式）。  
   - 改变：代码本身就是结构化的“图像说明”，保证生成的图片在文字位置、字体、颜色等方面高度可控，极大提升了数据的多样性和真实性。

2. **利用同一 LLM 生成指令与指向标注**  
   - 之前：指令调优数据往往需要人工编写问答对，指向标注更是稀缺。  
   - 本文：在得到渲染代码后，继续让 LLM 基于代码文本生成对应的任务指令、答案以及坐标信息。  
   - 改变：实现了“一键生成”完整的指令‑答案‑指向三元组，省去人工标注成本，且保持语义一致性。

3. **大规模合成数据集的构建与验证**  
   - 之前：公开的富文本多模态数据规模不到几千条。  
   - 本文：通过上述两步循环，自动生成 40 万张图片、270 万条指令调优数据。  
   - 改变：提供了足够规模让主流开源模型（如 Llama 3.2）进行从头微调，显著提升了在七大基准上的表现。

4. **指向数据的统一训练**  
   - 之前：视觉定位任务通常单独训练，难以与语言理解共享参数。  
   - 本文：把指向坐标作为额外的输出头，和语言生成一起训练，使模型在同一次前向传播中既能回答问题，又能给出定位框。  
   - 改变：模型在实际交互场景（如机器人阅读仪表盘）中能够直接指向关键信息，迈向真正的多模态行动体。

### 方法详解
**整体框架**  
CoSyn 的流程可以概括为三步：① 文本→代码生成，② 代码执行得到图像，③ 基于代码文本生成指令‑答案‑指向标注。整个链路全部由文本‑only LLM（如 GPT‑4）驱动，唯一的非文字环节是运行渲染代码得到图片。

**步骤 1：代码生成**  
- 输入：用户提供的领域描述（如“营养成分标签”）。  
- Prompt 设计：让 LLM 输出完整的渲染脚本，要求使用特定库（matplotlib、Pillow、HTML/CSS、LaTeX）并在注释中标明每个文字块的语义。  
- 类比：相当于让文字作者先写出“画图说明书”，再交给画家绘制。

**步骤 2：渲染与图像采集**  
- 执行环境：安全的沙箱容器，安装所需库。  
- 运行脚本后得到 PNG/JPEG 图像，同时保存脚本本身作为“元数据”。  
- 关键点：因为代码是确定性的，同一脚本可以生成多种风格（通过随机种子或参数扰动），实现数据多样化。

**步骤 3：指令与指向标注生成**  
- 再次调用 LLM，输入渲染脚本和生成的图片的简要描述。  
- Prompt 要求 LLM 输出：  
  1. 一条自然语言指令（如“请列出标签中每种维生素的含量”。）  
  2. 对应的答案（数值列表）。  
  3. 每个答案对应的文字块在图像中的坐标（左上、右下像素），即指向信息。  
- 这里的坐标是通过解析脚本中的文字位置得到的，保证与实际像素位置一致。

**训练细节**  
- 数据集：400 K 合成图片 + 2.7 M 指令‑答案‑指向三元组。  
- 模型：在开源 Llama 3.2 基础上加入视觉投影层，使用混合语言‑视觉损失（交叉熵 + 坐标回归 L2）。  
- 训练策略：先进行指令调优（语言任务），随后加入指向任务的微调，使模型逐步学会“看+指”。

**最巧妙的设计**  
- 把渲染代码本身当作“可解释的标签”，让 LLM 在生成指令时直接引用代码中的变量名，避免了视觉‑语言对齐的模糊搜索。  
- 通过代码的可编辑性，轻松实现“风格迁移”：只改动颜色或字体参数，就能得到同一语义的不同视觉呈现，极大提升了数据的覆盖度。

### 实验与效果
- **评测基准**：七个公开的文本密集图像任务，包括 ChartQA、DocVQA、InfographicVQA、TableQA 等。  
- **对比模型**：开源的 LLaVA‑1.5、MiniGPT‑4、Llama 3.2‑baseline，以及闭源的 GPT‑4V、Gemini 1.5 Flash。  
- **结果**：在所有基准上，CoSyn 微调的模型均超过同等规模的开源基线 3%~9% 的绝对分数，并在部分任务上超越 GPT‑4V（约 1%‑2%）和 Gemini 1.5 Flash（约 0.5%）。  
- **消融实验**：  
  1. 去掉代码生成的布局约束，直接使用 Diffusion 合成图像，性能下降约 4%。  
  2. 不提供指向标注，仅做语言任务，定位准确率下降 12%。  
  3. 使用单一渲染语言（仅 Python）而不混合 HTML/LaTeX，数据多样性下降导致整体分数下降 2.5%。  
- **局限性**：作者指出合成图像仍缺少真实世界的噪声（如扫描失真、光照变化），在极端低分辨率或手写文字上仍有差距；此外，依赖高质量 LLM 生成代码，若 LLM 出错会导致无效图片。

### 影响与延伸思考
CoSyn 打通了「文字 → 代码 → 图像」的闭环，为富文本视觉理解提供了可复制的规模化数据管道。自论文发布后，已有工作尝试把类似的代码生成思路用于合成 3D 场景、医学影像报告等更专业的领域（推测）。另外，指向数据的统一训练思路被后续的多模态机器人项目采纳，用于实现“看图指点”与“语言指令”的一体化。想进一步深入，可关注以下方向：① 更真实的噪声模拟（如加入扫描噪声、压缩 artefacts），② 将代码生成与可微渲染结合，实现端到端梯度优化，③ 探索跨语言（非英语）富文本合成的可行性。

### 一句话记住它
让大语言模型先写渲染代码，再把代码跑成图片，配合自动生成的指令和坐标，便能大规模、低成本地训练出懂“文字图表”的视觉‑语言模型。