# OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models

> **Date**：2023-05-13
> **arXiv**：https://arxiv.org/abs/2305.07895

## Abstract

Large models have recently played a dominant role in natural language processing and multimodal vision-language learning. However, their effectiveness in text-related visual tasks remains relatively unexplored. In this paper, we conducted a comprehensive evaluation of Large Multimodal Models, such as GPT4V and Gemini, in various text-related visual tasks including Text Recognition, Scene Text-Centric Visual Question Answering (VQA), Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten Mathematical Expression Recognition (HMER). To facilitate the assessment of Optical Character Recognition (OCR) capabilities in Large Multimodal Models, we propose OCRBench, a comprehensive evaluation benchmark. OCRBench contains 29 datasets, making it the most comprehensive OCR evaluation benchmark available. Furthermore, our study reveals both the strengths and weaknesses of these models, particularly in handling multilingual text, handwritten text, non-semantic text, and mathematical expression recognition. Most importantly, the baseline results presented in this study could provide a foundational framework for the conception and assessment of innovative strategies targeted at enhancing zero-shot multimodal techniques. The evaluation pipeline and benchmark are available at https://github.com/Yuliang-Liu/MultimodalOCR.

---

# OCRBench：大型多模态模型中 OCR 的隐藏谜团 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理和视觉语言学习里，巨型模型（如 GPT‑4V、Gemini）已经把很多任务的上限推高，但它们在“看图识字”这类视觉文字任务上的真实能力却少有人系统评估。传统的 OCR（光学字符识别）系统往往是专门为文字检测与识别设计的，训练数据、模型结构都围绕文字展开；而多模态大模型的训练目标更宽泛，文字往往只是众多视觉信息中的一小块。于是出现了两个根本性难点：一是缺少统一、覆盖面足够大的评测基准，二是大模型在多语言、手写体、数学符号等非标准文字上的表现几乎是未知数，这让研究者难以判断它们到底能否在零样本（zero‑shot）条件下直接完成 OCR 任务。

### 关键概念速览
**大模型（Large Multimodal Model）**：指同时接受图像和文字输入、拥有数十亿甚至上百亿参数的模型，像 GPT‑4V、Gemini，能够在同一次前向传播中完成图像理解和语言生成。  
**OCR（Optical Character Recognition）**：把图片中的字符转化为机器可读文本的技术，传统上分为文字检测、文字识别两步。  
**零样本（Zero‑Shot）**：模型在没有针对特定任务进行微调的情况下，仅凭已有的通用能力直接完成任务。类似于人类第一次看到新游戏规则后直接玩得还不错。  
**场景文字视觉问答（Scene Text‑Centric VQA）**：给模型一张包含文字的自然场景图像，提问与图中文字相关的问题，模型需要先读出文字再推理答案。  
**关键信息抽取（Key Information Extraction, KIE）**：从结构化或半结构化文档（如发票、表格）中定位并提取关键字段，如金额、日期等。  
**手写数学表达式识别（Handwritten Mathematical Expression Recognition, HMER）**：识别手写的数学公式，难点在于符号的多样性和空间布局的复杂性。  
**多语言 OCR**：指模型能够识别并输出多种语言的文字，包括拉丁字母、汉字、阿拉伯文等，涉及不同字符集和排版规则。  
**基准（Benchmark）**：一套标准化的数据集和评测协议，用来统一比较不同模型的性能。

### 核心创新点
1. **从“零散评测”到“一站式基准”**：过去研究者往往挑选单个公开数据集（如 ICDAR、COCO‑Text）进行零星测试，缺乏全景视角。本文把 29 个公开数据集统一收录，覆盖文字识别、VQA、KIE、手写数学等五大任务，形成 OCRBench——目前最全的 OCR 能力评测平台。  
   *之前的做法 → 只看单任务或少量数据 → 本文的做法 → 汇聚多任务、多语言、多书写体的 29 套数据 → 让研究者一次性看到模型的全局强弱点。*

2. **统一评测管线，兼容黑盒大模型**：大模型往往只能通过 API 调用，无法直接获取内部特征。作者设计了一套“提示工程 + 后处理”的评测流程：对每个任务编写统一的 Prompt，调用模型得到文字输出，再用标准化脚本计算准确率、F1 等指标。  
   *之前的做法 → 为每个模型单独写脚本，难以复现 → 本文的做法 → 统一 Prompt 模板和后处理，所有模型走同一条评测流水线 → 消除了评测偏差，提高可比性。*

3. **细粒度错误分析框架**：在基准内部加入了多维度错误标签（语言、手写/印刷、是否数学符号、是否语义文字等），帮助定位模型的薄弱环节。  
   *之前的做法 → 只给出整体准确率 → 本文的做法 → 按上述维度拆解错误 → 让研究者快速发现模型在多语言或手写体上的系统性缺陷。*

### 方法详解
整体思路可以概括为“三步走”：**任务统一化 → Prompt 统一化 → 结果标准化**。

1. **任务统一化**  
   - 将 OCR 相关的五大任务抽象为“给图像，输出文字序列”。例如，Scene Text‑Centric VQA 被转化为“请阅读图中所有文字并回答以下问题”。KIE 则是“请列出文档中所有关键字段及其对应文字”。这样，无论底层任务多么不同，都可以用同一种输入输出形式喂给大模型。

2. **Prompt 统一化**  
   - 为每类任务设计一套固定的提示模板（Prompt），模板中明确指示模型执行 OCR 并返回纯文本。比如：  
     ```
     [Image] 请识别图中所有文字，按阅读顺序返回 JSON 列表，每项包含 "text" 和 "bbox"。
     ```  
   - 通过少量示例（few‑shot）或零示例（zero‑shot）方式，让模型知道需要返回结构化结果。这里的技巧在于“让模型自行完成文字检测”，而不是先用传统 OCR 检测框再喂给模型。

3. **结果标准化**  
   - 模型返回的文本可能带有多余空格、换行或格式错误。作者实现了一个统一的后处理脚本：① 正则清洗；② 对齐字符集（统一 Unicode 规范）；③ 计算指标（字符准确率、词级 F1、BLEU 等）。  
   - 对于 KIE、VQA 等需要匹配答案的任务，还加入了“容错匹配”，比如对数字的千位分隔符、大小写不敏感等。

**最巧妙的地方**在于：作者没有尝试改动大模型本身，而是通过精心设计的 Prompt 与后处理，让模型在“看图说话”时自然输出 OCR 结果。这种“软改造”思路极大降低了实验成本，也让评测可以直接迁移到任何黑盒模型（如 OpenAI、Google API）。

### 实验与效果
- **数据集与任务**：OCRBench 包含 29 套公开数据集，覆盖 5 大任务：Text Recognition（如 ICDAR‑2015、SVT）、Scene Text‑Centric VQA（TextVQA、ST-VQA）、Document‑Oriented VQA（DocVQA）、Key Information Extraction（SROIE、FUNSD）以及 Handwritten Mathematical Expression Recognition（CROHME）。  
- **对比基线**：作者将 GPT‑4V、Gemini 两个最新的大模型与传统 OCR 系统（如 Tesseract、PaddleOCR）以及专门的多模态模型（如 Donut、LayoutLMv3）进行对比。  
- **主要结果**：论文声称在大多数文本识别任务上，GPT‑4V 的零样本表现已经接近或略超传统 OCR 系统的有监督版本；但在手写数学表达式和多语言（尤其是非拉丁文字）上仍落后 10%~20% 的字符准确率。Gemini 在文档类 VQA 上表现稍好，KIE 任务的 F1 提升约 5%。  
- **消融实验**：通过去掉 Prompt 中的结构化指示（如去掉 JSON 要求），模型输出的文字顺序混乱，整体准确率下降约 12%；去掉后处理脚本中的数字容错匹配，KIE 任务的 F1 下降约 8%。这些实验说明 Prompt 设计和后处理是评测成功的关键因素。  
- **局限性**：作者承认评测仍受限于 API 调用成本，无法对模型内部的文字检测模块进行细粒度分析；此外，Prompt 的语言（英文）可能对非英语模型产生偏差，导致多语言评测略低估真实能力。

### 影响与延伸思考
OCRBench 通过提供统一、覆盖面广的评测平台，直接推动了“大模型+ OCR”这一交叉领域的快速发展。自论文发布后，已有多篇工作借鉴其基准进行模型改进，例如在 GPT‑4V 上加入专门的文字检测微调、或在 Gemini 中加入手写体专用的视觉前置网络。后续研究可能会围绕 **“提示工程的系统化设计”** 与 **“多语言/手写体专用的视觉适配层”** 两条路线展开。想进一步了解的读者可以关注近期在 arXiv 上出现的 “Prompt‑tuned OCR for Multimodal LLMs” 系列论文，以及开源社区对 OCRBench 的二次扩展（如加入更多低资源语言数据集）。

### 一句话记住它
OCRBench 把 29 套文字视觉任务统一成“一张图，一段文字”，让我们首次能全景、零样本地审视大模型的 OCR 能力。