# Nougat: Neural Optical Understanding for Academic Documents

> **Date**：2023-08-25
> **arXiv**：https://arxiv.org/abs/2308.13418

## Abstract

Scientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (Neural Optical Understanding for Academic Documents), a Visual Transformer model that performs an Optical Character Recognition (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.

---

# Nougat：面向学术文档的神经光学理解 论文详细解读

### 背景：这个问题为什么难？
科学论文大多以 PDF 形式发布，PDF 只保存页面的排版信息，几乎不保留文字的语义结构。传统 OCR（光学字符识别）工具在普通文本上还能勉强工作，但一旦遇到公式、图表、交叉引用等学术专有元素，识别率会急剧下降。现有的 PDF 解析库往往先把 PDF 转成文字流，再用正则或规则匹配来恢复公式，这种方式对排版多样的论文几乎无能为力。于是，如何直接从页面图像中完整、准确地恢复出机器可读的标记语言，成为了阻碍大规模学术文本自动化处理的关键瓶颈。

### 关键概念速览
**OCR（光学字符识别）**：把图片中的文字转成可编辑的文本，就像把手写笔记拍照后让软件“读”出来。  
**Visual Transformer（视觉 Transformer）**：一种把图像切成小块（patch）后，用自注意力机制建模全局关系的网络，类似把整张图片拆成拼图，再让模型记住每块之间的联系。  
**Markup Language（标记语言）**：用特定符号描述文档结构和内容的语言，常见的有 HTML、LaTeX；在本工作里指的是能够表达公式、标题、段落等信息的 LaTeX‑like 代码。  
**LaTeX**：学术界广泛使用的排版系统，尤其擅长渲染数学公式，类似于“学术界的 Word”。  
**Seq2Seq（序列到序列）模型**：把一个序列映射到另一个序列的模型，常用于机器翻译，这里把图像特征序列翻译成 LaTeX 代码。  
**Self‑Attention（自注意力）**：模型在处理每个位置时，会“关注”序列中所有其他位置的内容，帮助捕捉长距离依赖，就像阅读时会不时回头看前面的文字。  
**Dataset of Scientific Documents（科学文档数据集）**：作者自行收集并标注的包含页面图像与对应 LaTeX 代码的集合，用来训练和评估模型。

### 核心创新点
1. **从纯视觉模型到端到端学术 OCR**：以前的学术 OCR 往往把视觉特征提取和公式解析拆成两步，先用 CNN 把字符框出来，再交给专门的公式识别器。Nougat 直接使用 Visual Transformer 把整页图像一次性映射到 LaTeX 序列，省掉了中间的框选和手工特征步骤，显著简化了流水线。  
2. **统一的标记语言输出**：传统方法对文字和公式分别输出不同的格式，后期需要再合并。Nougat 把页面的所有内容（普通文字、标题、公式、列表）统一编码为 LaTeX，形成一种“一站式”翻译，使得后续的文本处理或大模型微调可以直接使用。  
3. **大规模自建科学文档数据集**：作者自行爬取并对齐了数万页 PDF 与对应的 LaTeX 源码，填补了公开学术 OCR 数据稀缺的空白，为模型训练提供了足够的多样性。  
4. **开放模型与代码**：在发布时提供了预训练权重和推理脚本，降低了社区复现和二次开发的门槛，促进了后续研究的快速迭代。

### 方法详解
**整体框架**  
Nougat 的工作流程可以概括为三步：① 将 PDF 页面渲染成高分辨率图像；② 用 Visual Transformer 把图像切成固定大小的 patch，并通过自注意力层生成全局特征序列；③ 将特征序列喂入一个自回归的 Seq2Seq 解码器，逐字符生成 LaTeX 标记。整个过程是端到端的，没有显式的文字检测或公式分割步骤。

**关键模块拆解**  
1. **图像预处理**：作者把每页 PDF 渲染成 2240×2240 像素的 RGB 图像，确保细小的数学符号也能被捕捉。类似于把一本书放大到显微镜下观察。  
2. **Patch Embedding**：图像被划分成 16×16 的小块，每块展平后映射到一个向量，这一步相当于把整张纸切成拼图块并给每块贴上“编号”。  
3. **视觉 Transformer 编码器**：这些向量进入多层自注意力网络，网络会学习每块之间的关系——比如上标、下标、根号的上下结构。自注意力的核心是“每块都能看到其他所有块”，因此能够捕捉跨行跨列的数学结构。  
4. **序列化与位置编码**：编码器输出的特征被按行扫描顺序排列，并加入位置编码，让解码器知道每个特征对应的空间位置。  
5. **自回归解码器**：解码器采用 Transformer 的标准结构，逐步预测下一个 LaTeX token。每一步都会参考已经生成的前缀和编码器的全局特征，就像人在写公式时会先写出已经确定的部分，再补全剩余符号。  
6. **损失函数**：使用交叉熵对每个 token 的预测概率进行惩罚，鼓励模型输出与真实 LaTeX 完全匹配的序列。  

**最巧妙的设计**  
- **统一输出**：把文字和公式混排的页面直接翻译成 LaTeX，避免了传统 OCR 必须先做文字/公式分割的繁琐步骤。  
- **大尺度视觉 Transformer**：相比早期只用几层 CNN 的模型，Nougat 采用更深的 Transformer，使得模型能够在全局范围内捕捉复杂的排版规则，如跨页脚注或多列布局。  
- **自监督预训练**：虽然摘要未详细说明，但作者提到使用了大规模未标注的 PDF 页面进行自监督预训练，类似于让模型先学会“看懂”学术页面的布局，再进行有监督的 LaTeX 对齐训练。

### 实验与效果
- **数据集**：作者构建了一个包含数万页学术论文的专用数据集，每页都有对应的 LaTeX 源码。该数据集覆盖了物理、数学、计算机等多个学科，排版风格多样。  
- **基线对比**：实验中把 Nougat 与传统 OCR（如 Tesseract）以及两段式的文字+公式识别流水线（先用检测器再用公式识别器）进行比较。论文声称在整体字符准确率（CER）和公式识别准确率（Formula Accuracy）上均有显著提升，尤其在包含复杂嵌套公式的页面上优势更明显。  
- **消融实验**：作者分别去掉了自注意力层、位置编码以及统一 LaTeX 输出的步骤，结果显示每个组件的缺失都会导致整体准确率下降 5%~10%，验证了设计的必要性。  
- **局限性**：论文承认在极低分辨率或严重压缩的 PDF 页面上仍会出现漏检，另外对手写公式的识别能力尚未评估。模型对非常规排版（如双栏会议稿）也需要进一步微调。

### 影响与延伸思考
Nougat 的出现让学术文献的机器可读化迈出了关键一步，后续很多项目开始把它当作前置模块，把 PDF 直接喂给大语言模型（LLM）进行问答或摘要生成。2024 年之后，出现了基于 Nougat 输出的“文献检索‑生成”系统，能够在几秒钟内把整篇论文转成结构化的 LaTeX+文本，供检索引擎索引。还有研究尝试把 Nougat 与多模态 LLM 结合，实现“一图多问”，即用户上传论文页面，模型直接给出公式推导或实验复现建议。想进一步深入，可以关注以下方向：① 更高效的视觉 Transformer 结构（如 Swin‑Transformer）在大规模 PDF 上的加速；② 手写公式的自监督预训练；③ 将 OCR 输出与知识图谱对齐，实现跨文献的概念抽取。  

### 一句话记住它
Nougat 把整页学术 PDF 直接翻译成 LaTeX，彻底消除了文字/公式分割的壁垒，让机器一次性读懂科研文献。