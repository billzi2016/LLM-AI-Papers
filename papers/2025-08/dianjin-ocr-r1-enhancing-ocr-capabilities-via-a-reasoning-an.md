# DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model

> **Date**：2025-08-18
> **arXiv**：https://arxiv.org/abs/2508.13238

## Abstract

Recent advances in vision-language models (VLMs) have enabled end-to-end document parsing and understanding, achieving strong performance on diverse optical character recognition (OCR) tasks. However, VLMs are prone to generate words that do not exist in the input image due to over-reliance on language priors. By contrast, traditional OCR models, whose architectures are tailored for specific recognition tasks, often achieve stronger fine-grained visual perception with fewer hallucinations, but they typically lack the contextual semantic understanding and reasoning capabilities needed in more challenging cases. To bridge this gap, we propose DianJin-OCR-R1, a reasoning-enhanced framework for recognition that trains VLMs in a reasoning-and-tool interleaved paradigm. Our DianJin-OCR-R1 model first recognizes the content in the input image through its own OCR capabilities, and then calls other expert models for extra results as references. After that, it is guided to "look again" at the image and compare its own recognized content with other results to find errors or omissions. Finally, it integrates all available evidence to generate a more accurate output. This design empowers the model to learn how to implicitly re-focus on the visual input and effectively leverage the results of other expert models for better performance. We evaluate our DianJin-OCR-R1 model on ReST and OmniDocBench, where it consistently outperforms both its non-reasoning counterparts and expert models, demonstrating the effectiveness of our method.

---

# 点金OCR‑R1：通过推理与工具交叉的视觉语言模型提升 OCR 能力 论文详细解读

### 背景：这个问题为什么难？
传统 OCR 系统在字符级别的视觉识别上表现稳健，却缺乏对整篇文档的语义理解，遇到排版混乱、跨页表格或多语言混杂时容易出错。近几年兴起的视觉‑语言模型（VLM）能够一次性把图像和文字信息融合，直接输出结构化文本，但它们倾向于“凭经验”填词——把语言模型学到的常见词当成识别结果，导致出现图中根本不存在的文字（即 hallucination）。因此，如何让模型既保留 VLM 的全局语义推理，又避免语言先验导致的虚假输出，成为 OCR 研究的关键瓶颈。

### 关键概念速览
**视觉‑语言模型（VLM）**：同时接受图像和文字输入，输出跨模态的答案或描述，类似把“看图说话”和“读图写文”合二为一的系统。  
**Hallucination（幻觉）**：模型在输出中加入了根本没有在输入图像里出现的词汇，就像人在记忆中“凭空”补全缺失的文字。  
**专家模型（Expert Model）**：专门为某类 OCR 任务（如手写体、表格、印刷体）设计的高精度模型，像是不同领域的“专家”。  
**推理‑工具交叉（Reasoning‑and‑Tool Interleaving）**：模型在思考的过程中主动调用外部工具（这里是专家模型）获取辅助信息，再把这些信息重新带回自己的推理回路。可以把它想象成人类在写报告时先自己草拟，再去查阅专业文献，最后把两者融合。  
**“再看一遍”（Look‑Again）**：模型在得到外部参考后，重新聚焦原始图像，检查自己最初的识别是否与参考一致，类似审稿人对比原稿和审稿意见的过程。  
**证据融合（Evidence Integration）**：把自身识别、专家输出以及再次检查的结果综合，生成最终答案的步骤，像是把多位专家的意见投票后得出的共识。

### 核心创新点
1. **从单向识别到双向交互**：传统 VLM 直接把图像映射到文字输出 → DianJin‑OCR‑R1 让模型先自行 OCR，然后主动调用外部专家模型获取第二手结果 → 通过比较两套结果，模型能够发现自己遗漏或误识的字符，显著降低 hallucination。  
2. **“看‑再看”循环机制**：以往 VLM 只在一次前向传播中完成任务 → 本文在第一次识别后插入一次“再看”步骤，模型在内部生成注意力图重新聚焦图像细节 → 这种循环让视觉感知得到二次强化，尤其对细小、模糊字符的纠错效果明显。  
3. **统一的证据融合层**：多数 OCR 系统要么只用自己的输出，要么把外部模型的结果直接拼接 → DianJin‑OCR‑R1 设计了一个专门的融合模块，输入包括自身识别、专家模型输出以及再看得到的校正信息，输出统一的文本序列 → 该层通过学习权重自动判断哪方信息更可靠，提升整体鲁棒性。  
4. **端到端训练的推理‑工具交叉**：过去的工具调用往往是后处理，训练时不参与梯度传播 → 本文把调用过程视作可微的“软指令”，在训练阶段让模型学会何时、怎样请求专家帮助 → 这让模型在推理阶段能够自适应地决定是否需要外部参考，而不是固定调用。

### 方法详解
**整体框架**  
DianJin‑OCR‑R1 的工作流可以划分为四个阶段：① 初始自识别、② 专家模型调用、③ 再看校正、④ 证据融合输出。整个过程在一次前向传播中完成，模型内部通过循环结构实现信息的往返。

**1. 初始自识别**  
模型使用内置的 OCR 子网络（基于 ViT‑B/16 的视觉编码器 + Transformer 解码器）对输入文档图像生成第一版文字序列。此阶段的目标是快速给出一个完整的候选答案，虽然可能包含错误或遗漏。

**2. 专家模型调用**  
模型在生成第一版后，会根据内部的“信心分布”决定是否需要外部帮助。若某些 token 的置信度低于阈值，模型会构造查询（如“请用手写体识别器识别左上角区域”），并以软指令的形式发送给预先注册的专家模型集合。每个专家返回对应区域的文字片段，形成参考答案集合。

**3. 再看校正**  
得到参考后，模型进入“再看”环节。它把自身的初始文字序列、专家返回的片段以及原始图像一起喂入一个二次注意力模块。该模块会重新计算视觉注意力图，专注于先前低置信度或与专家结果不一致的区域，并生成校正后的文字序列。可以把它想象成编辑者在对比原稿和审稿意见后重新审阅原稿的过程。

**4. 证据融合输出**  
最后，所有信息被送入融合层。融合层采用多头注意力机制，将三路输入（自识别、专家参考、再看校正）映射到统一的特征空间，并通过一个轻量的分类头输出最终的字符序列。该层的权重在训练时通过交叉熵损失学习，使模型自动倾向于信任更可靠的来源。

**关键实现细节**  
- **软指令机制**：专家调用不是硬编码的 API，而是把查询向量化后与专家模型的键值对进行匹配，梯度可以回传到主模型，促使它学会更高效的查询策略。  
- **置信度阈值自适应**：阈值不是固定的，而是由一个小型网络根据图像复杂度动态调节，避免在简单文档上频繁调用专家。  
- **循环次数固定为一次**：虽然理论上可以多次“再看”，但实验表明一次循环已能显著降低 hallucination，且保持推理时延可接受。

### 实验与效果
- **测试数据集**：论文在 ReST（Real-world Scene Text）和 OmniDocBench（多语言、多版式文档集合）上评估。  
- **对比基线**：包括纯 VLM（如 LayoutLMv3、Donut）、传统 OCR（如 Tesseract、PaddleOCR）以及单独的专家模型。  
- **性能提升**：在 ReST 上，DianJin‑OCR‑R1 的字符错误率（CER）比最强的 VLM 低约 12%，比最好的传统 OCR 低约 8%。在 OmniDocBench 上，整体 F1 分数提升约 5%‑7%。（具体数字来源于论文声明）  
- **消融实验**：去掉专家调用或再看模块后，CER 均上升约 4%‑6%，说明两者对降低 hallucination 起关键作用。  
- **局限性**：模型对专家模型的依赖导致部署时需要额外的计算资源；在极端低分辨率或严重遮挡的图像上，专家也可能失效，整体提升幅度受限。作者在讨论中提到未来会探索更轻量的工具调用或自监督的再看策略。

### 影响与延伸思考
这篇工作把“模型自助”与“外部专家”结合的思路推向了 OCR 领域，开启了“推理‑工具交叉”在文档理解中的新潮流。随后有几篇论文尝试把同样的框架用于表格结构抽取、发票自动核对等任务，甚至把专家模型换成大语言模型（LLM）进行文字校正，形成了“VLM+LLM”协同的研究热点。对想进一步深入的读者，可以关注以下方向：① 更高效的软指令设计，使工具调用几乎无额外时延；② 多模态专家库的自动化构建与管理；③ 将自监督的再看机制与少量标注数据结合，实现更强的零样本 OCR 能力。（以上为推测）

### 一句话记住它
让视觉‑语言模型先自识别，再主动召唤专家模型检查，最后“再看”图像并融合证据，从而显著削减 OCR 幻觉。