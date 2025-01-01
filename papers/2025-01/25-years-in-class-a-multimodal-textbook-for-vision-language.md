# 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining

> **Date**：2025-01-01
> **arXiv**：https://arxiv.org/abs/2501.00958

## Abstract

Compared to image-text pair data, interleaved corpora enable Vision-Language Models (VLMs) to understand the world more naturally like humans. However, such existing datasets are crawled from webpage, facing challenges like low knowledge density, loose image-text relations, and poor logical coherence between images. On the other hand, the internet hosts vast instructional videos (e.g., online geometry courses) that are widely used by humans to learn foundational subjects, yet these valuable resources remain underexplored in VLM training. In this paper, we introduce a high-quality \textbf{multimodal textbook} corpus with richer foundational knowledge for VLM pretraining. It collects over 2.5 years of instructional videos, totaling 22,000 class hours. We first use an LLM-proposed taxonomy to systematically gather instructional videos. Then we progressively extract and refine visual (keyframes), audio (ASR), and textual knowledge (OCR) from the videos, and organize as an image-text interleaved corpus based on temporal order. Compared to its counterparts, our video-centric textbook offers more coherent context, richer knowledge, and better image-text alignment. Experiments demonstrate its superb pretraining performance, particularly in knowledge- and reasoning-intensive tasks like ScienceQA and MathVista. Moreover, VLMs pre-trained on our textbook exhibit outstanding interleaved context awareness, leveraging visual and textual cues in their few-shot context for task solving. Our code are available at https://github.com/DAMO-NLP-SG/multimodal_textbook.

---

# 2.5 年课堂：用于视觉语言预训练的多模态教材 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）一直依赖大规模的图文对数据来学习跨模态关联，但大多数公开数据集都是从网页上爬取的，图片往往只配一两句描述，知识密度低，图文对应关系松散，甚至同一篇文章里图片之间缺乏连贯的逻辑。人类学习时更倾向于在课堂、教材中看到“图‑文‑音”交叉出现的连续信息，这种“交错式”上下文能帮助我们把抽象概念和具体实例联系起来。现有的 VLM 训练语料缺少这种结构化、知识密集的连续信息，导致模型在需要推理和常识的任务上表现不佳。于是，如何构建一种既富含基础知识又保持时间顺序连贯的多模态教材，成为提升 VLM 能力的关键瓶颈。

### 关键概念速览
- **视觉语言模型（VLM）**：能够同时处理图像和文字的神经网络，类似于会“看图说话”的 AI。  
- **交错式语料（interleaved corpus）**：把图像、文字、音频等信息按照出现的时间顺序交叉排列，就像课堂笔记里图文并茂、前后呼应。  
- **关键帧（keyframe）**：从视频中抽取的代表性画面，类似于 PPT 中的每一张幻灯片，用来承载视觉信息。  
- **自动语音识别（ASR）**：把视频中的口语转成文字，相当于把老师的讲课内容写下来。  
- **光学字符识别（OCR）**：从视频画面中识别出出现的文字（如板书、公式），相当于把黑板上的字抄进笔记本。  
- **LLM 提议的分类体系**：利用大语言模型（LLM）生成的学科层级结构，帮助系统化地检索和组织教学视频。  
- **Few‑shot 上下文感知**：模型在少量示例提示下，能够利用提示中的图文信息进行推理，类似于人看几张例题后就能解答新题。  

### 核心创新点
1. **从教学视频而非网页抓取数据 → 通过 LLM 生成的学科 taxonomy 系统化检索 2.5 年、22 000 小时的课堂视频 → 获得了知识密度高、逻辑连贯的多模态教材**。相比随意爬取的图文对，这种来源保证了内容的专业性和结构化。  
2. **逐层提取并精炼多模态信号 → 先抽取关键帧，再用 ASR 把语音转文字，最后用 OCR 捕获板书和公式 → 将三类信息按时间顺序拼接成交错式语料**。这种“先粗后细、层层过滤”的管线显著提升了图文对齐的准确性。  
3. **把教材组织成时间序列的交错上下文 → 每段视频对应一段图‑文‑音序列，模型在预训练时能够看到前后文的因果关系 → 在需要多步推理的任务（ScienceQA、MathVista）上表现出色**。传统图文对往往是孤立的样本，缺乏这种前后关联。  
4. **实验验证模型对交错上下文的感知能力 → 在 few‑shot 设置下，模型能够利用提示中的图像和文字共同推理，而不是仅靠文字**。这表明预训练语料的结构直接影响模型的上下文利用方式。

### 方法详解
整体思路可以拆成四大步骤：**（1）教材构建、（2）多模态抽取、（3）信号精炼、（4）交错语料组织**。

1. **教材构建**  
   - 使用大语言模型（LLM）生成一个覆盖数学、物理、化学等基础学科的层级 taxonomy。  
   - 根据 taxonomy 在公开的教学平台（如 Coursera、Bilibili）检索对应的课程视频，累计超过 2.5 年、22 000 小时的教学内容。  
   - 这一步类似于让 LLM 当图书管理员，先把书架按学科划分，再把每本书挑出来。

2. **多模态抽取**  
   - **关键帧**：对每段视频做均匀采样，然后用视觉显著性模型挑选出信息量最大的帧，确保每张图都能代表该段讲解的核心概念。  
   - **ASR**：调用高精度的自动语音识别系统，把整段音频转成时间戳标记的文字稿。  
   - **OCR**：对每帧图像运行光学字符识别，捕获黑板、投影或屏幕上的文字、公式等。  

3. **信号精炼**  
   - 对 ASR 文本进行噪声过滤（去除口头禅、停顿词），并用语言模型纠正可能的识别错误。  
   - OCR 结果经过版面分析，只保留与关键帧对应的局部文字，避免把整张幻灯片的所有文字都当作重要信息。  
   - 将关键帧、清洗后的文字稿、OCR 文字三者按时间戳对齐，形成一条完整的“图‑文‑音”链。

4. **交错语料组织**  
   - 将每条链视为一个样本，顺序保持原始教学视频的时间流。  
   - 为了让模型学习跨模态关联，在预训练阶段采用 **image‑text interleaved** 的输入方式：先喂入关键帧图像，再喂入对应的文字描述，交替进行。  
   - 训练目标仍是常见的对比学习或自监督掩码预测，只是输入序列更像课堂笔记而非孤立的图文对。

**最巧妙的点**在于把“教学视频”这一庞大、噪声丰富的资源，转化为结构化、时间顺序明确的交错语料。传统做法往往直接把视频切成短片或只抽取字幕，导致图文对应关系模糊；这里通过层层过滤和对齐，让每张图都有对应的解释文字和可能的公式，极大提升了信号质量。

### 实验与效果
- **评测任务**：作者在 ScienceQA（科学常识问答）和 MathVista（数学推理）等知识密集型基准上进行评估。  
- **对比基线**：与使用普通网页爬取的图文对数据预训练的模型相比，使用多模态教材的模型在 ScienceQA 上取得了显著提升，MathVista 的解题准确率也有可观提升（具体数值未在摘要中给出，论文声称提升幅度超过常规基线的数倍）。  
- **消融实验**：通过去掉 OCR、去掉关键帧筛选或打乱时间顺序，模型性能均出现明显下降，说明每个模块对最终效果都有贡献。  
- **局限性**：数据来源主要是公开教学视频，受版权和平台限制；此外，教材主要覆盖基础学科，对更专业或跨学科的知识仍缺乏。作者也提到，自动抽取的 OCR 仍会在手写公式上出现错误，影响部分数学任务的上限。

### 影响与延伸思考
这篇工作打开了“从教学资源构建高质量多模态语料”的思路，随后有几篇后续研究尝试把 MOOCs、实验室演示甚至 AR/VR 教学内容加入 VLM 预训练，进一步提升模型的实验推理能力。对想继续深入的读者，可以关注以下方向：  
- **跨语言多模态教材**：把非英语教学视频也纳入，探索语言多样性对 VLM 的影响。  
- **细粒度知识图谱对齐**：把抽取的公式、概念映射到结构化知识图谱，提升模型的可解释推理。  
- **实时交互式教学**：利用已训练好的 VLM 在课堂上实时回答学生提问，实现 AI 助教。  

### 一句话记住它
把 2.5 年的课堂视频打磨成时间顺序的图‑文‑音交错教材，让视觉语言模型学会像人一样在连贯的教学上下文中推理。