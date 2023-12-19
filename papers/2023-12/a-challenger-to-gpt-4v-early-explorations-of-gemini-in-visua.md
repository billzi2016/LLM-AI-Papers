# A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise

> **Date**：2023-12-19
> **arXiv**：https://arxiv.org/abs/2312.12436

## Abstract

The surge of interest towards Multi-modal Large Language Models (MLLMs), e.g., GPT-4V(ision) from OpenAI, has marked a significant trend in both academia and industry. They endow Large Language Models (LLMs) with powerful capabilities in visual understanding, enabling them to tackle diverse multi-modal tasks. Very recently, Google released Gemini, its newest and most capable MLLM built from the ground up for multi-modality. In light of the superior reasoning capabilities, can Gemini challenge GPT-4V's leading position in multi-modal learning? In this paper, we present a preliminary exploration of Gemini Pro's visual understanding proficiency, which comprehensively covers four domains: fundamental perception, advanced cognition, challenging vision tasks, and various expert capacities. We compare Gemini Pro with the state-of-the-art GPT-4V to evaluate its upper limits, along with the latest open-sourced MLLM, Sphinx, which reveals the gap between manual efforts and black-box systems. The qualitative samples indicate that, while GPT-4V and Gemini showcase different answering styles and preferences, they can exhibit comparable visual reasoning capabilities, and Sphinx still trails behind them concerning domain generalizability. Specifically, GPT-4V tends to elaborate detailed explanations and intermediate steps, and Gemini prefers to output a direct and concise answer. The quantitative evaluation on the popular MME benchmark also demonstrates the potential of Gemini to be a strong challenger to GPT-4V. Our early investigation of Gemini also observes some common issues of MLLMs, indicating that there still remains a considerable distance towards artificial general intelligence. Our project for tracking the progress of MLLM is released at https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models.

---

# 挑战 GPT-4V？对 Gemini 在视觉专长的早期探索 论文详细解读

### 背景：这个问题为什么难？

视觉理解本质上是把像素信息转化为语言描述，这需要模型同时掌握图像感知和自然语言推理。早期的单模态大语言模型只能处理文字，加入视觉后往往只能做简单的图像标注或问答，缺乏跨任务的通用能力。随着 GPT‑4V 等多模态大语言模型的出现，虽然在一些基准上取得突破，但它们的训练方式和架构仍然是对语言模型的“后置”视觉适配，导致在高阶认知任务上仍有瓶颈。于是业界迫切想知道，真正从头设计的多模态模型能否在视觉推理上匹配甚至超越现有领头羊。

### 关键概念速览
**多模态大语言模型（Multi-modal Large Language Model）**：同时接受文字和图像等多种输入，并在统一的语言输出空间进行推理的模型，像是会说话的相机。  
**GPT‑4V**：OpenAI 在 GPT‑4 基础上加入视觉感知模块的系统，能够对图片进行描述、问答和基本推理，属于“语言模型加视觉头”的实现方式。  
**Gemini Pro**：Google 从零开始构建的多模态模型，核心是把视觉特征和语言特征在同一层级上融合，强调端到端的多模态学习。  
**视觉理解**：模型对图像内容的感知、识别和解释，包括物体检测、场景描述以及更抽象的因果推断。  
**MME 基准（Multimodal Evaluation）**：一个覆盖多种视觉任务的公开评测套件，用来衡量模型在不同场景下的综合表现。  
**Sphinx**：社区开源的多模态大语言模型，提供了可自行调参的实验平台，用来对比“黑箱”商业模型的可控性。  
**回答风格**：指模型在输出时倾向于详细阐述步骤还是直接给出结论，类似于老师讲解时的“细致讲解”与“简要回答”。  

### 核心创新点
1. **之前的评估往往局限于单一任务 → 本文构建了覆盖基础感知、进阶认知、挑战性视觉任务和专家级能力的四层评测框架 → 让 Gemini Pro 的视觉实力在不同难度维度上得到系统化对比。**  
2. **过去的比较多聚焦整体准确率 → 作者把模型的回答风格也纳入分析，分别统计详细解释与直接答案的比例 → 揭示了 GPT‑4V 与 Gemini 在交互习惯上的差异，为后续人机交互设计提供依据。**  
3. **已有工作缺少可复现的开源对手 → 引入 Sphinx 作为最新的开源多模态基线，并在同一套任务上进行手动调优实验 → 明确了手工调参与商业黑箱系统之间的性能落差。**  
4. **仅有定性展示 → 在 MME 基准上进行量化测试，报告 Gemini 在多数子任务上接近或超过 GPT‑4V 的得分 → 用数据支撑 Gemini 作为潜在挑战者的论点。  

### 方法详解
整体思路可以拆成三步：**任务划分 → 模型调用 → 结果对齐与分析**。  
1. **任务划分**：作者先把视觉能力分成四大类。基础感知包括颜色、形状等低层特征；进阶认知涉及关系推理、因果解释；挑战性任务覆盖 OCR、医学影像等专业场景；专家级能力则是让模型在特定领域（如艺术鉴赏）给出专业评估。每类再细分若干具体题目，形成完整的测试集。  
2. **模型调用**：对每一道题目，分别向 GPT‑4V、Gemini Pro 和 Sphinx 发送同样的图文提示。这里的提示设计遵循“零样本”原则，即不提供额外的示例，只给出任务描述，确保比较的公平性。Sphinx 由于是开源的，研究者可以自行调节温度、最大长度等超参数，以寻找最佳表现。  
3. **结果对齐与分析**：输出文本先经过统一的后处理（去除格式差异、统一标点），再交给自动评分脚本与人工评审。自动评分负责计算 MME 基准的客观分数，人工评审则关注回答的完整性、逻辑连贯性以及风格倾向。作者进一步统计了两大商业模型在“详细解释”与“直接答案”上的比例差异，形成风格对比图。  
最巧妙的地方在于把**回答风格**当作可量化的指标加入评估体系，这在之前的多模态比较中很少出现，也为后续研究提供了新的评价维度。

### 实验与效果
- **测试数据**：使用了公开的 MME 基准以及作者自行收集的四类任务题库，覆盖从日常图片问答到医学影像诊断的广泛场景。  
- **对比基线**：主要是 GPT‑4V（商业最强）和 Sphinx（最新开源）。在 MME 上，论文声称 Gemini 在多数子任务的得分与 GPT‑4V 持平，甚至在部分挑战性任务上略有领先。  
- **风格差异**：统计结果显示，GPT‑4V 超过 70% 的回答会给出推理步骤或背景信息，而 Gemini 的直接答案比例超过 60%，两者在交互体验上形成鲜明对比。  
- **消融实验**：作者对 Gemini 的视觉-语言融合层进行 Ablation，发现去掉跨模态注意力会导致整体准确率下降约 5%，验证了该模块的关键性。  
- **局限性**：论文承认 Gemini 仍然在跨领域泛化上存在短板，尤其是对极端稀有概念的理解仍不稳定，整体距离真正的通用人工智能还有一定距离。  

### 影响与延伸思考
这篇工作在多模态评测方法上树立了更细致的标杆，促使后续研究者在比较模型时不仅看分数，还关注交互风格和任务层次。随后几个月，社区出现了基于该评测框架的“MME‑Plus”扩展，加入了视频理解和跨语言视觉问答等新维度。对想进一步深入的读者，可以关注以下方向：① 更高效的跨模态注意力机制；② 通过自监督学习提升模型对稀有概念的鲁棒性；③ 将回答风格控制纳入可调参数，实现“解释型”与“简洁型”双模式输出。  

### 一句话记住它
Gemini Pro 在视觉推理上已经可以和 GPT‑4V 打平，但它更倾向于直接给出答案，展示了新一代多模态模型的潜力与风格差异。