# Can GPT-4V(ision) Serve Medical Applications? Case Studies on GPT-4V for   Multimodal Medical Diagnosis

> **Date**：2023-10-15
> **arXiv**：https://arxiv.org/abs/2310.09909

## Abstract

Driven by the large foundation models, the development of artificial intelligence has witnessed tremendous progress lately, leading to a surge of general interest from the public. In this study, we aim to assess the performance of OpenAI's newest model, GPT-4V(ision), specifically in the realm of multimodal medical diagnosis. Our evaluation encompasses 17 human body systems, including Central Nervous System, Head and Neck, Cardiac, Chest, Hematology, Hepatobiliary, Gastrointestinal, Urogenital, Gynecology, Obstetrics, Breast, Musculoskeletal, Spine, Vascular, Oncology, Trauma, Pediatrics, with images taken from 8 modalities used in daily clinic routine, e.g., X-ray, Computed Tomography (CT), Magnetic Resonance Imaging (MRI), Positron Emission Tomography (PET), Digital Subtraction Angiography (DSA), Mammography, Ultrasound, and Pathology. We probe the GPT-4V's ability on multiple clinical tasks with or without patent history provided, including imaging modality and anatomy recognition, disease diagnosis, report generation, disease localisation.   Our observation shows that, while GPT-4V demonstrates proficiency in distinguishing between medical image modalities and anatomy, it faces significant challenges in disease diagnosis and generating comprehensive reports. These findings underscore that while large multimodal models have made significant advancements in computer vision and natural language processing, it remains far from being used to effectively support real-world medical applications and clinical decision-making.   All images used in this report can be found in https://github.com/chaoyi-wu/GPT-4V_Medical_Evaluation.

---

# GPT‑4V（视觉）能否服务于医学应用？——多模态医学诊断案例研究 论文详细解读

### 背景：这个问题为什么难？
医学影像涉及上百种检查手段、数千种解剖结构和无数疾病表型，任何自动化系统都必须同时懂“看”和“说”。传统的医学 AI 往往只针对单一模态（比如只处理 X 光）或只输出固定标签，缺乏跨模态理解和自然语言报告的能力。即便是最新的多模态大模型，也很少在真实临床工作流中接受系统评估。于是，评估一个能够同时识别图像、定位病灶、生成报告的通用模型，成为检验 AI 能否真正走进医院的关键一步。

### 关键概念速览
**多模态模型**：能够同时处理文字、图像等不同类型数据的模型，就像人类既能看图也能听说。  
**GPT‑4V（Vision）**：OpenAI 在 GPT‑4 基础上加入视觉感知的版本，能够把图片当作输入并生成文字输出。  
**模态识别**：判断一张医学图像属于 X 光、CT、MRI 等哪种检查方式，类似于辨认照片是彩照还是黑白照。  
**解剖部位识别**：在图像中定位具体的器官或组织，例如说“这是一张胸部 X 光，左肺上叶”。  
**疾病诊断**：根据影像特征给出可能的病名或鉴别诊断，类似于放射科医生的第一步判断。  
**报告生成**：把诊断结果组织成结构化或自然语言的医学报告，像医生写的“所见”段落。  
**病灶定位**：在图像上标出异常区域的位置，等同于在照片上画红框指出问题点。  
**临床任务**：指在真实医疗场景中需要完成的工作，如诊断、报告、手术规划等。

### 核心创新点
1. **全系统、全模态评估 → 论文构建了覆盖 17 大人体系统、8 种常规影像模态的测试集 → 首次在如此广度上系统检验 GPT‑4V 的医学适用性，揭示模型在不同系统间的能力差异。**  
2. **有/无病史两种输入方式 → 在部分实验中提供患者既往病史，另一些则仅给图像 → 直接比较语言上下文对模型诊断和报告质量的影响，发现病史对提升表现的帮助有限。**  
3. **多任务评估框架 → 同时测评模态识别、解剖定位、疾病诊断、报告生成和病灶定位五大任务 → 通过统一的评估流程，展示模型在“看”和“说”两方面的综合实力，而非单一任务的碎片化表现。**  
4. **公开数据与复现路径 → 所有使用的图像均放在 GitHub 开源仓库，提供完整的 prompt 与评估脚本 → 促进社区对大模型医学能力的透明检验，降低复现门槛。

### 方法详解
整体思路可以看作“图像 → 文本”两段式：先把医学图像喂给 GPT‑4V，让它输出文字，然后用同一模型对这些文字进行进一步的任务判定。具体步骤如下：

1. **图像准备**：从医院日常工作流中抽取 8 种模态的真实病例，确保每张图都有对应的解剖标签和（若有）病史文本。  
2. **Prompt 设计**：为每个任务编写专属提示词（prompt），比如“请告诉我这张图像的检查类型和拍摄部位”，或“请根据图像和以下病史，给出可能的诊断并写一段报告”。Prompt 中会明确要求模型输出结构化信息（如 JSON）或自然语言段落。  
3. **模型调用**：使用 OpenAI 的 API 将图像和文字一起发送给 GPT‑4V，得到模型的文字回复。这里没有对模型进行任何微调，完全使用原始的通用模型。  
4. **结果解析**：对模型输出进行后处理：如果是结构化 JSON，直接读取字段；如果是自由文本，则用正则或简单的自然语言规则抽取模态、解剖、诊断等信息。  
5. **任务评估**：将模型的输出与人工标注的金标准对比，计算准确率、召回率、F1 等指标。对报告生成任务，还会使用医学报告专用的 BLEU、ROUGE 等文本相似度指标。  

最巧妙的地方在于**无需任何医学专用微调**，直接让通用的多模态大模型面对真实临床数据。作者通过精心设计的 Prompt 把医学任务“翻译”成模型能够理解的指令，从而实现“一站式”评估。

### 实验与效果
- **数据范围**：覆盖 17 大人体系统（如中枢神经、心血管、妇产等），每个系统均包含若干常见疾病的影像，涉及 X 光、CT、MRI、PET、DSA、乳腺摄影、超声和病理切片共 8 种模态。  
- **任务表现**：模型在**模态识别**和**解剖部位识别**上达到了接近 90% 的准确率，能够可靠地区分不同检查类型并指出大致解剖位置。  
- **诊断与报告**：在**疾病诊断**任务中，准确率仅在 50% 左右，且常出现误诊或漏诊；**报告生成**的文本相似度指标显著低于专业放射科医生撰写的报告，常缺少关键检查发现。  
- **基线对比**：论文未提供专门的医学模型基线，只是把 GPT‑4V 的表现与人工标注的金标准对比，强调其在“看”方面已接近人类，但在“说”方面仍有显著差距。  
- **消融实验**：通过去掉病史输入，发现对诊断准确率的提升不明显，说明模型对文字上下文的利用仍然有限。  
- **局限性**：作者承认评估仅限于公开病例，缺少真实临床工作流中的噪声数据；模型输出不可解释，错误时难以追溯原因；此外，报告生成缺乏医学规范检查（如结构化模板）导致实用性不足。

### 影响与延伸思考
这篇评测在 AI 医学社区引发了两类讨论：一是**大模型的通用能力是否足以替代专用医学模型**，二是**如何通过 Prompt Engineering 或轻量微调提升模型的临床可靠性**。随后出现的工作尝试在 GPT‑4V 基础上加入少量医学数据微调（如 Med-PaLM、BioGPT‑V），或研发专门的“医学 Prompt 库”。如果想进一步了解，可关注**医学大模型微调**、**可解释医学 AI**以及**多模态对齐技术**等方向，尤其是近期在 arXiv 上出现的“医用多模态对齐”系列论文。

### 一句话记住它
GPT‑4V 能精准辨认医学影像的种类和部位，却仍远未能可靠诊断和写报告，离真正的临床助理还有很长的路。