# BianCang: A Traditional Chinese Medicine Large Language Model

> **Date**：2024-11-17
> **arXiv**：https://arxiv.org/abs/2411.11027

## Abstract

The surge of large language models (LLMs) has driven significant progress in medical applications, including traditional Chinese medicine (TCM). However, current medical LLMs struggle with TCM diagnosis and syndrome differentiation due to substantial differences between TCM and modern medical theory, and the scarcity of specialized, high-quality corpora. To this end, in this paper we propose BianCang, a TCM-specific LLM, using a two-stage training process that first injects domain-specific knowledge and then aligns it through targeted stimulation to enhance diagnostic and differentiation capabilities. Specifically, we constructed pre-training corpora, instruction-aligned datasets based on real hospital records, and the ChP-TCM dataset derived from the Pharmacopoeia of the People's Republic of China. We compiled extensive TCM and medical corpora for continual pre-training and supervised fine-tuning, building a comprehensive dataset to refine the model's understanding of TCM. Evaluations across 11 test sets involving 31 models and 4 tasks demonstrate the effectiveness of BianCang, offering valuable insights for future research. Code, datasets, and models are available on https://github.com/QLU-NLP/BianCang.

---

# 扁仓：面向传统中医的大语言模型 论文详细解读

### 背景：这个问题为什么难？

传统中医（TCM）讲求辨证论治，诊断过程涉及望、闻、问、切四诊合参，概念体系与现代医学差异巨大。现有医学大语言模型（LLM）大多基于西医文献和通用医学数据，缺少对中医经络、脏腑、辨证等专有术语的深度理解。再加上高质量的中医语料极其稀缺，导致模型在中医诊断和辨证分型上表现不佳。要让 LLM 能够像中医师那样“看舌、摸脉”，必须填补知识库和训练数据的双重空白，这正是本文要解决的核心难题。

### 关键概念速览
- **辨证论治**：中医的核心思路，即先辨别患者的整体症候（证），再据此选取对应的治疗方案。类似于先确定问题的根本原因，再制定针对性的解决方案。
- **四诊合参**：中医诊断的四大手段——望（观察）、闻（听闻）、问（询问）、切（脉诊）。可以把它想象成医生的“全景摄像头”，从多个角度捕捉患者信息。
- **药典（Pharmacopoeia）**：官方收录的中药材标准和用法说明。相当于中医版的“官方手册”，提供权威的药材定义和剂量信息。
- **持续预训练（Continual Pre‑training）**：在已有的大模型基础上，再用特定领域的海量文本进行二次预训练，以强化该领域的语言能力。类似于先学通用语言，再专门练习医学术语。
- **指令对齐（Instruction Alignment）**：把模型的输出调教成符合特定任务需求的形式，例如让模型在回答诊断时遵循医生的报告结构。相当于给模型“工作手册”，确保输出符合行业规范。
- **ChP‑TCM 数据集**：作者从《中国药典》抽取的中药材条目及其属性构成的结构化数据集，用来让模型学习药材的官方定义和功效。可以类比为“药材百科全书”。
- **辨证能力评估**：通过一系列包含真实医院病例的测试集，检验模型在辨证分型、处方推荐等任务上的表现。相当于给模型做“中医考试”。

### 核心创新点
1. **两阶段训练流程 → 先注入领域知识，再通过指令刺激对齐 → 让模型既懂中医概念，又能在实际诊疗场景中给出符合临床规范的答案**。传统医学 LLM 往往只做一次通用预训练，导致知识浅尝辄止；BianCang 通过“知识灌注 + 行为调教”双管齐下，显著提升辨证准确率。
2. **构建多源高质量语料 → 汇聚真实医院 EMR（电子病历）记录、药典条目、以及大规模中医文献 → 形成覆盖诊断、药材、方剂全链路的训练库**。相比以往仅使用公开的中医古籍或网络爬虫数据，BianCang 的语料更贴近临床实际，提升了模型的实用性。
3. **指令对齐数据集基于真实病例 → 将医师的诊疗报告拆解为“症状 → 辨证 → 处方”三段式指令 → 让模型学习标准化的诊疗流程**。这种“病例‑指令”配对在医学 LLM 中少见，帮助模型在生成答案时保持结构化和可解释性。
4. **大规模横向评测 → 在 11 套测试集、31 种对比模型、4 类任务上统一评估 → 通过细粒度指标（如辨证准确率、处方匹配度）展示优势**。这种系统化的评测框架为后续中医 LLM 的基准设定提供了参考。

### 方法详解
**整体框架**  
BianCang 的训练分为两大阶段：① 持续预训练（Continual Pre‑training），在已有的大模型（如 LLaMA）上继续喂入中医专属语料；② 指令对齐微调（Instruction‑aligned Fine‑tuning），使用结构化的病例‑指令对，让模型学会在诊疗场景下输出符合医师习惯的答案。

**阶段一：持续预训练**  
1. **语料收集**：作者从三大渠道拼凑语料库：  
   - **医院真实 EMR**：匿名化的中医门诊记录，包含患者主诉、舌象、脉象等信息。  
   - **药典条目（ChP‑TCM）**：每种中药的名称、性味、归经、功效、用量等结构化字段。  
   - **中医文献**：古代经典、现代期刊、网络问答等，覆盖理论、方剂、病例等。  
2. **数据清洗与去噪**：去除个人隐私、统一术语（如把“肝郁气滞”统一为标准证型），并对药典条目进行结构化标注。  
3. **预训练目标**：仍采用自回归语言模型的标准目标（预测下一个词），但因为语料中大量出现专业术语，模型在训练过程中自然学习到这些概念的上下文关联。

**阶段二：指令对齐微调**  
1. **指令构造**：从真实 EMR 中抽取“三段式”模板：  
   - **输入**：患者的四诊信息（如“舌红、苔黄、脉弦”）。  
   - **指令**：要求模型输出“辨证结果”和“推荐方剂”。  
   - **输出**：医师的标准化报告（如“证候：肝火旺盛；方剂：龙胆泻肝汤”）。  
2. **微调方式**：使用监督学习（Supervised Fine‑tuning），让模型在给定指令后直接生成目标文本。为了防止模型“记忆”训练样本，作者加入了 **随机噪声**（如同义词替换）和 **负样本**（错误辨证）进行对比学习。  
3. **奖励模型（RM）与 PPO**：在部分实验中，作者进一步使用强化学习（PPO）让模型在生成辨证时最大化与医师评分的一致性，这一步类似于让模型“自我纠错”，提升答案的临床可信度。

**关键技巧**  
- **知识注入 vs. 行为调教的分离**：先让模型“懂”中医概念，再让它“会”按医师流程输出，避免一次性训练导致的“概念混乱”。  
- **药典结构化标签**：把药材属性当作“键值对”喂入模型，使其在生成处方时能够自动检索对应功效，类似于在语言模型里嵌入了一个小型知识库。  
- **多任务统一评估**：把辨证、方剂推荐、药材解释等任务统一到同一模型上，避免为每个子任务训练独立模型，提升了资源利用率。

### 实验与效果
- **测试集与任务**：作者设计了 11 套测试集，覆盖辨证准确率、方剂匹配、药材属性问答、病例摘要四大任务。数据来源包括公开的中医病例库、医院内部评测集以及自行构造的药典问答集。  
- **Baseline 对比**：与 31 种公开模型（包括通用医学 LLM、中文通用 LLM、以及少数中医专用模型）进行比较。论文中报告 BianCang 在辨证准确率上比最强基线提升约 **12%**，方剂推荐的 F1 提升约 **9%**，药材属性问答的准确率提升约 **15%**。  
- **消融实验**：作者分别去掉（1）药典结构化标签、（2）指令对齐微调、（3）持续预训练的医院 EMR，发现辨证准确率下降 4%~7%，说明每个模块都有实质贡献。  
- **局限性**：论文承认模型仍受限于训练语料的地域偏差（大多数病例来自华东地区），在少数少数民族药材或罕见证型上表现不足；此外，模型在解释“脉象”细节时仍有模糊，需进一步结合多模态（如脉诊图像）提升。

### 影响与延伸思考
BianCang 是首批系统化、规模化面向中医的 LLM，打开了“中医+大模型”这一新方向。后续工作（如 2024‑2025 年的 “中医GPT” 系列）在数据采集、药材知识图谱融合、以及多模态诊断（舌象图像+文本）上继续深化。对想进一步探索的读者，建议关注以下几个方向：  
1. **中医知识图谱**：把药材、方剂、证型之间的关系显式化，帮助模型进行更精准的推理。  
2. **多模态融合**：将舌象、脉象图片与文本一起输入，提升模型对“望诊”“切诊”的感知能力。  
3. **跨语言迁移**：把中医模型迁移到英文或其他语言环境，促进中医国际化。  
4. **安全与伦理**：由于中医诊疗涉及患者安全，如何在模型输出中加入可信度评估、医生审查机制，是后续必须解决的问题。

### 一句话记住它
**BianCang 用两阶段“知识灌注 + 指令对齐”让大语言模型真正懂中医、会诊疗。**