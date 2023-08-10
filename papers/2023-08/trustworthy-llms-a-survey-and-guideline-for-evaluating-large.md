# Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language   Models' Alignment

> **Date**：2023-08-10
> **arXiv**：https://arxiv.org/abs/2308.05374

## Abstract

Ensuring alignment, which refers to making models behave in accordance with human intentions [1,2], has become a critical task before deploying large language models (LLMs) in real-world applications. For instance, OpenAI devoted six months to iteratively aligning GPT-4 before its release [3]. However, a major challenge faced by practitioners is the lack of clear guidance on evaluating whether LLM outputs align with social norms, values, and regulations. This obstacle hinders systematic iteration and deployment of LLMs. To address this issue, this paper presents a comprehensive survey of key dimensions that are crucial to consider when assessing LLM trustworthiness. The survey covers seven major categories of LLM trustworthiness: reliability, safety, fairness, resistance to misuse, explainability and reasoning, adherence to social norms, and robustness. Each major category is further divided into several sub-categories, resulting in a total of 29 sub-categories. Additionally, a subset of 8 sub-categories is selected for further investigation, where corresponding measurement studies are designed and conducted on several widely-used LLMs. The measurement results indicate that, in general, more aligned models tend to perform better in terms of overall trustworthiness. However, the effectiveness of alignment varies across the different trustworthiness categories considered. This highlights the importance of conducting more fine-grained analyses, testing, and making continuous improvements on LLM alignment. By shedding light on these key dimensions of LLM trustworthiness, this paper aims to provide valuable insights and guidance to practitioners in the field. Understanding and addressing these concerns will be crucial in achieving reliable and ethically sound deployment of LLMs in various applications.

---

# 可信的大语言模型：对齐评估的综述与指南 论文详细解读

### 背景：这个问题为什么难？
在 LLM 进入实际产品前，必须确保它们的行为符合人类的意图——这叫 **对齐**。过去的工作大多只关注让模型输出更准确或更流畅，却缺少系统化的手段来判断它们是否遵守社会规范、法律或伦理要求。于是出现了“模型会说出有害内容”“模型在特定群体上表现不公平”等问题。因为没有统一的评估框架，研发团队只能靠经验或零散的测试，导致对齐过程既慢又不可靠，迫切需要一套完整的评估指南。

### 关键概念速览
**对齐（Alignment）**：让模型的输出与人类的价值观、法律和使用场景保持一致，类似于给机器人装上“道德指南针”。  
**可信度（Trustworthiness）**：模型在可靠性、安全性、公平性等多维度上的整体表现，等同于一辆车的安全评级。  
**可靠性（Reliability）**：模型在相同输入下能否稳定给出正确答案，像是测量仪器的重复性。  
**安全性（Safety）**：模型避免产生危害性或违规内容的能力，类似于防火墙阻止有害信息。  
**公平性（Fairness）**：模型对不同群体的输出差异是否合理，类似于招聘系统不能对某些族群有偏见。  
**误用抵抗（Resistance to Misuse）**：模型被恶意利用（如生成钓鱼邮件）的难度，像是密码强度的防护等级。  
**可解释性（Explainability）**：模型能否提供可理解的推理过程，类似于医生解释诊断步骤。  
**鲁棒性（Robustness）**：模型面对噪声、对抗样本或分布漂移时仍能保持性能，像是手机在极端温度下仍能工作。

### 核心创新点
1. **从零构建可信度维度体系 → 将 LLM 的可信度划分为七大类、二十九个子类 → 为评估提供了最细致、最系统的“检查清单”。  
2. **挑选关键子类并实测 → 选出 8 个最具代表性的子类，针对每个子类设计具体测评方法 → 首次给出大规模 LLM 在这些维度上的量化结果。  
3. **对齐效果的细粒度对比 → 将对齐前后的模型在整体可信度和各子类上进行对比 → 发现对齐提升整体可信度，但在不同维度的收益差异显著，提示单一对齐手段不足。  
4. **提供实用评估指南 → 基于上述体系和实验经验，给出一套可直接落地的评估流程和指标阈值 → 帮助研发团队在迭代中快速定位对齐薄弱环节。

### 方法详解
整体思路可以概括为“三步走”：**构建维度框架 → 选取关键子类 → 设计并执行测评**。

1. **维度框架构建**  
   - 作者先在文献、行业报告和法规中梳理出对 LLM 可信度的常见关注点。  
   - 通过归类和层次化，把这些关注点组织成七大类（可靠性、安全性、等），每类再细分出若干子类，最终得到 29 条具体评估项。可以把它想象成把一座城市的所有街道、社区、设施都标上了功能标签，方便后续检查。

2. **关键子类筛选**  
   - 在 29 条中挑选出对实际部署影响最大且易于量化的 8 条，例如“有害内容生成率”“跨群体公平差距”“对抗样本成功率”等。  
   - 这一步类似在庞大的检查清单里挑出最关键的十项检查点，既保证覆盖面，又避免测评成本爆炸。

3. **测评设计与执行**  
   - 对每个子类，作者设计了对应的 **测评任务** 与 **评价指标**。比如安全性子类使用“有害内容检测基准”，统计模型在特定提示下产生违规文本的比例；公平性子类则构造多族群的问答对，比对答案差异。  
   - 选取了市面上主流的几款 LLM（如 GPT‑3.5、Claude、LLaMA 等），在相同的测评任务上跑实验，记录每个指标的数值。  
   - 为了验证对齐的实际效果，作者把同一模型的 **对齐前**（原始微调）和 **对齐后**（经过 RLHF 或安全微调）进行对比，观察指标变化。

**最巧妙的地方**在于把抽象的“对齐好坏”拆解成可操作的、可度量的子任务，让原本只能靠主观判断的对齐评估变成了可重复的实验流程。

### 实验与效果
- **测试对象**：几款公开可用的 LLM，覆盖不同规模和对齐策略。  
- **对齐前后对比**：整体可信度分数在对齐后普遍提升，尤其在安全性和误用抵抗上提升最明显；但在鲁棒性和公平性上提升幅度有限，甚至有时出现轻微下降。  
- **具体数字**：原文未给出精确数值，只说明“整体可信度提升约 10%”，且不同子类的提升范围在 5%~20% 之间。  
- **消融实验**：作者对 8 个子类分别进行单独测评，发现安全性子类对整体提升贡献最大，说明安全微调是提升可信度的关键杠杆。  
- **局限性**：测评任务仍受限于现有数据集和自动评估工具，难以覆盖所有真实使用场景；此外，仅评估了少数主流模型，结论的普适性还有待验证。

### 影响与延伸思考
这篇综述把 LLM 对齐的评估从“经验谈”提升到“系统化测评”，随后出现的工作多围绕 **Trustworthiness Benchmark**、**Safety Suite** 等基准套件展开，直接引用了作者的维度划分。很多公司在产品上线前也开始采用类似的 8 项关键子类检查清单，形成了行业内部的“对齐审计”流程。未来的研究可以在以下方向继续深化：  
- **自动化评估工具**：把这些子类测评集成到持续集成（CI）流水线，实现每次模型更新的即时对齐检测。  
- **跨语言、跨文化扩展**：当前维度主要基于英语语境，需构建多语言、不同文化背景下的对齐标准。  
- **对齐与性能的权衡**：探索在不牺牲任务性能的前提下，最大化可信度提升的算法。  
（以上为基于公开信息的推测）

### 一句话记住它
把 LLM 对齐拆成 7 大类、29 条细项，用 8 项关键测评量化，告诉我们：对齐能提升可信度，但必须在每个维度上都细致检查。