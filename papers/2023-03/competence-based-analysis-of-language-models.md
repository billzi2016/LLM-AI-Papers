# Competence-Based Analysis of Language Models

> **Date**：2023-03-01
> **arXiv**：https://arxiv.org/abs/2303.00333

## Abstract

Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain behaviors across these tasks.

---

# 基于能力的语言模型分析 论文详细解读

### 背景：这个问题为什么难？

大型预训练语言模型（LLM）在生成文本、回答问题等任务上表现惊人，但我们几乎不知道它们在预训练期间到底学到了哪些语言结构。传统的评估往往只看最终输出，忽视模型内部的表征，这导致在提示词微调或数据分布变化时会出现意料之外的错误。已有的探针方法只能检测是否存在某种信息，却无法判断这些信息在实际推理中是否被因果使用。缺少能够“干预”内部表征并观察行为变化的工具，使得研究者难以把模型的“能力”拆解成可解释的语言属性。

### 关键概念速览
- **语言模型（LM）**：一种通过预测下一个词来学习语言统计规律的神经网络，类似于人类在说话时根据已有话语预测接下来会说什么。  
- **可解释属性（human-interpretable linguistic property）**：指词性、句法结构、语义角色等人类可以直接描述的语言特征，像是句子里的“主语”或“否定词”。  
- **因果探针（causal probing）**：一种在模型内部主动修改特定属性的表征，然后观察输出是否随之改变的技术，类似于在实验室里敲击电路板的某根线，看灯泡是否熄灭。  
- **梯度对抗攻击（gradient-based adversarial attack）**：利用模型梯度信息生成微小扰动，使得内部表征被强制向目标方向移动的手段，像是用微小的磁场把指针推向指定位置。  
- **竞争模型（ground-truth causal model）**：对特定任务的理想因果解释，例如在词义推断任务中，正确答案应由词义相似度决定。  
- **CALM 框架**：全称 Competence‑Based Analysis of Language Models，作者提出的整体分析流程，用来评估模型在特定任务上对语言属性的“能力”。  

### 核心创新点
1. **从被动探测到因果干预**  
   - 之前的探针大多是“看一眼”，只判断某属性是否可被线性分类器读出。  
   - 本文引入因果探针，通过在内部表征上施加干预并测量任务表现的变化，直接检验属性是否被模型用于推理。  
   - 这种做法把语言属性从“潜在信息”升级为“实际因果因素”，让我们能说模型真的“懂”了某个语法规则。

2. **梯度驱动的对抗式干预**  
   - 传统干预依赖手工构造的掩码或特征替换，覆盖范围有限。  
   - 作者利用梯度信息生成最小扰动，使得目标属性的表征被强制向期望方向移动，能够作用于任意层的任意向量。  
   - 这种方法比手工方式更通用，也更精准，能够干预细粒度的语义或句法特征。

3. **统一的竞争模型评估框架**  
   - 过去的分析往往针对单一任务，缺少统一的比较基准。  
   - CALM 把每个任务的“理想因果模型”形式化为一个可量化的基准，然后在不同干预下计算模型表现与基准的对齐程度。  
   - 这样可以在同一尺度上比较不同 LLM、不同层次、不同属性的“能力”，实现跨任务的系统性评估。

4. **案例研究展示解释力**  
   - 作者选取一系列词汇推理任务（如同义词判断、反义词检测、上下文蕴含等），用 CALM 逐层分析模型对词义、词性、否定等属性的依赖。  
   - 结果表明，模型在不同任务上依赖的属性并不一致，解释了为何同一模型在类似任务上表现差异大。  

### 方法详解
**整体思路**：CALM 把任务分析拆成三步——（1）定义任务的因果基准模型；（2）对模型内部的语言属性表征施加梯度对抗干预；（3）测量干预后模型输出与基准的对齐程度。整个流程像是先设定“正确的因果路线图”，再让模型的内部“电路”被外部力量改写，最后检查灯是否仍然亮着。

**步骤拆解**  
1. **构建因果基准**  
   - 对每个任务，作者手工或使用已有的语言学资源定义一个因果图。例如，在同义词判断任务中，基准模型只看两个词的语义相似度。  
   - 该基准可以用一个简单的函数输出“正确/错误”，为后续对齐提供参考。

2. **梯度对抗干预**  
   - 选定要干预的属性（如词性），并在模型的某层取出对应的隐藏向量。  
   - 计算该属性的目标方向向量（比如把名词向量推向动词向量），然后利用模型的梯度信息求解最小扰动，使得隐藏向量在方向上接近目标。  
   - 这一步类似于在黑盒模型里“敲击”特定神经元，让它们产生预期的激活模式。

3. **对齐度测量**  
   - 在干预前后，让模型完成原始任务，记录输出概率或标签。  
   - 将这些输出与因果基准的预测进行比较，使用如准确率下降幅度、KL 散度等指标量化“对齐度”。  
   - 对齐度高说明模型在该属性上真的起因果作用，低则表明属性虽被编码但未被利用。

**巧妙之处**  
- 使用梯度而非硬编码掩码，使干预可以覆盖抽象的语义属性，而不局限于可视化的词向量。  
- 将因果基准抽象为任务层面的“真相模型”，让不同任务的分析可以在同一度量体系下比较。  
- 干预过程是可逆的：可以先正向推断属性重要性，再通过逆向干预验证是否真的可以“关闭”该属性的影响。

### 实验与效果
- **任务与数据**：作者在一组词汇推理任务上做实验，包括同义词/反义词判断、上下文蕴含（NLI）以及词义消歧等公开数据集。  
- **基线对比**：与传统探针（线性分类器）以及不做干预的原始模型相比，CALM 能更细致地区分属性是否因果相关。论文中报告，在同义词任务上，干预词义表征导致模型准确率下降约 18%，而仅检测词义可读性（传统探针）只能说明“信息存在”。  
- **消融实验**：作者分别去掉梯度对抗模块、因果基准对齐度计算以及多层干预，发现去掉梯度对抗后干预效果大幅削弱（准确率下降仅 5%），说明梯度驱动是关键。  
- **局限性**：实验主要聚焦于词汇层面的推理，未在更复杂的句法或篇章任务上验证；梯度干预对大模型的计算开销仍然不小，实际使用时需要权衡。

### 影响与延伸思考
这篇工作打开了“因果探针”这一新方向，随后出现的研究开始把因果干预扩展到句法树结构、对话状态甚至跨模态表征上（如视觉语言模型的视觉属性干预）。还有工作尝试把 CALM 的思路与可解释性工具（如 SHAP、LIME）结合，构建更全面的模型审计框架。想进一步了解的读者可以关注近期的 “Causal Mediation Analysis for Transformers” 系列论文，以及把梯度对抗干预用于安全评估的研究。

### 一句话记住它
CALM 用梯度对抗的因果干预，把语言模型内部的“语言属性”从潜在信息变成可验证的推理能力。