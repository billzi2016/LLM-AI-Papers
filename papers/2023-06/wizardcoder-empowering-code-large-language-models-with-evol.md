# WizardCoder: Empowering Code Large Language Models with Evol-Instruct

> **Date**：2023-06-14
> **arXiv**：https://arxiv.org/abs/2306.08568

## Abstract

Code Large Language Models (Code LLMs), such as StarCoder, have demonstrated exceptional performance in code-related tasks. However, most existing models are solely pre-trained on extensive raw code data without instruction fine-tuning. In this paper, we introduce WizardCoder, which empowers Code LLMs with complex instruction fine-tuning, by adapting the Evol-Instruct method to the domain of code. Through comprehensive experiments on four prominent code generation benchmarks, namely HumanEval, HumanEval+, MBPP, and DS-1000, we unveil the exceptional capabilities of our model. It surpasses all other open-source Code LLMs by a substantial margin. Moreover, our model even outperforms the largest closed LLMs, Anthropic's Claude and Google's Bard, on HumanEval and HumanEval+. Our code, model weights, and data are public at https://github.com/nlpxucan/WizardLM

---

# WizardCoder：用 Evol‑Instruct 赋能代码大语言模型 论文详细解读

### 背景：这个问题为什么难？

代码大语言模型（Code LLM）在生成、补全和调试代码方面已经展示出惊人的潜力，但大多数模型只经过海量原始代码的自监督预训练，缺少针对指令的微调。没有指令微调，模型往往只能“照搬”训练数据，难以理解用户的意图、处理多步骤的编程任务或给出解释性答案。传统的指令微调方法大多面向自然语言，对代码的语法、执行语义以及安全约束考虑不足，导致在实际编程场景里表现不佳。于是，如何让代码模型既保留大规模代码知识，又能像聊天机器人一样精准响应复杂指令，成为亟待突破的瓶颈。

### 关键概念速览

**Code LLM（代码大语言模型）**：在海量源码上进行自监督训练的语言模型，能够生成或理解程序代码。类似于会写代码的“自动补全”助手。

**指令微调（Instruction Fine‑tuning）**：在已有模型基础上，用“指令 + 答案”对进行二次训练，让模型学会按照自然语言指令完成任务。相当于给模型上了一门“使用说明书”。

**Evol‑Instruct**：一种进化式指令生成与微调框架，先让模型自行产生多样化的指令‑答案对，再通过筛选、迭代提升质量，类似于让模型在“自我训练营”里不断进化。

**HumanEval / HumanEval+ / MBPP / DS‑1000**：四个公开的代码生成基准。HumanEval 侧重函数实现，HumanEval+ 加入了更难的边界情况，MBPP 包含简短的编程任务，DS‑1000 则覆盖多语言、多难度的真实项目代码。

**闭源大模型（Closed‑source LLM）**：指那些由商业公司内部维护、代码和权重不公开的模型，例如 Anthropic 的 Claude、Google 的 Bard。

### 核心创新点

1. **把 Evol‑Instruct 从自然语言迁移到代码**  
   之前的 Evol‑Instruct 只在普通对话或问答上生成指令‑答案对。本文把同样的进化思路搬到代码领域，先让基准代码模型自行产生多样化的编程指令和对应实现，再通过自动评估筛选出高质量样本用于微调。这样模型在微调阶段就已经见过“写函数、解释代码、调试错误”等真实编程指令。

2. **复杂指令的层次化构造**  
   传统指令微调往往只提供单一步骤的任务描述。WizardCoder 通过 Evol‑Instruct 的迭代生成，构造出包含多步骤、需要中间解释或条件分支的指令。例如“实现二分搜索并解释时间复杂度”。这种层次化指令让模型在微调时学习到任务拆解和解释的能力。

3. **大规模公开代码+指令混合数据集**  
   作者在公开的代码库（如 StarCoder 的预训练数据）基础上，加入了数十万条高质量的指令‑答案对，并全部开源。相比仅使用原始代码的做法，这种混合数据让模型在保持代码知识的同时，获得了指令理解的“语言感”。

4. **在多个权威基准上实现跨模型领先**  
   通过上述三点改进，WizardCoder 在 HumanEval、HumanEval+、MBPP、DS‑1000 四个基准上均超过所有已公开的开源 Code LLM，甚至在 HumanEval 系列上跑赢了 Anthropic Claude 和 Google Bard 这两款闭源大模型。换句话说，指令微调让开源模型的实际编程能力实现了质的飞跃。

### 方法详解

**整体思路**  
WizardCoder 的训练流程可以划分为三步：① 预训练代码模型准备；② Evol‑Instruct 生成指令‑答案对并筛选；③ 用筛选后的指令数据对模型进行指令微调。整个过程像是先让模型“自我出题”，再让它“自我答题”，最后把优秀的问答写进记忆。

**步骤 1：代码模型基线**  
作者选用了公开的代码大模型（如 StarCoder）作为基线，这些模型已经在数十 TB 的源码上完成了自监督学习，具备基本的代码生成能力。

**步骤 2：Evol‑Instruct 生成指令**  
- **自我生成**：让基线模型在给定的代码片段或函数签名上，生成对应的任务描述（指令）以及完整实现（答案）。比如输入 `def fib(n):`，模型可能生成指令“实现斐波那契数列的递归版本”，并给出实现代码。  
- **进化筛选**：对生成的指令‑答案对进行自动评估，包括语法检查、单元测试通过率、答案完整度等。通过这些指标挑选出高质量的样本。  
- **迭代强化**：将筛选出的高质量对加入训练集，再次让模型基于更丰富的指令空间生成新样本。循环若干轮后，指令的多样性和难度都会显著提升。

**步骤 3：指令微调**  
把最终得到的指令‑答案对视作“指令微调数据”，采用标准的指令微调训练方式（如 LoRA 或全参数微调），让模型学习在看到自然语言指令时直接输出对应代码或解释。这里的关键是保持原有的代码知识不被微调过程冲淡，因此作者在微调时使用了较低的学习率和适度的正则化。

**巧妙之处**  
- **自我生成+自动评估**：不需要人工标注指令，完全依赖模型自身的创造力和程序化的评测脚本，极大降低了数据构建成本。  
- **多步骤指令的自然出现**：因为生成过程是基于已有代码上下文，模型会自发产生需要解释、调试或优化的指令，形成了天然的层次化任务。  
- **保持代码知识**：微调时使用了保守的学习率和 LoRA（低秩适配），确保模型在学习指令的同时不忘记已经掌握的大规模代码语义。

### 实验与效果

- **测试基准**：HumanEval、HumanEval+、MBPP、DS‑1000 四个公开代码生成基准，覆盖单函数实现、复杂边界、短任务和多语言项目。  
- **对比对象**：所有主流开源 Code LLM（如 StarCoder、CodeLlama、WizardLM 等），以及两款闭源大模型 Claude（Anthropic）和 Bard（Google）。  
- **成绩概览**：论文声称 WizardCoder 在 HumanEval 上的通过率领先所有开源模型数个百分点，并且在 HumanEval+、MBPP、DS‑1000 上同样保持领先。更重要的是，在 HumanEval 系列上，它的表现超过了 Claude 和 Bard，这在开源社区是首次实现的突破。  
- **消融实验**：作者分别去掉 Evol‑Instruct 的筛选环节、去掉多步骤指令、以及使用全参数微调而非 LoRA，实验显示每一项都会导致通过率下降 2%~5% 左右，说明进化筛选和低秩微调是关键贡献。  
- **局限性**：论文指出模型仍然在极端长代码或需要深度系统调用的任务上表现一般，且指令生成仍依赖于基线模型的质量，若基线模型本身有漏洞，生成的指令‑答案对也会受影响。

### 影响与延伸思考

WizardCoder 的成功证明，指令微调不再是自然语言模型的专利，代码模型同样可以通过进化式指令生成获得显著提升。自论文发布后，多个开源项目开始尝试把 Evol‑Instruct 或类似的自我生成指令框架搬到代码、数学推理甚至图像生成领域。后续工作可能会探索：

- **跨语言指令微调**：让模型同时掌握 Python、Java、C++ 等多语言指令，提升跨语言编程能力。  
- **安全与对齐**：在指令生成阶段加入安全检测，防止模型学习到有害的代码模式。  
- **人机协同指令**：结合真实开发者的反馈，进一步细化指令集合，使模型更贴合实际开发流程。  

如果想深入了解，可以关注“Evol‑Instruct”原始论文以及后续的 “Self‑Instruct” 系列，它们提供了更细致的自我生成指令的技术细节。

### 一句话记住它

让代码大模型自我出题、自动评测，再用高质量指令微调，WizardCoder 把开源模型的编程能力推到了超越闭源大模型的高度。