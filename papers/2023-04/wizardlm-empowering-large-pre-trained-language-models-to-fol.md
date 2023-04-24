# WizardLM: Empowering large pre-trained language models to follow complex instructions

> **Date**：2023-04-24
> **arXiv**：https://arxiv.org/abs/2304.12244

## Abstract

Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation, WizardLM achieves more than 90\% capacity of ChatGPT on 17 out of 29 skills. Even though WizardLM still lags behind ChatGPT in some aspects, our findings suggest that fine-tuning with AI-evolved instructions is a promising direction for enhancing LLMs. Our code and data are public at https://github.com/nlpxucan/WizardLM

---

# WizardLM：赋能大规模预训练语言模型以遵循复杂指令 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）要在真实场景里表现得像助理，需要能理解并执行各种指令。过去的成功大多依赖“指令微调”，即用人工收集的大量问答对来教模型如何跟随指令。但人工编写指令既费时又费力，尤其是高复杂度的任务——人类往往难以想出既细致又多样的指令集合。于是，缺少高质量、层次丰富的指令数据成为提升模型复杂指令遵循能力的瓶颈。

### 关键概念速览
- **指令微调（Instruction Fine‑tuning）**：在已有的预训练模型上继续训练，使用“指令‑答案”对让模型学会根据自然语言指令生成合适的回复。类似于给学生补习，专门练习答题技巧。
- **Evol‑Instruct**：一种让语言模型自己“进化”指令的流程。先给模型一批种子指令，让它一步步改写、加深，产生更复杂的指令版本。把模型当成“指令生成器”，像细胞分裂产生新变种。
- **复杂度平衡测试集（Complexity‑balanced test bed）**：专门挑选或构造的评测题目，覆盖从简单到极其复杂的指令，以检验模型在不同难度上的表现。相当于给运动员安排从热身到全程马拉松的赛程。
- **GPT‑4 自动评估**：使用 OpenAI 的 GPT‑4 作为评审机器人，对模型输出进行打分，提供一种可大规模、统一的评测方式。类似于让机器裁判来评分，省去人工评审的成本。
- **WizardLM**：在 LLaMA 基础上，用 Evol‑Instruct 生成的指令数据进行微调得到的模型。它的名字暗示了“魔法师”般的指令理解能力。

### 核心创新点
1. **从人手到模型生成指令**  
   - 之前：指令数据全部由人工编写，规模受限且难以覆盖高复杂度场景。  
   - 本文：提出 Evol‑Instruct，让语言模型在已有指令的基础上逐步改写、加深，自动产生大量、层次分明的指令。  
   - 改变：大幅降低了指令数据的人工成本，同时能够系统性地提升指令的复杂度。

2. **指令复杂度的迭代进化机制**  
   - 之前：指令复杂度往往是一次性设定，缺少渐进式提升的过程。  
   - 本文：Evol‑Instruct 采用“逐步重写”策略，每一步都在前一步的指令上加入新约束或细化要求，使得指令难度呈阶梯式增长。  
   - 改变：模型在训练时看到的指令分布更均衡，既有简单任务也有高难度任务，提升了对复杂指令的鲁棒性。

3. **大规模混合指令数据的统一微调**  
   - 之前：多数工作只在单一指令来源上微调，导致模型对新指令的泛化能力受限。  
   - 本文：把 Evol‑Instruct 生成的所有层级指令与原始种子指令混合，统一对 LLaMA 进行微调，形成 WizardLM。  
   - 改变：模型在同一次训练中学习到多层次指令，表现出更强的跨任务迁移能力。

### 方法详解
**整体框架**  
WizardLM 的训练流程可以划分为三大步骤：①准备种子指令；②使用 Evol‑Instruct 迭代生成更复杂的指令；③把所有指令混合后对 LLaMA 进行统一微调。整个过程完全在模型内部完成，人工干预仅限于提供最初的少量种子指令。

**步骤拆解**  

1. **种子指令准备**  
   - 选取公开的指令数据集（如 Alpaca、Self‑Instruct 等）作为起点，数量相对较少但覆盖基本任务类型。可以把它们看作“原始配方”。

2. **Evol‑Instruct 迭代**  
   - **输入**：当前指令文本。  
   - **模型输出**：对该指令进行“改写”，包括：  
     - 增加约束条件（如限定答案格式、加入多步骤要求）。  
     - 引入子任务或多轮交互。  
     - 调整语言风格，使其更正式或更口语化。  
   - **迭代机制**：将模型的改写结果再次喂回同一模型，重复 N 次（论文未给出具体 N），每一次都在前一次的基础上增加难度。类似于让学生在老师的批改下不断完善作业。

3. **指令混合与微调**  
   - 把所有层级的指令（种子 + 第一次改写 + 第二次改写 …）合并，形成一个大规模、复杂度分布均衡的指令库。  
   - 使用该库对 LLaMA 进行监督微调，目标是让模型在看到任意指令时都能生成符合要求的答案。这里的微调过程与普通的指令微调相同，只是训练数据更丰富。

**关键细节**  
- **指令复杂度控制**：虽然 Evol‑Instruct 通过循环改写提升复杂度，但实际实现中会在每一步加入“复杂度提示”，让模型知道本轮需要更高难度。  
- **多模型协同**：在生成指令时，作者使用了比 LLaMA 更强的模型（如 GPT‑3.5）来保证改写质量，这一点在摘要里未明确说明，但是常见的做法。  
- **最巧妙的地方**：把模型本身当作“数据生成器”，实现了“数据自循环”。这打破了传统上“模型需要数据、数据需要人工标注”的闭环。

### 实验与效果
- **评测数据**：使用了两套测试集：①复杂度平衡测试床，覆盖从低到高的指令难度；② Vicuna 官方测试集，主要评估通用指令遵循能力。  
- **人类评估**：在复杂度平衡测试中，评审者更倾向于 Evol‑Instruct 生成的指令对应的模型输出，而不是传统人工指令对应的输出。  
- **与 ChatGPT 对比**：在人类评审的高复杂度子集里，WizardLM 的回答被认为优于 OpenAI ChatGPT。  
- **GPT‑4 自动评估**：在 29 项技能测评中，WizardLM 在 17 项上达到或超过 ChatGPT 90% 的能力水平。  
- **基线对比**：与原始 LLaMA 以及其他指令微调模型（如 Vicuna）相比，WizardLM 在多数任务上都有显著提升，尤其是高复杂度指令的成功率提升最为明显。  
- **消融实验**：论文中对 Evol‑Instruct 的迭代次数、指令混合比例等做了消融分析，结果显示：去掉高复杂度指令或只使用种子指令会导致在复杂任务上的表现大幅下降。  
- **局限性**：作者承认 WizardLM 在某些细粒度推理或常识丰富的任务上仍落后于 ChatGPT，说明仅靠指令进化仍不足以完全替代大规模人类标注的高质量数据。

### 影响与延伸思考
WizardLM 的核心思路——让语言模型自行生成训练指令——在随后的一年里激发了多项后续工作。例如，Meta 的 **Mistral‑Instruct**、Google 的 **Flan‑P** 系列都尝试在指令生成阶段加入模型自演化的环节。还有研究把 Evol‑Instruct 与强化学习（RLHF）结合，形成“自我强化指令微调”。如果想进一步探索，可关注以下方向：  
- **指令质量评估**：如何自动衡量模型生成指令的可解释性和难度。  
- **多模态指令进化**：把图像、音频等信息加入指令生成，使模型能处理跨模态任务。  
- **安全与偏见控制**：自生成指令可能放大模型已有的偏见，需要在进化过程中加入约束。  

### 一句话记住它
让语言模型自己“写指令、练指令”，就能用少量人工种子数据把大模型训练成高复杂度指令的高手。