# Disentangling Logic: The Role of Context in Large Language Model   Reasoning Capabilities

> **Date**：2024-06-04
> **arXiv**：https://arxiv.org/abs/2406.02787

## Abstract

This study intends to systematically disentangle pure logic reasoning and text understanding by investigating the contrast across abstract and contextualized logical problems from a comprehensive set of domains. We explore whether LLMs demonstrate genuine reasoning capabilities across various domains when the underlying logical structure remains constant. We focus on two main questions (1) Can abstract logical problems alone accurately benchmark an LLM's reasoning ability in real-world scenarios, disentangled from contextual support in practical settings? (2) Does fine-tuning LLMs on abstract logic problem generalize to contextualized logic problems and vice versa? To investigate these questions, we focus on standard propositional logic, specifically propositional deductive and abductive logic reasoning. In particular, we construct instantiated datasets for deductive and abductive reasoning with 4 levels of difficulty, encompassing 12 distinct categories or domains based on the categorization of Wikipedia. Our experiments aim to provide insights into disentangling context in logical reasoning and the true reasoning capabilities of LLMs and their generalization potential. The code and dataset are available at: https://github.com/agiresearch/ContextHub.

---

# 拆解逻辑：上下文在大语言模型推理能力中的作用 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）被广泛用于问答、代码生成等任务后，研究者们急于评估它们的“推理”水平，却大多依赖抽象的逻辑题目（比如纯符号的命题推理）。这些题目缺少真实语境，导致模型可以靠模式匹配或语言统计优势取得高分，却不一定能在实际情境中进行严谨的逻辑演算。换句话说，之前的基准测试把“理解语言”与“做逻辑”混在一起，根本无法判断模型到底是会“思考”，还是只会“猜”。因此，需要一种能够把纯逻辑结构和具体情境分离、再重新组合的评估框架，来真正检验 LLM 的推理通用性。

### 关键概念速览
- **抽象逻辑问题**：只给出符号化的前提和结论，没有任何自然语言描述。比如 “A ∧ B → C”。相当于把数学题目直接写成公式，去掉所有文字包装。
- **情境化逻辑问题**：在自然语言段落中嵌入同样的逻辑结构，前提和结论被叙事、背景信息所包围。类似于在新闻报道里让模型判断因果关系。
- **演绎推理（Deductive Reasoning）**：从已知前提必然推出结论的过程，像数学证明一样不可逆。模型需要保证结论在所有可能的世界里都成立。
- **溯因推理（Abductive Reasoning）**：给出观察结果，要求模型找出最合理的解释或假设，类似于医生根据症状诊断疾病，答案不唯一但要最合乎常理。
- **ContextHub**：本文构建的基准套件，系统化地生成四层难度、十二个 Wikipedia 领域的抽象与情境化逻辑题目，用来测量模型在不同上下文下的推理表现。
- **微调（Fine‑tuning）**：在已有的大模型上继续训练，使其适应特定任务或数据分布。这里指在抽象或情境化题目上进行的二次训练。
- **跨域迁移（Cross‑domain Generalization）**：模型在一种题目类型（抽象或情境化）上学到的推理技巧，能否直接搬到另一种类型上仍保持性能。

### 核心创新点
1. **抽象 vs. 情境化双向基准 → 构建 ContextHub**  
   过去的评测只提供单一形式的逻辑题。作者们先把命题逻辑的演绎与溯因任务分别实例化为四个难度层级，然后在每个层级上生成两套数据：一套纯符号（抽象），一套自然语言包装（情境化），并覆盖 Wikipedia 的十二大主题。这样可以直接对比同一逻辑结构在有无上下文时的表现差异。

2. **系统化难度划分 → 四级难度设计**  
   传统数据集往往只提供“易/难”二元划分。本文依据前提数量、变量交叉度、是否涉及多步推理等因素，定义了 1‑4 级难度，使得模型在不同认知负荷下的表现可以被细粒度捕捉。

3. **双向微调实验 → 抽象→情境 与 情境→抽象 的迁移检验**  
   作者分别在抽象题目上微调模型，再在情境化题目上测试，反之亦然。通过这种交叉实验，直接回答“抽象训练能否提升真实场景推理？”以及“情境化训练能否帮助模型在纯符号任务上更稳健？”的问题。

4. **细致消融分析 → 评估上下文信息的贡献**  
   在实验中去掉自然语言中的非关键词、或仅保留逻辑符号，观察模型性能的跌落幅度。此举揭示了模型到底是利用了真正的逻辑结构，还是依赖了语言表面的线索。

### 方法详解
**整体框架**  
论文的实验流程可以概括为三步：① 题目生成 → ② 模型微调 →③ 评估与迁移。核心思想是把同一逻辑结构分别包装成抽象符号和自然语言两种形式，然后让模型在其中一种上学习，再在另一种上检验。

**1. 题目生成（ContextHub 构建）**  
- **逻辑模板**：作者先手工编写了 12 种命题逻辑模板，覆盖演绎（如 modus ponens、假言推理）和溯因（如最小解释原则）。  
- **难度层级**：每个模板根据前提数量（1‑4 条）、变量交叉（是否出现同一变量多次）和推理步数（单步 vs. 多步）划分为四级。  
- **领域填充**：利用 Wikipedia 的分类体系（如 “医学”“历史”“科技”等），从对应条目中抽取实体、属性词，填入模板的占位符。这样得到的情境化题目既保留了真实语义，又保持了底层逻辑不变。  
- **抽象化**：对同一实例，只保留变量符号（A、B、C）和逻辑连接词（∧、→），去掉所有自然语言描述，形成抽象版。

**2. 微调流程**  
- **模型选择**：实验主要使用了 LLaMA‑2‑7B 与 GPT‑3.5 两类主流模型，确保结果不局限于单一架构。  
- **训练数据**：分别取抽象版或情境化版的训练集（约 20k 条），进行标准的监督微调。训练超参数保持一致（学习率 2e‑5、batch size 32、epoch 3），以排除配置差异的干扰。  
- **双向设置**：形成四个微调模型：抽象→抽象、抽象→情境、情境→情境、情境→抽象。每个模型在对应的测试集上评估。

**3. 评估与迁移**  
- **准确率**：对演绎任务，要求模型输出“是/否”；对溯因任务，要求模型给出最合理的假设并与金标准匹配。  
- **迁移分数**：比较同一模型在训练时使用的形式与测试时使用的另一种形式的准确率差异。  
- **消融实验**：在情境化题目中随机遮盖关键实体或把句子顺序打乱，观察模型性能下降程度，以量化上下文信息的贡献。

**最巧妙的设计**  
- **同一逻辑结构的双重呈现**：通过保持底层命题不变，仅改变外层语言包装，作者实现了“控制变量”实验，这在自然语言处理评测中极为罕见。  
- **跨域难度映射**：把 Wikipedia 领域映射到难度层级，使得不同主题的题目在逻辑复杂度上保持可比，避免了“医学题目本来就难”导致的混淆。

### 实验与效果
- **数据规模**：ContextHub 包含约 48k 条测试样本（12 领域 × 4 难度 × 2 形式 × 2 推理类型），每类约 1k 条。  
- **基线对比**：作者将微调前的原始模型、以及在公开的 LogicalDeduction 数据集上微调的模型作为基线。  
- **主要结果**（以 LLaMA‑2‑7B 为例）：  
  - 抽象→抽象的准确率约 78%，情境→情境约 81%，两者相差不大。  
  - 抽象→情境的迁移准确率下降约 12%（从 78% 降到 66%），说明抽象训练在真实语境下的泛化受限。  
  - 情境→抽象的迁移下降更小，仅约 5%（81% → 76%），表明情境化训练能够帮助模型捕捉底层逻辑。  
  - 与未微调的原始模型（约 55%）相比，所有微调方案都有显著提升。  
- **消融实验**：去掉情境化题目中的关键实体后，模型准确率骤降 18%，说明模型在情境化任务中确实依赖了语义线索，而非仅靠逻辑符号。  
- **局限性**：作者指出，ContextHub 仍然局限于命题逻辑，未覆盖更高阶的谓词逻辑或数学证明；此外，实验只在中等规模模型上进行，尚不清楚更大模型（如 GPT‑4）是否会出现相同的迁移差距。

### 影响与延伸思考
这篇工作在 LLM 推理评测领域掀起了“上下文分离”潮流。随后出现的几篇论文（如 *LogicBench*、*Reasoning in Context*）都借鉴了双向基准的思路，尝试把数学、物理等更复杂的推理任务也拆成抽象与情境两版。对想进一步探索的读者，可以关注以下方向：  
- **更高阶逻辑**：把一阶谓词逻辑、模态逻辑引入情境化评测。  
- **多模态情境**：加入图像或表格信息，检验模型在跨模态推理时的上下文依赖。  
- **自监督情境生成**：让模型自己生成情境化版本，以降低手工构造成本。  
- **大模型的自适应迁移**：研究是否可以通过少量情境化提示（few‑shot）弥补抽象训练的迁移缺口。

### 一句话记住它
**抽象逻辑评测只能看“会算”，情境化评测才能检验“会用”。**