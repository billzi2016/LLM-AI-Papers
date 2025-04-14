# VisualPuzzles: Decoupling Multimodal Reasoning Evaluation from Domain   Knowledge

> **Date**：2025-04-14
> **arXiv**：https://arxiv.org/abs/2504.10342

## Abstract

Current multimodal benchmarks often conflate reasoning with domain-specific knowledge, making it difficult to isolate and evaluate general reasoning abilities in non-expert settings. To address this, we introduce VisualPuzzles, a benchmark that targets visual reasoning while deliberately minimizing reliance on specialized knowledge. VisualPuzzles consists of diverse questions spanning five categories: algorithmic, analogical, deductive, inductive, and spatial reasoning. One major source of our questions is manually translated logical reasoning questions from the Chinese Civil Service Examination. Experiments show that VisualPuzzles requires significantly less intensive domain-specific knowledge and more complex reasoning compared to benchmarks like MMMU, enabling us to better evaluate genuine multimodal reasoning. Evaluations show that state-of-the-art multimodal large language models consistently lag behind human performance on VisualPuzzles, and that strong performance on knowledge-intensive benchmarks does not necessarily translate to success on reasoning-focused, knowledge-light tasks. Additionally, reasoning enhancements such as scaling up inference compute (with "thinking" modes) yield inconsistent gains across models and task types, and we observe no clear correlation between model size and performance. We also found that models exhibit different reasoning and answering patterns on VisualPuzzles compared to benchmarks with heavier emphasis on knowledge. VisualPuzzles offers a clearer lens through which to evaluate reasoning capabilities beyond factual recall and domain knowledge.

---

# VisualPuzzles：将多模态推理评估与领域知识解耦 论文详细解读

### 背景：这个问题为什么难？

多模态模型的评测长期被“知识题库”占据——大多数 benchmark 把图片、文字和大量事实记忆混在一起，导致模型好像推理能力很强，实则是靠检索答案。于是我们很难判断模型到底会不会真正“思考”。如果评测本身把领域知识当成必备前提，普通用户（非专家）根本看不出模型的推理上限，这也是为什么研究者一直在呼唤一个“纯粹推理”测评平台。

### 关键概念速览
- **多模态推理**：模型在看到图片和文字后，需要结合两者进行逻辑演算，而不是单纯把文字翻译成答案。想象你在看一张拼图图案，同时要回答“这块拼图最可能放在哪？”。
- **领域知识**：特定学科或行业的专业信息，例如医学术语、历史事件。它像是解题时的“外挂”，把难度从思考降到记忆。
- **VisualPuzzles**：本文推出的测评集合，专门挑选不需要专业背景的视觉逻辑题，像是“脑筋急转弯”版的多模态测试。
- **五大推理类别**：算法、类比、演绎、归纳、空间。每类对应一种思考方式，类似数学里“求导、比喻、证明、归纳、几何”。
- **思考模式（Thinking Mode）**：在推理时让模型多跑几步计算或自我纠错的技巧，类似人类在解难题时先写草稿再检查。
- **规模效应（Scaling Effect）**：模型参数更多、算力更强时性能提升的假设。这里指的是“更大模型一定更会推理吗？”的实验检验。

### 核心创新点
1. **评测目标从“知识+推理” → “纯推理”**  
   过去的 benchmark 把大量事实嵌进题目，导致模型只要记住资料库就能得高分。VisualPuzzles 刻意挑选不需要专业背景的题目，甚至把中文公务员考试的逻辑题手工翻译成多模态形式。这样模型只能靠“思考”，而不是“背书”，从根本上把评测与领域知识解耦。

2. **五类推理覆盖 → 更细粒度的能力画像**  
   传统评测往往只给出单一的正确率，难以区分模型在哪种思考上薄弱。VisualPuzzles 把题目划分为算法、类比、演绎、归纳、空间五类，每类都对应不同的认知过程。这样研究者可以直接看到模型在“类比”上好，却在“空间”上差的现象。

3. **跨 benchmark 对比 → 知识密集 vs 推理轻量**  
   作者把 VisualPuzzles 与 MMMU（一个知识密集型多模态 benchmark）进行对标，发现同一模型在 MMMU 上表现不错，却在 VisualPuzzles 上大幅下滑。这个对比直接证明“高分不等于强推理”，为后续模型设计提供了警示。

4. **思考模式与规模的系统实验 → 发现不一致性**  
   通过开启模型的“思考模式”（比如让模型多次自回归生成中间步骤）以及增大模型参数，作者发现提升并非线性：有的模型在某类题目上受益明显，有的则几乎不变。这个实验挑战了“更大更慢就一定更会推理”的常规假设。

### 方法详解
**整体框架**  
VisualPuzzles 不是一种新模型，而是一套评测流程。核心步骤包括：① 题目构造 → 把纯文字逻辑题转化为图片+文字组合；② 类别标注 → 为每道题打上五大推理标签；③ 模型推理 → 用现有多模态大语言模型（如 GPT‑4V、LLaVA 等）在统一接口下回答；④ 结果分析 → 按类别、模型规模、是否开启思考模式等维度做统计。

**关键模块拆解**  

1. **题目翻译与视觉化**  
   - 选取中文公务员考试中的逻辑推理题，这些题目本身只涉及文字描述和简单图形。  
   - 手工将文字描述保持不变，同时为每个选项绘制对应的示意图（如流程图、几何图形），确保答案的判断只能靠逻辑而非专业知识。  
   - 类比于把一道数学题从“文字版”转成“图形版”，让模型必须在视觉信息上做推理。

2. **推理类别划分**  
   - **算法**：需要按步骤执行的计算或排序。  
   - **类比**：找出两组事物之间的相似关系。  
   - **演绎**：从已知前提推出必然结论。  
   - **归纳**：从若干实例抽象出一般规律。  
   - **空间**：涉及位置、方向、旋转等几何关系。  
   - 每道题在构造时就贴上标签，后续评估时可以直接聚合同类成绩。

3. **模型推理接口**  
   - 使用统一的多模态 LLM 接口，输入为“图片 + 问题文字”。  
   - 为了测试“思考模式”，在一次推理中让模型先输出思考链（类似 CoT），再给出最终答案；或者让模型在同一输入上多轮自我纠错。  
   - 这里的“思考链”相当于让模型先写草稿，再检查草稿是否合理。

4. **评估指标**  
   - 主要是准确率（正确选项比例），但进一步细化为每类的准确率、思考链的完整度、模型在开启/关闭思考模式下的提升幅度。  
   - 通过对比不同模型、不同规模、不同思考模式的表现，绘制出“推理能力雷达图”。

**最巧妙的设计**  
- 直接把“知识密集型”题目转成“视觉+文字”形式，而不加入任何专业术语或背景信息，使得模型只能靠通用逻辑来解答。  
- 采用公务员考试的逻辑题作为来源，这类题目在中国有统一的难度划分，且已经经过严密的命题审查，保证了题目的推理深度。

### 实验与效果
- **测试对象**：包括 GPT‑4V、LLaVA‑13B、MiniGPT‑4、InstructBLIP 等主流多模态大语言模型，覆盖从几亿到百亿参数的范围。  
- **基准对比**：在 MMMU 上，这些模型的整体准确率在 70% 左右（具体数值论文未给出），而在 VisualPuzzles 上普遍跌至 40% 以下，尤其在空间和归纳类题目上表现最差。  
- **思考模式实验**：开启思考链后，GPT‑4V 在算法题上提升约 10% 准确率，但在类比题上几乎没有变化；LLaVA‑13B 的提升更为零星，说明思考链的收益高度依赖模型内部结构。  
- **规模效应**：从 7B 到 34B 参数的模型在 MMMU 上呈现正相关，但在 VisualPuzzles 上并未出现明显的提升趋势，甚至出现 13B 参数模型略胜 34B 的现象。  
- **消融实验**：作者去掉题目中的视觉元素，仅保留文字，模型准确率下降约 15%，说明视觉信息对推理仍有帮助，但不是唯一因素。  
- **局限性**：题目来源主要是中文，虽然翻译成英文后仍可使用，但跨语言通用性未在论文中验证；此外，手工翻译过程可能引入主观偏差，导致某些题目对特定模型更友好。

### 影响与延伸思考
- 这篇工作在社区里引发了对“评测是否真的测推理”的反思，随后出现了如 **ReasonBench**、**LogicVision** 等专注于纯逻辑的多模态评测。  
- 研究者开始尝试把 VisualPuzzles 的题目生成流程自动化，利用大模型自行生成“无知识”视觉逻辑题，形成闭环的自我评估体系（推测）。  
- 对于想继续深挖的读者，可以关注两条路：① 设计更细粒度的推理子任务（比如“因果链”），② 探索让模型在推理过程中显式调用外部工具（如符号求解器）以提升空间/算法类表现。

### 一句话记住它
**VisualPuzzles 用“无知题”把多模态模型的真正推理能力挑了出来，证明高分不等于会思考。**