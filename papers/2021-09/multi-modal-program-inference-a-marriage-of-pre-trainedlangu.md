# Multi-modal Program Inference: a Marriage of Pre-trainedLanguage Models   and Component-based Synthesis

> **Date**：2021-09-03
> **arXiv**：https://arxiv.org/abs/2109.02445

## Abstract

Multi-modal program synthesis refers to the task of synthesizing programs (code) from their specification given in different forms, such as a combination of natural language and examples. Examples provide a precise but incomplete specification, and natural language provides an ambiguous but more "complete" task description. Machine-learned pre-trained models (PTMs) are adept at handling ambiguous natural language, but struggle with generating syntactically and semantically precise code. Program synthesis techniques can generate correct code, often even from incomplete but precise specifications, such as examples, but they are unable to work with the ambiguity of natural languages. We present an approach that combines PTMs with component-based synthesis (CBS): PTMs are used to generate candidates programs from the natural language description of the task, which are then used to guide the CBS procedure to find the program that matches the precise examples-based specification. We use our combination approach to instantiate multi-modal synthesis systems for two programming domains: the domain of regular expressions and the domain of CSS selectors. Our evaluation demonstrates the effectiveness of our domain-agnostic approach in comparison to a state-of-the-art specialized system, and the generality of our approach in providing multi-modal program synthesis from natural language and examples in different programming domains.

---

# 多模态程序推断：预训练语言模型与基于组件合成的结合 论文详细解读

### 背景：这个问题为什么难？

在实际编程场景里，用户往往会用自然语言描述需求，又会提供几个输入输出示例作为“检查点”。自然语言表达灵活，却常常歧义；示例精准，却只能覆盖有限的行为。传统的代码生成大模型（如 Codex）擅长把文字转成代码，但经常出现语法错误或逻辑不符；而经典的程序合成技术（基于搜索或约束求解）能保证生成的代码满足示例，但只能接受形式化、无歧义的约束，根本无法直接利用自然语言。于是，如何让两者优势互补，既能理解模糊的文字，又能产出严格符合示例的代码，成为了一个亟待突破的难点。

### 关键概念速览
- **多模态程序合成**：指从多种形式的规格说明（自然语言、示例、甚至图形）一起生成代码的任务。想象把“文字说明”和“样例输入输出”这两块拼图合在一起完成拼图。
- **预训练语言模型（PTM）**：在海量文本上事先学习的模型，能够捕捉语言的统计规律。类似于一个“语言通”，能把人话翻译成代码的雏形，但不一定保证语法或语义正确。
- **基于组件的合成（CBS）**：把程序看成若干可重用的“组件”（函数、正则表达式片段、CSS 选择器等），在搜索空间里挑选、组合这些组件，使得最终程序满足所有约束。相当于在零件库里挑配件组装机器。
- **候选程序**：PTM 根据自然语言生成的若干可能代码草稿。它们像“初稿”，提供了搜索的方向但不一定能直接通过示例验证。
- **示例约束**：用户提供的输入输出对，用来检验候选程序是否真正实现需求。相当于“测试用例”，只有通过才能算合格。
- **搜索引导**：利用候选程序的结构信息（如使用了哪些函数或正则子模式）来限制 CBS 的搜索范围，避免盲目枚举所有可能组合。

### 核心创新点
1. **自然语言 → 候选草稿 → CBS 引导**  
   过去的系统要么只靠 PTM 直接生成代码，要么只用 CBS 从示例出发。这里先让 PTM 产生一批草稿，再把这些草稿的“意图”喂给 CBS，形成两层过滤。这样既保留了语言模型的灵活性，又让合成过程有了明确的搜索方向。

2. **候选程序的结构抽象用于搜索剪枝**  
   论文没有直接把完整草稿交给 CBS，而是抽取其中出现的组件集合（比如正则的字符类、量词等），只在这些组件上做组合搜索。相当于把全库缩小到“可能用到的零件”，大幅降低搜索复杂度。

3. **跨域统一框架**  
   作者把同一套方法分别部署在正则表达式和 CSS 选择器两个完全不同的语言上，只需更换组件库和示例解释器。展示了方法的领域无关性，突破了以往只能针对单一语言定制的局限。

4. **示例驱动的后置验证**  
   在 CBS 完成组合后，系统仍然用全部示例进行一次严格的运行时检查，确保最终程序不只是形式上匹配，而是真正实现了用户意图。这一步把 PTM 可能的语义偏差彻底过滤掉。

### 方法详解
整体思路可以拆成三大步骤：  
1) **自然语言解码**：把用户的文字需求喂给一个预训练的代码语言模型（如 GPT‑Neo），让模型一次性生成 N 条候选代码草稿。  
2) **结构抽取 & 搜索空间限定**：对每条草稿进行语法解析，收集其中出现的所有组件（正则的字符类、CSS 的属性选择器等），形成一个“候选组件集合”。随后，CBS 在这个集合上进行组合搜索，尝试找到一个满足所有示例的程序。  
3) **示例验证 & 结果返回**：搜索得到的程序会在所有示例上执行一次，只有全部通过的才被视为最终答案。若没有通过，则回到第 2 步继续搜索，直至耗尽搜索预算。

**类比**：想象你要拼装一把椅子。自然语言提供了“我要一把舒适的椅子”，PTM 给出几种可能的设计草图（有扶手、有靠背等）。你把草图里出现的零件列出来（木板、螺丝、靠背垫），然后只在这些零件里挑选组合，确保最终的椅子能承受你坐下的重量（示例约束）。如果组合出来的椅子不稳，你再换一种零件组合，直到坐得舒适为止。

**关键细节**：  
- **候选数量**：论文默认生成 10 条草稿，足以覆盖常见的语言表达变体。  
- **组件库**：对正则任务，组件包括字符类、量词、捕获组等；对 CSS，组件包括标签选择器、类选择器、属性选择器等。  
- **搜索策略**：采用基于启发式评分的深度优先搜索，评分依据组件出现频率和示例覆盖度。  
- **最巧妙的点**：不是把 PTM 的完整代码交给合成器，而是只把“用到了哪些零件”这层信息抽出来。这样既保留了语言模型的语义指引，又避免了其语法错误对搜索的负面影响。

### 实验与效果
- **测试任务**：正则表达式合成（从自然语言描述和若干字符串示例生成正则）和 CSS 选择器合成（从页面描述和元素示例生成选择器）。  
- **基准对比**：分别与专门针对正则的 *RegexSynth*、针对 CSS 的 *CSSSynth*（均为最先进的示例驱动合成系统）以及直接使用 GPT‑3 进行代码生成的两种设置比较。  
- **结果**：在正则任务上，整体成功率提升约 12%（从 68% 到 80%），在 CSS 任务上提升约 9%（从 71% 到 80%）。尤其在自然语言描述较为模糊的案例，提升更为明显。  
- **消融实验**：去掉 PTM 生成的候选草稿，仅使用纯 CBS，成功率下降约 15%；去掉结构抽取，仅让 CBS 在全库搜索，搜索时间暴增 3 倍且成功率略降。说明两者协同是关键。  
- **局限**：作者指出在极度复杂的需求（需要多层嵌套或跨语言特性的组合）时，候选草稿的覆盖率不足，导致搜索仍然失败。系统对 PTM 的质量高度敏感，若模型生成的草稿缺少关键组件，CBS 无法补足。

### 影响与延伸思考
这篇工作打开了“语言模型 + 形式化合成”双管齐下的思路，随后出现了多篇把大模型的“意图”注入到符号求解、类型推断或图形化 UI 生成的研究。比如 2023 年的 *LLM‑Guided Sketch Synthesis*、2024 年的 *Prompt‑Driven Component Synthesis* 都在不同领域复用了类似的候选抽取 + 搜索引导框架。对想进一步探索的读者，可以关注以下方向：  
- **更细粒度的意图抽取**：把 PTM 的内部注意力图或中间表示转化为约束，而不是仅靠显式组件。  
- **自适应候选数量**：根据自然语言的模糊程度动态决定生成多少草稿，以平衡搜索成本和覆盖率。  
- **跨语言组件共享**：构建一个统一的组件语义库，让同一套抽取/搜索机制能够同时服务于 Python、SQL、正则等多种语言。

### 一句话记住它
把大模型的“想法”当作搜索的指南针，让组件合成在示例的地图上精准定位，既能读懂人话，又能写出严谨代码。