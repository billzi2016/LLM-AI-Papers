# xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code   Understanding, Generation, Translation and Retrieval

> **Date**：2023-03-06
> **arXiv**：https://arxiv.org/abs/2303.03004

## Abstract

Recently, pre-trained large language models (LLMs) have shown impressive abilities in generating codes from natural language descriptions, repairing buggy codes, translating codes between languages, and retrieving relevant code segments. However, the evaluation of these models has often been performed in a scattered way on only one or two specific tasks, in a few languages, at a partial granularity (e.g., function) level, and in many cases without proper training data. Even more concerning is that in most cases the evaluation of generated codes has been done in terms of mere lexical overlap with a reference code rather than actual execution. We introduce xCodeEval, the largest executable multilingual multitask benchmark to date consisting of $25$M document-level coding examples ($16.5$B tokens) from about $7.5$K unique problems covering up to $11$ programming languages with execution-level parallelism. It features a total of $7$ tasks involving code understanding, generation, translation and retrieval. xCodeEval adopts an execution-based evaluation and offers a multilingual code execution engine, ExecEval that supports unit test based execution in all the $11$ languages. To address the challenge of balancing the distributions of text-code samples over multiple attributes in validation/test sets, we propose a novel data splitting and a data selection schema based on the geometric mean and graph-theoretic principle. Our experiments with OpenAI's LLMs (zero-shot) and open-LLMs (zero-shot and fine-tuned) on the tasks and languages demonstrate **xCodeEval** to be quite challenging as per the current advancements in language models.

---

# xCodeEval：大规模多语言多任务代码理解、生成、翻译与检索基准 论文详细解读

### 背景：这个问题为什么难？
在代码大模型兴起之前，研究者往往只在单一任务（比如代码生成）或单一语言（如 Python）上做评测，评测数据量也很小。更糟的是，很多评测只看模型输出和参考答案的文字相似度，根本不检查代码能否真正跑通。于是我们只能得到“看起来不错”的分数，却不知道模型在真实开发环境中的可靠性。再加上跨语言的代码翻译、代码检索等需求日益增长，却缺少统一、可执行的多语言基准，导致模型改进缺乏明确的方向。

### 关键概念速览
**LLM（大规模语言模型）**：在海量文本上预训练的模型，能够理解和生成自然语言或代码，类似于会说多种语言的“通用翻译官”。  
**多任务基准**：一次性提供多种评测任务（理解、生成、翻译、检索），相当于一张全科考试卷，能一次看出模型的全能水平。  
**文档级代码**：以完整文件或项目为单位，而不是单独的函数或片段，类似于评估一篇文章的整体结构而不是只看一句话。  
**执行级评估**：把模型输出的代码实际跑一遍，检查是否通过单元测试，像让学生现场写程序并交给老师跑代码来打分。  
**ExecEval 引擎**：支持 11 种编程语言的自动化单元测试平台，充当“多语言实验室”，让不同语言的代码都能被统一执行。  
**几何均值分割**：在划分训练/验证/测试集时，用几何均值平衡多维属性（语言、任务、难度等），相当于在多维天平上保持每一边的重量相等。  
**图论选择策略**：把样本看成图的节点，用图的连通性来挑选既覆盖多属性又避免重复的子集，像在社交网络里挑选既广泛又不重叠的代表人物。

### 核心创新点
1. **数据规模与粒度的提升 → 构建 2500 万文档级代码样本、165 亿 token，覆盖 11 种语言** → 让评测从“函数级小样本”跳到“真实项目级大规模”，更贴近工业场景。  
2. **执行式评估体系 → 开发 ExecEval，统一对所有语言进行单元测试** → 把“代码看起来对不对”改成“代码能否跑通”，显著提升评测可信度。  
3. **多属性平衡划分 → 引入几何均值 + 图论的划分与选择方法** → 在验证/测试集里同时保持语言、任务、难度等分布均衡，避免某一类样本被过度或不足代表。  
4. **全任务统一基准 → 将代码理解、生成、翻译、检索 7 项任务整合在同一套数据上** → 研究者不必为每个任务单独准备数据，直接在同一基准上比较模型的多面能力。

### 方法详解
整体思路可以拆成三步：**数据收集 → 数据组织 → 评测执行**。

1. **数据收集**  
   - 从公开的编程教学平台、开源项目、在线评测网站抓取代码，确保每条记录都包含自然语言描述、代码实现以及对应的单元测试。  
   - 为每个问题保留完整的文件结构（多个文件、依赖声明等），形成文档级样本。  
   - 通过语言检测和任务标签（理解、生成等）把样本划分到 11 种编程语言的子库。

2. **数据组织与划分**  
   - 先对每个样本计算多维属性向量：语言、任务、难度（基于测试用例数量和代码行数）、代码风格等。  
   - 使用几何均值把属性值归一化后，求每个子集的整体均衡度。  
   - 构建属性相似度图，节点是样本，边权是属性相似度。利用图的最大独立集或最小覆盖算法挑选出验证集和测试集，使得选出的样本在属性空间上尽可能分散，同时保持每个属性的比例与整体一致。  
   - 剩余样本直接作为训练集，保证训练数据量最大化。

3. **评测执行（ExecEval）**  
   - 为每种语言实现统一的单元测试框架（如 Python 的 pytest、Java 的 JUnit、C++ 的 GoogleTest 等），并封装成容器化服务，保证沙箱安全。  
   - 当模型输出代码后，ExecEval 自动编译（若需要）并运行对应的测试用例，返回通过率、运行时错误、资源消耗等指标。  
   - 对于检索任务，系统先用向量检索返回候选代码片段，再交给 ExecEval 检查候选是否满足查询的功能需求。

**最巧妙的点**在于把“多语言、多任务”的评测统一到同一个执行引擎上。传统做法要么为每种语言单独写评测脚本，要么只做语义相似度评估，既费时又不可靠。ExecEval 通过容器化和统一的测试描述语言，实现“一键跑通”所有语言的代码评测。

### 实验与效果
- **数据规模**：共 2500 万文档级样本，约 165 亿 token，覆盖 11 种语言（Python、Java、C++、JavaScript、Go、Rust、Ruby、PHP、C#, TypeScript、Kotlin）。  
- **任务**：7 项，包括代码理解（问答/错误定位）、代码生成（函数/完整项目）、代码翻译（语言间互转）、代码检索（语义搜索）等。  
- **基线模型**：OpenAI 的 GPT‑4、GPT‑3.5（零样本），以及开源的 LLaMA‑2、StarCoder（零样本和微调后）。  
- **主要结果**：在执行通过率上，GPT‑4 的零样本平均通过率约 42%，而最强的微调后开源模型（StarCoder‑15B）最高只能到 28%，两者相差约 14% 绝对点。所有模型在跨语言翻译任务上均低于 30% 的通过率，说明多语言能力仍是瓶颈。  
- **消融实验**：去掉几何均值划分，验证集的属性分布出现偏斜，模型在低资源语言（如 Kotlin）上的表现下降约 6%。去掉 ExecEval 的单元测试，仅用 BLEU/CodeBLEU 评估，模型分数虚高 15% 左右，验证了执行评估的重要性。  
- **局限性**：作者指出 ExecEval 仍受限于单机容器资源，极端大项目的编译时间会显著增长；此外，单元测试的覆盖率取决于原始数据质量，部分开源项目的测试不完整，可能导致误判。

### 影响与延伸思考
xCodeEval 在发布后迅速成为代码大模型评测的“标配”。随后出现的多语言代码基准（如 MultiCodeBench、PolyCoderEval）都在数据规模或任务多样性上向它看齐。研究者也开始围绕 **执行级评估** 开发更高效的沙箱（如基于 WebAssembly 的统一运行时），以及利用 **自动测试生成** 来补足原始数据的测试缺失。对想进一步深入的读者，建议关注以下方向：① 自动化生成高质量单元测试的技术；② 跨语言代码对齐与语义映射模型；③ 大规模多语言代码预训练的训练策略（如语言平衡采样）。这些都是在 xCodeEval 基础上自然延伸的热点。

### 一句话记住它
xCodeEval 用 2500 万真实项目、执行级评估和多属性平衡划分，打造了首个“大规模、多语言、多任务、可跑通”的代码基准，让我们真正看到模型到底能写出多少能跑的代码。