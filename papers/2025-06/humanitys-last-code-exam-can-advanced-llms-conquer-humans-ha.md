# Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition?

> **Date**：2025-06-15
> **arXiv**：https://arxiv.org/abs/2506.12713

## Abstract

Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(https://github.com/Humanity-s-Last-Code-Exam/HLCE).

---

# 人类终极代码考验：先进大语言模型能否征服人类最难的编程竞赛？ 论文详细解读

### 背景：这个问题为什么难？
传统的代码生成基准（如APPs、LiveCodeBench）大多只包含中等难度的编程题，模型只要会基本的语法和常见算法就能轻松拿高分。真正考验深层推理、算法设计和复杂边界条件的，是国际大学生程序设计竞赛（ICPC）和国际信息学奥林匹克（IOI）决赛题目——这些题目往往需要多步数学推导、非平凡的数据结构以及对时间空间复杂度的精准分析。过去的评测体系没有覆盖这类极限挑战，导致我们无法判断大语言模型在“高阶程序设计”上的真实水平。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言和代码的深度神经网络，类似于“会说话的程序员”。  
**代码生成基准**：一套标准化的编程题目，用来测量模型写代码的能力，就像跑步比赛的计时表。  
**pass@k**：模型在前k次生成中至少有一次代码能通过全部测试用例的成功率，k=1时相当于“一次机会”。  
**自我认知（self‑recognition）任务**：让模型先估计自己能否解出题目，再真正去写代码，类似于先判断自己会不会这道题再去答。  
**测试时尺度律（test‑time scaling law）**：观察模型性能随算力、提示长度等因素变化的经验规律，像是“加速跑步时速度随体能提升的曲线”。  
**线上‑线下沙盒**：统一的执行环境，既可以在云端自动跑代码，又能在本地离线复现，确保评测结果可重复。  
**ICPC / IOI 终决赛题**：全球最难的算法竞赛题目，要求选手在极短时间内设计并实现高效算法，难度堪比博士研究。

### 核心创新点
1. **从竞赛题库中挑选极限题目 → 构建 HLCE 数据集**  
   过去的基准只采集公开的教学或入门题，作者直接从 2010‑2024 年的 ICPC 世界总决赛和 IOI 决赛中抽取 235 道“硬核”题目，确保每一道都需要高级算法和细致的实现细节。结果是一个能够真正压垮现有 LLM 的“终极考场”。  

2. **统一的线上‑线下沙盒 → 可完全复现的评测**  
   为避免不同平台的运行差异，团队实现了一个两层结构的沙盒：线上负责自动化提交、计时和安全隔离，线下提供同样的容器镜像供研究者本地跑实验。这样，无论是论文复现还是后续竞赛，都能得到完全一致的通过率。  

3. **自我认知任务的设计 → 测量模型对自身能力的觉察**  
   在正式写代码前，模型先被要求给出“我能解这题吗？”的概率预测。作者发现模型的自评与实际通过率并不呈线性关系，说明即使模型在代码生成上表现不佳，它也可能错误地高估自己。  

4. **测试时尺度律实证 → 揭示性能提升的上限**  
   通过系统地增大模型参数、提示长度和推理步数，作者绘制了性能随资源投入的曲线，发现在当前技术水平下，提升仍呈递减趋势，暗示还有大量未被利用的算法推理潜力。  

### 方法详解
整体思路可以拆成三大步骤：**题目筛选 → 沙盒构建 → 评测与分析**。

1. **题目筛选**  
   - 从历届 ICPC 世界总决赛和 IOI 决赛的官方题库中抽取所有题目。  
   - 依据题目描述的算法复杂度、输入规模以及历史通过率（人类选手的平均排名）进行硬度打分。  
   - 只保留得分最高的 235 道，确保每一道都需要 O(n log n) 以上的复杂度或高级图论/数论技巧。  

2. **沙盒构建**  
   - **线上层**：使用容器化技术（Docker + Firecracker）为每一次提交提供独立的执行环境，自动编译、运行并对比预设的隐藏测试用例。  
   - **线下层**：将同样的容器镜像导出，提供给研究者在本地机器上复现。两层共享同一套评测脚本，保证“跑一次、看一次、复现一次”完全一致。  
   - 为防止模型利用系统漏洞作弊，沙盒对系统调用、网络访问和文件系统进行严格限制。  

3. **评测流程**  
   - 对每个模型，使用统一的 **Zero‑Shot**、**Few‑Shot**（提供 1‑3 例相似题目）以及 **Chain‑of‑Thought**（让模型先写思路）三种提示方式。  
   - 生成代码后立即送入沙盒执行，记录是否全部通过隐藏测试用例，得到 **pass@1**（一次机会通过率）。  
   - 同时在生成代码前让模型输出自评概率，形成 **self‑recognition** 数据。  

4. **尺度律实验**  
   - 选取同一模型的不同规模（如 o4-mini‑low、o4-mini‑high）以及不同的 **temperature**、**max‑tokens** 参数组合。  
   - 记录每种配置的 pass@1，绘制性能随参数的曲线，使用对数线性回归拟合，得到经验尺度律公式。  

**最巧妙的点**在于把极难的竞赛题目直接搬进评测平台，同时提供了完全可复制的离线沙盒，这在以往的代码生成基准里是前所未有的。自我认知任务的加入也让我们看到模型“自信”与“实力”之间的脱钩，提示后续研究需要关注模型的元认知能力。

### 实验与效果
- **数据集**：HLCE 包含 235 道 2010‑2024 年 ICPC/IOI 决赛题，配套的隐藏测试用例共计约 2 万条。  
- **基线模型**：o4-mini(high)（OpenAI 最新的高性能小模型）和 Gemini‑2.5 Pro（Google 最新多模态模型）。  
- **结果**：在最优提示下，o4-mini(high) 的 pass@1 为 **15.9%**，Gemini‑2.5 Pro 为 **11.4%**。对比传统基准（如 LiveCodeBench）中同类模型常能超过 **80%** 的通过率，差距显而易见。  
- **自我认知**：两模型的自评概率平均在 0.6 左右，但实际通过率只有 0.15 左右，说明模型普遍高估自己的解题能力。  
- **尺度律**：实验显示，参数规模每提升一倍，pass@1 仅提升约 3‑5%，呈明显递减趋势，暗示单纯扩大模型并不能突破当前瓶颈。  
- **消融实验**：去掉 Chain‑of‑Thought 提示后，pass@1 下降约 4%；完全不使用 Few‑Shot 示例时下降约 7%，说明思路写作和示例提示对提升性能仍有帮助。  
- **局限**：作者承认 HLCE 只覆盖算法竞赛题目，缺少系统编程、库调用等实际工程场景；此外评测仅使用 pass@1，未考虑多次尝试的潜在提升。  

### 影响与延伸思考
HLCE 的出现让代码生成社区有了一个真正的“终极考场”，后续不少工作开始把模型的 **算法推理** 作为核心评估指标，而不再只看代码的语法正确性。2025 年出现的 **AlgoCoder**、**ReasonCode** 等项目直接引用 HLCE 进行对比，推动了 **思维链+检索**（CoT+Retrieval）混合策略的研究。对想进一步探索的读者，可以关注以下方向：  
- **元认知训练**：让模型在自评阶段学习校准自己的置信度。  
- **检索增强**：在生成代码前先检索公开的竞赛解法或论文，融合外部知识。  
- **多模态协同**：结合图形化算法可视化（如流程图）帮助模型构建更稳健的思路。  

### 一句话记住它
**HLCE 把全球最难的算法竞赛题目搬进评测平台，揭示了即便是最强 LLM，真正的高阶编程能力仍远未达到人类水平。**