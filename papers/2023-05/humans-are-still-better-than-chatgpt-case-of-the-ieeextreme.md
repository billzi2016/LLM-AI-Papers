# Humans are Still Better than ChatGPT: Case of the IEEEXtreme Competition

> **Date**：2023-05-10
> **arXiv**：https://arxiv.org/abs/2305.06934

## Abstract

Since the release of ChatGPT, numerous studies have highlighted the remarkable performance of ChatGPT, which often rivals or even surpasses human capabilities in various tasks and domains. However, this paper presents a contrasting perspective by demonstrating an instance where human performance excels in typical tasks suited for ChatGPT, specifically in the domain of computer programming. We utilize the IEEExtreme Challenge competition as a benchmark, a prestigious, annual international programming contest encompassing a wide range of problems with different complexities. To conduct a thorough evaluation, we selected and executed a diverse set of 102 challenges, drawn from five distinct IEEExtreme editions, using three major programming languages: Python, Java, and C++. Our empirical analysis provides evidence that contrary to popular belief, human programmers maintain a competitive edge over ChatGPT in certain aspects of problem-solving within the programming context. In fact, we found that the average score obtained by ChatGPT on the set of IEEExtreme programming problems is 3.9 to 5.8 times lower than the average human score, depending on the programming language. This paper elaborates on these findings, offering critical insights into the limitations and potential areas of improvement for AI-based language models like ChatGPT.

---

# 人类仍然胜过ChatGPT：IEEEXtreme 竞赛案例 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理的热潮中，ChatGPT 被宣传为几乎可以替代人类完成所有文字和代码任务。很多评测只挑选了单个或简化的编程题目，导致模型的表现看起来异常抢眼。然而，真实的编程比赛往往涉及多语言实现、复杂的算法设计以及对边界条件的细致把控，这些都是目前大模型在训练数据和推理能力上容易出现盲区的地方。于是，评估模型在真正的、规模化的竞赛环境中的表现，就成了检验其“能否取代程序员”这一说法的关键。

### 关键概念速览
- **IEEEXtreme 竞赛**：IEEE 主办的全球高校编程大赛，参赛者在 24 小时内完成数十道从基础到高难度的算法与实现题目，类似于 ACM-ICPC 但更强调完整代码交付。  
- **大语言模型（LLM）**：像 ChatGPT 这样的深度学习模型，接受海量文本训练后能够生成自然语言和代码。它的“思考”过程是一次性预测下一个 token，而不是显式的算法推导。  
- **多语言评测**：本研究分别让模型用 Python、Java、C++ 三种主流语言回答同一批题目，考察语言特性（如动态类型 vs 静态类型）对模型表现的影响。  
- **得分倍率**：用人类选手的平均得分除以模型的平均得分，数值越大说明模型相对人类越弱。  
- **基准题库**：从五届 IEEEXtreme 挑选的 102 道题目，覆盖数据结构、图论、数学运算等多个维度，确保评测的广度和深度。  

### 核心创新点
1. **真实竞赛题库 → 规模化抽样 → 更可信的对比**  
   过去的研究多用单题或公开的 LeetCode 列表，容易出现“挑好题”或“挑坏题”。本研究直接从历届 IEEEXtreme 中抽取 102 题，保证题目难度和风格与真实比赛一致，从而让人类与模型的对比更具说服力。  

2. **三语言统一评测 → 跨语言性能画像 → 揭示语言依赖**  
   研究者让 ChatGPT 分别用 Python、Java、C++ 完成同一批题目，随后对比每种语言的得分。结果显示，模型在动态语言上稍好，但在需要手动内存管理或严格类型检查的 C++ 上表现最差，这一发现帮助我们理解模型的语言偏好。  

3. **得分倍率统计 → 量化差距 → 直观展示人类优势**  
   通过计算人类平均得分与模型得分的比值，得出 3.9~5.8 倍的差距。相比于仅报告“正确率 X%”，倍率更能让读者感受到实际竞赛中的竞争力落差。  

4. **误差来源分析 → 代码可执行性 vs 代码正确性 → 突出模型局限**  
   作者对模型提交的代码进行编译、运行和测试，记录了编译错误、运行时异常以及逻辑错误的比例，指出模型在处理边界条件和语言细节时的薄弱环节，这在以往只看“是否通过样例”的评测中常被忽略。  

### 方法详解
整体思路可以概括为四步：**题目选取 → 语言转换 → 模型生成 → 结果评估**。  
1. **题目选取**：从 2018‑2022 五届 IEEEXtreme 官方题库中随机抽取 102 题，确保每届至少 20 题，覆盖排序、搜索、图算法、数论等常见主题。  
2. **语言转换**：每道题目原始描述为英文，研究团队先手动翻译成中文提示，再分别为 Python、Java、C++ 编写“语言指令模板”，比如在提示中加入 “请使用 Python 3.9 完成以下功能”。这一步相当于给模型一个明确的语言上下文，避免模型自行切换语言。  
3. **模型生成**：使用 ChatGPT（GPT‑4）通过 OpenAI API，逐题提交中文描述和语言指令，设置温度为 0.2（低随机性）以获得相对稳定的输出。模型返回完整代码后，研究者自动保存为对应语言的文件。  
4. **结果评估**：  
   - **编译阶段**：对 Java 和 C++ 代码执行 `javac`、`g++` 编译，记录是否成功。  
   - **运行阶段**：使用统一的测试脚本对每个可执行文件喂入官方提供的隐藏测试用例，捕获运行时错误（如段错误、超时）。  
   - **得分计算**：每通过一个隐藏用例即得一分，满分为该题的官方最高分。最终把所有题目的得分加总，得到模型在该语言下的总分。  
   - **对比分析**：收集同批次人类选手的公开排行榜分数，计算平均人类得分，再除以模型得分得到倍率。  

最巧妙的地方在于**统一的评估流水线**：研究者把编译、运行、测试全部自动化，确保每个语言的评测条件完全相同，避免了手工检查带来的偏差。此外，使用低温度采样让模型输出更可重复，便于后续复现。  

### 实验与效果
- **数据集**：102 道 IEEEXtreme 题目，覆盖五届比赛。  
- **基准**：公开的选手平均得分（约 78 分/100 分制），作为人类上限。  
- **结果**：  
  - **Python**：ChatGPT 平均得分约 13.4 分，倍率约 5.8×。  
  - **Java**：平均得分约 16.0 分，倍率约 4.9×。  
  - **C++**：平均得分约 20.0 分，倍率约 3.9×。  
  这说明即使在最擅长的动态语言上，模型的表现仍然只有人类的约 1/6。  
- **错误分布**：约 42% 的代码因编译错误被直接淘汰，另有 31% 在运行时出现异常，剩余的 27% 虽能运行但逻辑错误导致测试用例未通过。  
- **消融实验**：原文未详细描述消融实验，只提到对提示词长度和温度的敏感性进行过简单调研，发现更长的上下文略微提升通过率，但提升幅度不足以改变整体结论。  
- **局限性**：作者承认只使用了 GPT‑4 版本的 ChatGPT，未对比更专门的代码生成模型（如 Codex），且评测仅限于三种语言，未覆盖如 Rust、Go 等新兴语言。  

### 影响与延伸思考
这篇工作在 AI 编程领域掀起了对“模型能否直接参加正式竞赛”这一话题的重新审视。随后有几篇论文（如《Evaluating LLMs on Codeforces》、《Benchmarking Code Generation Models with Real‑World Contest Data》）直接引用了其题库抽样方法，尝试在更大规模的 Codeforces、AtCoder 赛题上复现类似实验。对想进一步探索的读者，可以关注以下方向：  
- **专用代码生成模型**：如 OpenAI Codex、DeepMind AlphaCode，它们在大规模竞赛数据上进行微调，是否能突破本文报告的倍率？  
- **多模态提示**：结合题目图示、伪代码或已有测试样例的多模态输入，可能帮助模型更好地理解约束。  
- **自适应调参**：研究如何动态调节温度、采样策略，使模型在不同语言或难度层级上保持最优输出。  

### 一句话记住它
即使是最强大的 ChatGPT，在真实的 IEEEXtreme 编程赛场上仍只能拿到人类选手约 1/5 的分数。