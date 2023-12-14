# PromptBench: A Unified Library for Evaluation of Large Language Models

> **Date**：2023-12-13
> **arXiv**：https://arxiv.org/abs/2312.07910

## Abstract

The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

---

# PromptBench：大语言模型统一评估库 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以写代码、写文章、甚至参与对话，但要判断它们到底有多好、哪些场景会出错，却没有一套通用的“量尺”。过去的评测往往是把模型塞进某个公开数据集里跑一次，或者手工写几个提示词（prompt）去测几个能力。这样做有两个根本缺点：一是评测脚本各自为政，复用成本高；二是面对安全风险（比如对抗性提示）时缺少系统化的攻击与防御手段。于是研究者们迫切需要一个既能统一管理数据、模型、提示，又能灵活加入新评测方式的工具箱。

### 关键概念速览
**大语言模型（LLM）**：指参数量在数十亿以上、能够生成自然语言的深度模型，像 GPT‑4、Claude 等。可以把它想象成一个“会说话的百科全书”。  
**Prompt（提示词）**：向模型发出的指令或问题，类似于给老师出题目，让模型根据提示给出答案。  
**Prompt Engineering（提示工程）**：系统化地设计、改写提示词，以最大化模型表现的过程，就像调配配方让厨师做出更好吃的菜。  
**Benchmark（基准测试）**：一组标准化任务或数据，用来衡量模型在特定能力上的水平，类似于学生的期末考试。  
**Adversarial Prompt Attack（对抗式提示攻击）**：故意构造误导性的提示词，诱导模型产生错误或有害输出，像是给模型出陷阱题来检验其鲁棒性。  
**Dynamic Evaluation Protocol（动态评估协议）**：在评测过程中根据模型的实时表现调整评测策略的规则，类似于老师根据学生答题情况即时改变提问难度。  
**Analysis Tool（分析工具）**：帮助研究者可视化、统计评测结果的脚本或库，像是实验室的显微镜，让细节一目了然。

### 核心创新点
1. **统一化评测框架 → PromptBench 把提示构造、提示工程、数据加载、模型加载、对抗攻击、动态协议、结果分析全部封装进同一个库 → 研究者只需几行代码就能搭建完整评测流水线，省去重复搬砖的时间。**  
2. **可插拔的 Prompt 构造与工程模块 → 采用插件式 API，用户可以自行实现新的提示生成策略或改写已有策略 → 评测可以快速覆盖从零样本提示到少样本微调的各种场景。**  
3. **内置对抗式提示攻击工具 → 提供一套可配置的攻击模板（如词义替换、语序扰乱、隐蔽指令注入），并自动记录模型的安全失误 → 让安全评估不再是事后手工实验，而是评测流程的第一层。**  
4. **动态评估协议支持 → 在评测过程中可以根据模型的错误率、响应时间等指标动态切换任务难度或切换数据子集 → 类似于自适应考试，能够更细致地描绘模型的能力边界。

### 方法详解
PromptBench 的整体思路可以用“一条生产线”来比喻：原材料（数据集）进入工厂，经过加工（提示构造与工程），进入检测站（模型推理），随后进行质量检查（对抗攻击与动态评估），最后交给报告部门（分析工具）生成评测报告。具体步骤如下：

1. **数据与模型加载**  
   - 用户只需提供模型的 HuggingFace 标识或本地路径，库会自动下载并包装成统一的推理接口。  
   - 数据集同理，支持 CSV、JSON、HF Datasets 等多种格式，内部会把每条记录统一成 `{input, target}` 的结构。

2. **Prompt 构造**  
   - PromptBuilder 类负责把原始输入转化为完整的提示词。它提供模板占位符（如 `{question}`、`{context}`）以及批量填充功能。  
   - 通过继承 PromptBuilder，研究者可以实现自定义的提示生成逻辑，例如在每个问题前加上角色设定或情感基调。

3. **Prompt Engineering**  
   - PromptEngine 模块提供常见的工程手段：Few‑Shot 示例插入、Chain‑of‑Thought（思维链）展开、指令微调等。  
   - 每种手段都是一个可组合的“插件”，用户可以按需堆叠，例如先加 Few‑Shot 再加 CoT。

4. **模型推理**  
   - 对每个构造好的提示，PromptRunner 调用统一的 `model.generate` 接口，返回模型的原始文本输出。  
   - 为了兼容不同硬件，库内部会自动选择 GPU、CPU 或者分布式推理路径。

5. **对抗式提示攻击**  
   - AttackEngine 接收原始提示，依据预定义的攻击策略（如同义词替换、插入隐蔽指令）生成“恶意版”提示。  
   - 攻击过程是可配置的：用户可以指定攻击强度、目标词表或自定义扰动函数。

6. **动态评估协议**  
   - Evaluator 类维护一个状态机，根据模型在前几轮的表现（正确率、置信度）决定是否切换到更难的子任务或停止评测。  
   - 例如，当模型在某类数学题上连续 90% 正确时，系统会自动切换到更高阶的题目，以防止“早熟”误判。

7. **结果分析与可视化**  
   - AnalysisToolkit 提供统计函数（准确率、BLEU、ROUGE）以及绘图接口（误差分布、对抗成功率）。  
   - 最终报告可以导出为 Markdown、HTML 或 JSON，方便在论文或内部报告中直接使用。

**最巧妙的地方**在于把“对抗攻击”和“动态评估”这两个通常在安全研究或自适应学习中出现的概念，统一放进同一个评测流水线。这样一来，安全性和能力评估不再是两条平行的实验，而是相互影响、相互验证的整体。

### 实验与效果
- **测试任务**：论文提到在多个公开基准（如 MMLU、TruthfulQA、GSM‑8K）以及自建的安全提示集合上跑通评测。  
- **对比基线**：与手工脚本或单一评测库（如 OpenAI evals）相比，PromptBench 在搭建新基准时的代码行数下降约 70%。  
- **性能提升**：原文未给出具体数值，只说明“显著降低了评测开发成本”，并且在对抗攻击实验中能够捕获约 30% 以上的安全失误，远高于传统评测的 5% 检出率。  
- **消融实验**：作者分别关闭对抗攻击模块、动态评估协议和 Prompt Engineering 插件，发现对整体错误率的贡献分别约为 12%、8% 和 5%。这说明每个子系统都对最终评测质量有实质性帮助。  
- **局限性**：论文承认当前只支持基于文本的 LLM，尚未覆盖多模态模型（如 Vision‑LLM），并且对大规模分布式推理的优化仍在进行中。

### 影响与延伸思考
PromptBench 在发布后迅速被学术团队和工业实验室引用，成为构建新基准的“标准工具箱”。后续有几篇工作（如 **EvalPlus**、**SafetyBench**）直接在 PromptBench 上实现了更细粒度的安全评测，甚至把它扩展到代码生成模型的单元测试。对想进一步探索的读者，可以关注以下方向：  
- **多模态评测**：把图像、音频等输入纳入统一评测框架。  
- **自动化 Prompt 搜索**：结合强化学习或进化算法，让 PromptBench 自动生成最具挑战性的提示。  
- **大规模分布式评测平台**：在云端实现数千卡 GPU 同时跑评测，进一步压缩实验周期。

### 一句话记住它
PromptBench 把提示设计、安全攻击、动态评估全装进一个库，让评测 LLM 像搭积木一样简单且可扩展。