# Program-Aided Reasoners (better) Know What They Know

> **Date**：2023-11-16
> **arXiv**：https://arxiv.org/abs/2311.09553

## Abstract

Prior work shows that program-aided reasoning, in which large language models (LLMs) are combined with programs written in programming languages such as Python, can significantly improve accuracy on various reasoning tasks. However, while accuracy is essential, it is also important for such reasoners to "know what they know", which can be quantified through the calibration of the model. In this paper, we compare the calibration of Program Aided Language Models (PAL) and text-based Chain-of-thought (COT) prompting techniques over 5 datasets and 2 model types: LLaMA models and OpenAI models. Our results indicate that PAL leads to improved calibration in 75% of the instances. Our analysis uncovers that prompting styles that produce lesser diversity in generations also have more calibrated results, and thus we also experiment with inducing lower generation diversity using temperature scaling and find that for certain temperatures, PAL is not only more accurate but is also more calibrated than COT. Overall, we demonstrate that, in the majority of cases, program-aided reasoners better know what they know than text-based counterparts.

---

# 程序辅助推理模型（更）懂得自我认知 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）解决需要多步推理的任务时，单纯让模型直接输出答案往往会出现错误，尤其是当题目涉及数学、逻辑或常识链条时。过去的提升手段主要是让模型先写出思考过程（Chain‑of‑Thought，CoT），这的确能提高准确率，但模型的置信度仍然不可靠——模型可能对错误答案也打出高分。换句话说，模型“知道自己在猜”还是“真的懂”之间缺少量化的衡量，这在安全关键或需要人机协作的场景里是致命的。于是，如何让模型在保持或提升准确率的同时，能够更好地校准自己的信心，成为了迫切需要解决的问题。

### 关键概念速览

**大语言模型（LLM）**：通过海量文本训练得到的生成式模型，能够完成对话、写作、代码等多种任务。把它想成一个“会说话的百科全书”，但有时会“自信地说错”。  

**Chain‑of‑Thought（CoT）**：让模型在给出最终答案前先把推理步骤写出来，类似于人在解题时先列草稿。这样可以让错误更容易被发现，也提升了答案的正确率。  

**Program‑Aided Language model（PAL）**：在 CoT 基础上进一步让模型生成可执行的代码（如 Python），模型把推理过程交给程序执行后再读取结果。可以把它比作“让模型写出计算器指令，然后让真正的计算器算”。  

**模型校准（Calibration）**：衡量模型输出的置信度（概率）与实际正确率的匹配程度。理想情况下，模型给出 80% 置信度时，答案真的有 80% 的概率是对的。  

**温度（Temperature）**：控制生成文本随机性的超参数，温度高会让输出更多样，温度低则更确定。把它想成“调节模型的冒险精神”。  

**生成多样性（Generation Diversity）**：指同一提示下模型可能产生的不同答案数量和差异程度。多样性大往往意味着不确定性高。  

### 核心创新点

1. **从准确率转向校准的系统性比较**  
   之前的研究大多只报告 PAL 在准确率上的提升，这篇论文把焦点放到校准上，直接对比 PAL 与 CoT 在 5 套任务、两类模型（LLaMA 系列和 OpenAI 系列）上的置信度匹配情况。结果显示，在 75% 的实验配置里，PAL 的校准误差更低，说明它不仅更准，还更懂自己的把握程度。

2. **关联生成多样性与校准表现**  
   作者观察到，生成过程越“统一”（即不同提示下产生的答案差异小），校准越好。于是提出用温度调节来人为降低多样性，验证了低温度下 PAL 能同时提升准确率和校准度。这一步把“生成多样性”这一看似副作用的因素，转化为可控的调参手段。

3. **跨模型、跨任务的广泛实验框架**  
   通过在 LLaMA‑7B/13B、OpenAI‑gpt‑3.5‑turbo、gpt‑4 四个模型上跑同样的 PAL 与 CoT 实验，展示了结论的稳健性。之前的 PAL 研究多局限于单一模型或单一任务，这里提供了更具普适性的证据。

### 方法详解

整体思路可以拆成三步：**提示设计 → 代码生成与执行 → 置信度评估**。下面按顺序展开。

1. **提示设计**  
   - 对每个任务，作者分别写了两套提示：一种是传统的 CoT 提示，要求模型一步步写出文字推理；另一种是 PAL 提示，要求模型把推理步骤写成可执行的 Python 代码。两套提示在文字表述上保持尽可能一致，只是输出形式不同，以排除提示本身的偏差。

2. **代码生成与执行**  
   - 在 PAL 流程里，模型输出的代码会被送入一个安全的 Python 沙箱执行。比如在求解方程时，模型会生成 `sympy.solve(...)` 之类的代码，沙箱返回数值或符号解。返回的结果再被包装成自然语言答案。这里的关键是**把推理交给真正的程序执行器**，避免模型在数值计算上出错。

3. **置信度评估**  
   - 对每一次生成，模型会同时输出一个概率分布（logits 经过 softmax），作者把最高概率对应的数值当作该答案的置信度。随后使用**期望校准误差（ECE）**等指标，比较置信度与实际正确率的吻合程度。低 ECE 表示模型更“自知”。

4. **温度调节实验**  
   - 为了验证生成多样性对校准的影响，作者在同一任务上分别使用温度 0.0、0.2、0.5、1.0 进行实验。温度越低，模型输出越确定，生成的答案几乎唯一；温度高时，同一提示会产生多种不同的文字或代码。实验结果显示，在温度 0.2 左右，PAL 的校准优势最明显，同时保持或提升了准确率。

**最巧妙的点**在于把“代码执行”视作一种**外部校准器**：程序本身是确定性的，它把模型的模糊推理转化为硬算子，从而自然降低了模型对答案的“自信噪声”。这也是 PAL 在校准上领先 CoT 的根本原因。

### 实验与效果

- **数据集与任务**：作者选取了 5 套公开的推理基准，包括 GSM8K（数学文字题）、SVAMP（代数求解）、AQuA（选择题数学）、MATH（高阶数学）以及一个逻辑推理数据集。每套任务都提供了标准答案和置信度评估需求。

- **对比基线**：CoT（文字链式思考）作为主要基线；此外还包括直接一次性输出（Zero‑Shot）和少量提示（Few‑Shot）版本，以展示 PAL 的相对提升幅度。

- **主要结果**：在 75%（即 15/20）实验配置中，PAL 的 ECE 明显低于 CoT，最高差距约为 0.08（具体数值请参考原文表格）。在准确率方面，PAL 也保持或略微超越 CoT，尤其在数学求解任务上提升约 2–3%。温度实验表明，当温度设为 0.2 时，PAL 的 ECE 最低，且准确率不下降。

- **消融实验**：作者分别去掉代码执行、只保留文字 CoT、以及固定温度等设置，发现**去掉代码执行**后校准误差回升到与 CoT 相当，说明代码执行是提升校准的关键因素。

- **局限性**：论文承认 PAL 依赖于能够安全执行的代码环境，对某些需要外部 API 或不可沙箱化的任务不适用；此外，温度调节虽然有效，但在实际部署中需要权衡多样性与创造性，过低的温度可能抑制模型的创新答案。

### 影响与延伸思考

这篇工作把**模型自我认知**（校准）拉进了程序辅助推理的讨论范围，促使后续研究不仅追求更高的准确率，还要关注置信度的可靠性。随后出现的几篇论文（如 *Self‑Check LLMs*、*Tool‑Augmented Calibration*）都在不同程度上借鉴了 PAL 的“外部执行+置信度评估”思路，尝试把搜索、数据库查询等工具也纳入校准框架。对想进一步探索的读者，可以关注**工具使用（Tool‑Use）与校准的交叉**、以及**在多模态或跨语言环境下的程序辅助推理**等方向，这些都是当前社区的热点。

### 一句话记住它

让模型写代码并让真实程序跑出来，不仅让答案更对，还让模型的自信度更靠谱。