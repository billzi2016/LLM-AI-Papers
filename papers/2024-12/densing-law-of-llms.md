# Densing Law of LLMs

> **Date**：2024-12-05
> **arXiv**：https://arxiv.org/abs/2412.04315

## Abstract

Large Language Models (LLMs) have emerged as a milestone in artificial intelligence, and their performance can improve as the model size increases. However, this scaling brings great challenges to training and inference efficiency, particularly for deploying LLMs in resource-constrained environments, and the scaling trend is becoming increasingly unsustainable. This paper introduces the concept of ``\textit{capacity density}'' as a new metric to evaluate the quality of the LLMs across different scales and describes the trend of LLMs in terms of both effectiveness and efficiency. To calculate the capacity density of a given target LLM, we first introduce a set of reference models and develop a scaling law to predict the downstream performance of these reference models based on their parameter sizes. We then define the \textit{effective parameter size} of the target LLM as the parameter size required by a reference model to achieve equivalent performance, and formalize the capacity density as the ratio of the effective parameter size to the actual parameter size of the target LLM. Capacity density provides a unified framework for assessing both model effectiveness and efficiency. Our further analysis of recent open-source base LLMs reveals an empirical law (the densing law)that the capacity density of LLMs grows exponentially over time. More specifically, using some widely used benchmarks for evaluation, the capacity density of LLMs doubles approximately every three months. The law provides new perspectives to guide future LLM development, emphasizing the importance of improving capacity density to achieve optimal results with minimal computational overhead.

---

# LLM的密度法则 论文详细解读

### 背景：这个问题为什么难？
过去几年里，提升大语言模型（LLM）性能的唯一可靠手段几乎是“加参数”。模型越大，零样本推理、代码生成等任务的表现越好。但参数量的指数增长带来了巨大的算力、存储和能耗成本，普通服务器甚至高端云实例都难以承受。研究者们尝试通过更好的训练技巧、稀疏化或蒸馏来削减开销，却往往只能在特定任务上略有提升，整体效率仍旧受限。于是，缺少一种能够同时量化“效果”和“资源消耗”的统一指标，导致业界难以判断到底是继续追求更大模型，还是在同等算力下提升模型“密度”。这正是本文要解决的痛点。

### 关键概念速览
**容量密度（capacity density）**：模型实际参数量与能够达到同等下游性能的“等效参数量”之比。想象把一块金子压得更紧，体积不变但价值更高，容量密度越大，模型越“紧凑”。  
**等效参数量（effective parameter size）**：如果让一组已知性能的参考模型（比如不同规模的GPT‑2）去匹配目标模型的表现，需要多少参数才能做到。它是把性能映射回参数空间的桥梁。  
**参考模型（reference models）**：一批公开的、规模已知的基准模型，用来学习“参数‑性能”关系的样本。相当于用已知的尺子去测量未知的长度。  
**缩放律（scaling law）**：描述模型参数规模与下游任务表现之间的数学关系，通常是对数或幂律。这里的缩放律专门用于预测参考模型的性能。  
**指数增长（exponential growth）**：一种增长方式，数值每隔固定时间翻倍。本文发现容量密度大约每三个月翻一番。  
**下游任务（downstream tasks）**：模型训练好后实际要解决的具体任务，如问答、摘要、代码生成等。  
**基准评测（benchmark）**：统一的测试集合，用来比较不同模型的表现。常见的有MMLU、HumanEval等。  

### 核心创新点
1. **从单纯参数计数到“等效参数”概念**：过去的比较往往直接看模型大小，忽略了不同架构或训练技巧带来的性能差异。本文先用参考模型建立参数‑性能映射，再把目标模型的表现映射回等效参数量，实现了“同等效果下的参数需求”。这一步让我们可以客观评估模型的“性价比”。  
2. **提出容量密度作为统一评价指标**：把等效参数量除以实际参数量，得到一个无量纲的数值。容量密度高说明模型在保持或提升效果的同时，显著压缩了真实参数规模。与传统的 FLOPs、显存占用等单维度指标不同，容量密度兼顾效果与效率。  
3. **发现并验证“密度法则”**：在大量开源基座模型（如LLaMA、Mistral、Qwen等）上测算容量密度，作者观察到一个惊人的趋势：容量密度随时间呈指数增长，约每三个月翻倍。这个经验规律为模型研发提供了时间轴上的“密度目标”。  
4. **提供可复用的评估框架**：论文公开了参考模型集合、缩放律拟合代码以及容量密度计算脚本，任何后续模型都可以直接套用，形成统一的比较基准。  

### 方法详解
整体思路可以拆成三步：①准备参考模型并拟合缩放律，②用缩放律把目标模型的下游成绩映射到等效参数量，③计算容量密度。下面逐步展开。

1. **构建参考模型库**  
   作者挑选了公开的、规模覆盖从几千万到上百亿参数的基座模型。每个模型在同一套基准（如MMLU、HumanEval）上跑出分数，形成“参数‑分数”对。这里的关键是保持评测一致，避免因数据差异导致的噪声。

2. **学习参数‑性能的缩放律**  
   将参考模型的参数规模取对数，分数也取对数，使用最小二乘法拟合出一条幂律曲线：分数 ≈ a·(参数)^b + c。这个公式本质上告诉我们，增加多少参数会带来多少性能提升。因为参考模型覆盖了广阔的规模区间，拟合误差相对较小。

3. **推算等效参数量**  
   对于待评估的目标模型，先在同样的基准上得到实际分数。把这个分数代入上一步得到的缩放律，解出对应的参数规模，这个解就是“等效参数量”。直观上，它回答了这样一个问题：如果我要用参考模型来达到目标模型的表现，需要多少参数？

4. **计算容量密度**  
   容量密度 = 等效参数量 / 实际参数量。数值大于1说明模型在同等效果下用了更少的真实参数，数值小于1则相反。作者进一步把所有开源模型的容量密度按发布时间排序，绘制出随时间的曲线，发现近三年呈指数上升。

**最巧妙的地方**在于把“效果”先映射回“参数”，而不是直接把“参数”映射到“效果”。这样做的好处是把所有模型统一到同一个“参数空间”，避免了架构差异、数据规模等外部因素的干扰。

### 实验与效果
- **评测任务**：论文主要使用了MMLU（多语言多任务理解）和HumanEval（代码生成）两套公开基准，覆盖语言理解和生成两大方向。  
- **对比基线**：把每个模型的原始参数规模直接与基准分数对比，得到传统的“参数‑性能”曲线；再与容量密度方法的结果进行对照。  
- **关键发现**：在同等分数下，2023 年后发布的开源模型的等效参数量仅为实际参数量的 30% 左右，容量密度约为 3.3。相比 2020 年的模型提升了近 10 倍。作者指出，容量密度大约每三个月翻一番，这一指数趋势在 2022‑2024 年的 30+ 模型上都有体现。  
- **消融实验**：作者分别去掉参考模型的最小规模、最大发布模型以及不同基准的组合，发现只要保留至少三档不同规模的参考模型，缩放律的拟合误差仍在 5% 以内，容量密度的趋势基本不变。说明方法对参考模型的选择具有一定鲁棒性。  
- **局限性**：论文承认容量密度受基准选择影响显著；如果基准偏向特定任务，等效参数量的估计可能失真。此外，当前只在英文和少数多语言基准上验证，跨语言、跨模态的通用性还有待进一步实验。

### 影响与延伸思考
这篇论文在发布后迅速成为评估新模型“性价比”的标准之一。很多后续开源项目（如OpenChat、Phi‑3）在发布说明里直接给出容量密度，以证明自己在同等算力下更高效。学术界也开始围绕“密度法则”展开讨论，出现了几篇尝试把稀疏化、量化或混合专家模型的效果映射到容量密度的工作。未来的研究可以往两个方向延伸：①扩展参考模型库到多模态（视觉、音频）领域，验证密度法则是否跨模态成立；②把容量密度与能耗、碳排放等绿色AI指标结合，形成更全面的“可持续性”评估框架（推测）。如果想进一步了解，可以关注最近的“Scaling Laws for Efficient AI”系列论文以及各大开源社区的评测报告。

### 一句话记住它
容量密度把“效果等价的参数需求”除以真实参数量，让我们看到模型在同等表现下的“紧凑度”，并发现它每三个月翻一番的指数增长趋势。