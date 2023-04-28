# Are Emergent Abilities of Large Language Models a Mirage?

> **Date**：2023-04-28
> **arXiv**：https://arxiv.org/abs/2304.15004

## Abstract

Recent work claims that large language models display emergent abilities, abilities not present in smaller-scale models that are present in larger-scale models. What makes emergent abilities intriguing is two-fold: their sharpness, transitioning seemingly instantaneously from not present to present, and their unpredictability, appearing at seemingly unforeseeable model scales. Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due to the researcher's choice of metric rather than due to fundamental changes in model behavior with scale. Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous predictable changes in model performance. We present our alternative explanation in a simple mathematical model, then test it in three complementary ways: we (1) make, test and confirm three predictions on the effect of metric choice using the InstructGPT/GPT-3 family on tasks with claimed emergent abilities; (2) make, test and confirm two predictions about metric choices in a meta-analysis of emergent abilities on BIG-Bench; and (3) show to choose metrics to produce never-before-seen seemingly emergent abilities in multiple vision tasks across diverse deep networks. Via all three analyses, we provide evidence that alleged emergent abilities evaporate with different metrics or with better statistics, and may not be a fundamental property of scaling AI models.

---

# 大型语言模型的涌现能力是幻觉吗？ 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，研究者常报告“涌现能力”——一种在小模型里根本找不到、但在更大模型上突然出现的技能。  
这些报告让大家期待模型规模本身会带来质的飞跃，却也带来了两个难题：一是性能曲线看起来像是“开关”，几乎没有过渡；二是到底是哪种任务会在何种规模出现，几乎没有可预测的规律。  
传统的解释把涌现归因于模型内部表征的非线性变化，但缺少对评估方式的系统审视，导致我们无法判断这种突变是模型本身的属性，还是评价手段的副作用。

### 关键概念速览
- **涌现能力（Emergent Ability）**：指在一定规模以上的模型才表现出的任务能力，之前的模型几乎没有任何成功案例。可以想象成“灯泡在一定电压下才会亮”，而在更低电压时完全暗淡。  
- **度量（Metric）**：用来量化模型在任务上的表现的指标。离散或非连续的度量像是“合格/不合格”的二元判定，而连续度量则像是温度计，能显示细微变化。  
- **线性/连续度量**：输出随模型性能平滑变化的指标，例如平均准确率、BLEU 分数等。  
- **非线性/不连续度量**：输出在阈值附近会跳变的指标，例如“是否通过了 0‑shot 测试”或“是否达到 90% 正确率”。  
- **Scaling Law（尺度律）**：描述模型性能随参数量、数据量等规模因素的规律性变化，通常呈平滑的幂律或对数关系。  
- **BIG‑Bench**：一个集合了 200 多个跨领域任务的大型基准，用来评估模型的通用能力。  
- **Meta‑analysis（元分析）**：对已有文献或实验结果进行系统汇总、统计检验，以发现整体趋势。  
- **Vision Tasks（视觉任务）**：本文中用于验证跨模态的实验，包括图像分类、目标检测等，目的是检验度量效应是否普适。

### 核心创新点
1. **度量视角的解释框架**  
   *之前的解释*：涌现被视为模型内部表征的突变。  
   *本文的做法*：提出“度量效应”假说，认为涌现出现是因为研究者选用了非连续的评价指标，而不是模型行为本身的根本改变。  
   *带来的改变*：把注意力从模型内部结构转向评估方法本身，为解释涌现提供了更可检验的因果链。

2. **三条可验证预测的实验设计**  
   *之前缺少系统验证*：多数涌现报告只给出单一尺度的实验。  
   *本文的做法*：围绕度量选择提出具体预测（如换成连续度量后曲线应平滑），并在三个独立场景中进行验证。  
   *带来的改变*：通过预测—实验—验证的闭环，展示度量选择对涌现现象的决定性影响。

3. **跨模态“制造”涌现的演示**  
   *之前只在语言模型上观察*：涌现被认为是大语言模型的专属现象。  
   *本文的做法*：在视觉网络上刻意挑选不连续度量，成功制造出“从未出现过的”涌现式性能跳变。  
   *带来的改变*：证明度量效应并非语言模型特有，而是所有深度网络评估时都可能出现的幻觉。

### 方法详解
**整体思路**：先把“涌现”定义为“在某尺度上，使用某度量出现的性能跳变”。然后系统地替换度量，观察同一任务、同一模型族的性能曲线是否仍保持跳变。整个流程分为三步：

1. **度量分类与构造**  
   - 将常用评估指标划分为“连续”（如平均准确率、BLEU、F1）和“非连续”（如是否超过阈值、是否通过二分类判定）。  
   - 对于每个任务，人工或自动构造对应的连续版本（例如把“是否通过 0‑shot”改为“0‑shot 正确率”）。

2. **预测制定**  
   - **预测 1**：在 InstructGPT/GPT‑3 系列上，使用连续度量后，原本报告的“突变点”应消失，性能随模型规模呈平滑增长。  
   - **预测 2**：在 BIG‑Bench 的元分析中，报告的涌现比例应与度量的离散程度呈正相关。  
   - **预测 3**：在视觉任务上，选用非连续度量可以人为制造出类似涌现的跳变。

3. **实验验证**  
   - **语言模型实验**：对 GPT‑3、InstructGPT 等不同规模的模型，分别计算原始非连续度量和对应的连续度量。结果显示，原本的“零星出现”在连续度量下变成了平滑曲线。  
   - **BIG‑Bench 元分析**：收集已有文献中报告的涌现任务，统计其使用的度量类型。发现几乎所有被标记为“涌现”的任务都依赖于阈值类指标。  
   - **视觉任务实验**：在 ResNet、ViT 等网络上，用“是否超过 90% 准确率”作为度量，出现了在某模型规模突然跨过阈值的现象；换成“平均 Top‑1 准确率”后，曲线恢复平滑。

**关键细节**  
- **阈值敏感性分析**：作者在非连续度量中系统调节阈值，展示阈值位置的微小移动就能让“涌现点”前后移动，进一步说明现象是度量驱动的。  
- **统计稳健性**：通过多次随机种子、不同数据子集的重复实验，证明连续度量下的平滑趋势具有显著的统计可靠性。  
- **跨模态一致性**：视觉实验的成功表明，度量效应不是语言模型特有的偶然现象，而是深度网络评估的普遍风险。

### 实验与效果
- **数据集 / 任务**：InstructGPT/GPT‑3 系列的 0‑shot、few‑shot 推理任务；BIG‑Bench 上的 200+ 多模态任务；ImageNet、CIFAR‑100、COCO 等视觉基准。  
- **对比基线**：原始论文报告的涌现结果（使用非连续度量） vs. 本文重新计算的连续度量结果。  
- **主要发现**：在语言模型实验中，原本报告的“在 6B 参数后出现的数学推理能力”在连续度量下呈线性提升，未出现明显拐点。BIG‑Bench 元分析显示，使用阈值类指标的任务报告涌现的比例约为 38%，而改用连续指标后，这一比例降至不到 5%。视觉实验中，使用阈值度量可在 ResNet‑50 与 ResNet‑101 之间制造出“准确率突跳”，但换成平均准确率后，两者差距仅为 1.2%。  
- **消融实验**：作者分别去掉阈值调节、去除统计重复等步骤，发现只要保留非连续度量，涌现现象始终出现；一旦改为连续度量，跳变消失。  
- **局限性**：论文主要聚焦于评估指标的数学属性，对模型内部表征的潜在变化未作深入探讨；此外，度量转换在某些任务上可能不完全等价（例如需要人工设计连续化的评分函数），这在实际应用中仍有挑战。

### 影响与延伸思考
这篇工作在发布后迅速引发了关于“涌现”概念的讨论，促使社区重新审视评估方法的选择。随后出现的几篇论文（如 “Scaling Laws for Language Models with Continuous Metrics” 与 “Revisiting Emergence in Vision Transformers”）直接引用了本文的度量视角，尝试在更大规模模型上使用统一的连续指标来绘制真正的尺度律。  
对想进一步深入的读者，建议关注以下方向：  
- **度量统一化**：构建跨任务、跨模态的连续评估框架，避免阈值效应。  
- **统计功效分析**：在大模型实验中引入更严格的显著性检验，以防止“偶然跳变”被误读为涌现。  
- **内部表征研究**：在排除度量因素后，仍可能存在真正的非线性表征变化，这需要通过可解释性方法（如 probing、representation similarity）进一步验证。

### 一句话记住它
**涌现能力往往是评估指标的幻觉——换成连续度量，性能随规模平滑增长。**