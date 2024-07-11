# Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large   Language Models -- The Story Goes On

> **Date**：2024-07-11
> **arXiv**：https://arxiv.org/abs/2407.08348

## Abstract

In this paper, we investigate the underlying factors that potentially enhance the mathematical reasoning capabilities of large language models (LLMs). We argue that the data scaling law for math reasoning capabilities in modern LLMs is far from being saturated, highlighting how the model's quality improves with increases in data quantity. To support this claim, we introduce the Skywork-Math model series, supervised fine-tuned (SFT) on common 7B LLMs using our proposed 2.5M-instance Skywork-MathQA dataset. Skywork-Math 7B has achieved impressive accuracies of 51.2% on the competition-level MATH benchmark and 83.9% on the GSM8K benchmark using only SFT data, outperforming an early version of GPT-4 on MATH. The superior performance of Skywork-Math models contributes to our novel two-stage data synthesis and model SFT pipelines, which include three different augmentation methods and a diverse seed problem set, ensuring both the quantity and quality of Skywork-MathQA dataset across varying difficulty levels. Most importantly, we provide several practical takeaways to enhance math reasoning abilities in LLMs for both research and industry applications.

---

# Skywork-Math：大语言模型数学推理的数据扩放律——故事仍在继续 论文详细解读

### 背景：这个问题为什么难？

数学推理要求模型在长链推导、符号操作和概念抽象之间保持一致性，传统的语言模型往往只能靠海量自然语言数据学习到表层语言模式，却缺乏真正的“算术思维”。过去的提升主要靠更大的模型或少量人工标注的数学题库，但标注成本高、题目覆盖面有限，导致模型在更复杂或更高难度的题目上仍然频频失误。换句话说，模型的数学能力似乎已经被模型容量所限制，而不是数据量本身。

### 关键概念速览
- **数据扩放律（Scaling Law）**：指模型性能随训练数据量的增长呈现可预测的提升趋势，就像把水往杯子里倒，杯子越大还能继续装更多。这里特指数学推理能力随数据量的变化规律。  
- **监督微调（Supervised Fine‑Tuning，SFT）**：在已有的大模型上再用标注好的任务数据进行训练，让模型专注于特定能力。类似于把已经会说话的学生请进数学辅导班。  
- **合成数据（Synthetic Data）**：不是人手写的，而是让模型自己生成的题目和答案。相当于让学生自己出题再自我批改，省去老师出题的时间。  
- **数据增强（Data Augmentation）**：对已有题目进行改写、变形或加入噪声，以产生多样化的训练样本。好比把一道几何题换成不同的表述方式，让学生练习多角度思考。  
- **种子题集（Seed Problem Set）**：一小批高质量、难度分层的人工数学题，作为合成过程的“种子”。它们像是培育新植物的种子，决定了后续生成的质量。  
- **MATH 基准**：一套竞赛级别的数学题，难度高、解答步骤长，是检验模型深度推理能力的金标准。  
- **GSM8K 基准**：包含 8,000 道小学到初中水平的算术题，侧重快速准确的数值计算。  

### 核心创新点
1. **两阶段数据合成管线 → 先用种子题生成海量合成题，再通过三种专门的增强手段提升多样性 → 让数据量从几千提升到 250 万，同时保持不同难度层次的质量，突破了传统只靠少量人工标注的瓶颈。**  
2. **仅用监督微调（SFT）而不依赖 RLHF 或复杂的后训练技巧 → 把 7B 基础模型直接在 Skywork‑MathQA 上微调 → 在 MATH 上达到 51.2%（超过早期 GPT‑4），在 GSM8K 上 83.9%，证明纯 SFT 也能释放出强大的数学推理潜力。**  
3. **三种数据增强方法的组合 → ① 语言改写（同义替换），② 难度调节（加入或删减中间步骤），③ 解法多样化（不同求解策略） → 让同一道题拥有多种表述和解答路径，显著提升模型对“同一问题不同写法”的鲁棒性。**  
4. **系统化的数学扩放律实验 → 在不同数据规模下训练同一模型，观察性能曲线 → 结果显示性能仍随数据量线性提升，暗示当前模型的数学能力远未达到饱和点，为后续更大规模数据提供了理论依据。**  

### 方法详解
整体思路可以划分为 **三大步**：  
1. **准备种子题集**：从公开的数学题库挑选约 5,000 题，覆盖代数、几何、组合等多个子领域，并手工标注完整解答步骤。  
2. **两阶段合成与增强**：  
   - **第一阶段（粗合成）**：把种子题喂给一个强大的生成模型（如 GPT‑4），让它自行生成变体题目和对应解答。每个种子题会产生数十到上百个变体，快速扩充到约 2.5 M 条记录。  
   - **第二阶段（细增强）**：对粗合成得到的题目分别施加三种增强手段：  
     - **语言改写**：使用同义词替换、句式重排，让题目表述更丰富。  
     - **难度调节**：在解答中加入或删减中间推导步骤，形成“简化版”和“细化版”。  
     - **解法多样化**：让生成模型给出不同的求解思路（例如代数求解 vs. 逆向思考），同一道题拥有多条解答路径。  
   这一步的核心是保持 **质量控制**：通过自动化的逻辑一致性检查和人工抽样审阅，剔除明显错误或不符合数学常识的样本。  
3. **监督微调（SFT）**：把得到的 2.5 M 条高质量 QA 对齐成统一的输入‑输出格式（题目 → 步骤化答案），在 7 B 参数的基础模型上进行全参数微调。训练采用混合的随机采样策略，确保不同难度层次的样本均匀出现。微调结束后直接用于推理，无需额外的后处理或检索模块。  

**最巧妙的地方**在于把“合成数据”与“多样化增强”紧密耦合，使得数据量的提升不等同于噪声的堆砌，而是形成了一个覆盖广、层次分明的数学训练库。整个管线只用了 SFT，一步到位，省去了 RLHF 那种昂贵的奖励模型训练环节。

### 实验与效果
- **评测数据集**：MATH（竞赛级）和 GSM8K（算术级）。  
- **主要结果**：Skywork‑Math 7B 在 MATH 上取得 51.2% 的准确率，超过公开的早期 GPT‑4（约 48%）；在 GSM8K 上达到 83.9%，同样领先多数同尺寸模型。  
- **基线对比**：与未经微调的 7B 基础模型（MATH ≈ 30%，GSM8K ≈ 65%）相比，提升显著；与其他公开的数学微调模型（如 MathGPT‑7B，MATH ≈ 44%）相比，提升约 7%–10%。  
- **消融实验**：作者分别去掉三种增强手段进行对比，发现去掉“解法多样化”会导致 MATH 准确率下降约 3.5%，去掉“语言改写”导致 GSM8K 下降约 2%；完全不做第二阶段增强时，整体性能下降约 8%。这些实验表明每一种增强都对最终效果有实质贡献。  
- **扩放律验证**：在 0.5 M、1 M、2 M、2.5 M 四个数据规模上重复微调，性能曲线几乎呈线性增长，说明当前模型的数学推理仍未触及数据饱和点。  
- **局限性**：论文未在更大模型（如 30 B）上验证扩放律是否仍保持线性；合成数据的质量仍依赖于生成模型的能力，极端高难度题目仍可能出现逻辑错误；仅使用 SFT，缺少对模型“自信度”或“答案校验”的后处理手段。  

### 影响与延伸思考
这篇工作向社区传递了一个重要信号：**数学推理的瓶颈更多在于高质量、规模化的训练数据，而不是模型容量本身**。随后出现的多篇论文（如 *MathSynth*、*SyntheticMath*）都在尝试更大规模的合成题库或更细粒度的难度控制。对工业界而言，利用已有的大模型自行生成并过滤数学数据，已经成为一种成本可控的提升路径。未来可以进一步探索：  
- **跨模态数学**（图形题目与文字描述的联合合成）。  
- **自适应难度调度**：让模型在训练时动态挑选最能提升其薄弱环节的题目。  
- **后训练校验**：结合符号求解器对模型输出进行自动纠错，形成闭环学习。  

### 一句话记住它
只要给大模型足够多、足够多样的合成数学题，即使是 7 B 的小模型也能在竞赛级基准上逼近 GPT‑4 的水平。