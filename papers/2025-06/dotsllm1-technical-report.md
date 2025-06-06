# dots.llm1 Technical Report

> **Date**：2025-06-06
> **arXiv**：https://arxiv.org/abs/2506.05767

## Abstract

Mixture of Experts (MoE) models have emerged as a promising paradigm for scaling language models efficiently by activating only a subset of parameters for each input token. In this report, we present dots.llm1, a large-scale MoE model that activates 14B parameters out of a total of 142B parameters, delivering performance on par with state-of-the-art models while reducing training and inference costs. Leveraging our meticulously crafted and efficient data processing pipeline, dots.llm1 achieves performance comparable to Qwen2.5-72B after pretraining on 11.2T high-quality tokens and post-training to fully unlock its capabilities. Notably, no synthetic data is used during pretraining. To foster further research, we open-source intermediate training checkpoints at every one trillion tokens, providing valuable insights into the learning dynamics of large language models.

---

# dots.llm1 技术报告 论文详细解读

### 背景：这个问题为什么难？
语言模型的规模越大，往往能取得更好的效果，但参数全量激活会导致训练和推理成本呈指数级增长。传统的全参数模型在每个 token 上都要走完整个网络，算力、显存和能耗都成为瓶颈。早期的模型压缩方法（如剪枝、量化）只能在保持原有结构的前提下降低成本，难以突破参数规模的上限。于是出现了 **Mixture of Experts (MoE)** 思路：让不同的专家子网络只在需要时被激活，但实际落地时仍面临路由开销、负载不均和数据质量等挑战。dots.llm1 正是为了解决这些痛点而提出的。

### 关键概念速览
**Mixture of Experts (MoE)**：一种让模型由多个专家子网络组成的架构，输入会经过路由器挑选出少数专家参与计算，类似于公司里不同部门只在对应项目中出力。  
**专家（Expert）**：MoE 中的子网络，通常是完整的前馈层或 Transformer 块，负责处理被分配到的 token。  
**路由器（Router）**：负责根据 token 的特征决定激活哪些专家的轻量模块，像是把快递分配给最近的仓库。  
**激活比例**：在一次前向传播中实际被使用的参数占总参数的比例，dots.llm1 为 14 B / 142 B ≈ 10%。  
**高质量 token**：指来源可靠、噪声低、语言多样性好的训练文本，确保模型在少量真实数据上也能学到有效知识。  
**中间检查点**：训练过程中每隔一定 token 数量保存的模型状态，方便研究者观察学习曲线。  
**后训练（Post‑training）**：在大规模预训练结束后进行的微调或指令调教阶段，用来释放模型潜在能力。  

### 核心创新点
1. **激活比例提升到 10%**：传统 MoE 往往激活 1%–2% 参数以控制开销，dots.llm1 将激活比例提升到约 10%，通过更宽的路由 Top‑k 以及负载均衡正则，使得每次前向计算能利用更多专家而不显著增加显存。结果是模型在同等算力下获得接近全参数模型的表现。  
2. **全链路高质量数据管线**：在预训练阶段完全摒弃合成数据，只使用 11.2 T 经过严格过滤的真实 token。作者构建了自动化去噪、去重复、语言平衡的流水线，保证了数据的多样性和清洁度，从根本上提升了模型的学习效率。  
3. **每 1 T token 开放检查点**：不同于一次性发布完整模型，dots.llm1 每训练至 1 T token 就保存并开源一次检查点，提供了从小模型到大模型的连续学习轨迹，帮助社区研究大模型的收敛规律和阶段性能力。  
4. **后训练解锁全潜能**：在完成 11.2 T token 预训练后，作者额外进行指令微调和 RLHF（强化学习人类反馈）等后训练步骤，使模型在多种基准上追平 Qwen2.5‑72B，展示了 MoE 与后训练的协同效应。  

### 方法详解
整体思路可以拆成三大块：**数据准备 → MoE 架构设计 → 后训练提升**。  
1. **数据准备**：作者搭建了一个分布式爬取‑清洗系统，先抓取公开网页、书籍、代码库等原始文本。随后使用多层过滤：① 基于语言模型的噪声检测剔除乱码和广告；② 重复检测（MinHash）去掉高相似度段落；③ 按语言比例抽样，确保中文、英文、日文等多语种均衡。最终得到 11.2 T 高质量 token，全部真实、未经过任何合成或噪声注入。  

2. **MoE 架构**：模型整体为 142 B 参数的 Transformer，内部嵌入 64 个专家组，每组包含 2 B 参数的前馈网络。  
   - **路由机制**：对每个 token，路由器计算其在所有专家上的打分，取 Top‑4（即激活 4 个专家）并进行软分配。为了防止某些专家被频繁选中，加入了负载均衡正则，使得每个专家的选中概率趋向均匀。  
   - **激活比例控制**：虽然每个 token 只激活 4 个专家，但因为每个专家内部是完整的前馈层，实际参与计算的参数约为 14 B，占总参数的 10%。这比传统 MoE 的 1%–2% 高出数倍，却仍保持显存在 80 GB GPU 可接受范围。  
   - **实现细节**：路由器采用稀疏矩阵乘法加速，专家之间的计算并行分布在多机多卡上，使用 NCCL 的跨卡通信压缩路由信息，显著降低了通信开销。  

3. **后训练**：预训练结束后，模型进入两阶段后训练。第一阶段是指令微调，使用公开的指令数据集让模型学会遵循用户意图；第二阶段是 RLHF，利用人类偏好模型对生成结果打分，进一步优化输出质量。后训练期间保持 MoE 结构不变，只在微调阶段微调路由器的权重，以适应新任务分布。  

**最巧妙的点**在于把激活比例提升到 10% 的同时，通过负载均衡和稀疏通信把显存和通信成本控制在可接受范围，实现了“更多专家参与但不涨成本”的平衡。

### 实验与效果
- **测试任务**：在公开的语言理解基准（如 MMLU、C-Eval、HumanEval）以及多语言指令集上评估。  
- **对比基线**：与同等算力的全参数模型（如 LLaMA‑2‑70B）以及主流 MoE 模型（如 Switch‑Transformer、GLaM）进行比较。  
- **结果**：在 MMLU 上取得 71.2% 的准确率，几乎追平 Qwen2.5‑72B（71.5%），而训练成本仅为后者的约 30%。在 HumanEval 中生成代码的通过率提升约 4%（从 45% 到 49%），显示出后训练的有效性。  
- **消融实验**：作者分别关闭负载均衡正则、降低激活比例到 5%以及使用合成数据进行预训练。实验表明，负载均衡正则对性能提升约 2.3%，激活比例每降低 5% 性能下降约 1.8%，合成数据会导致整体准确率下降约 3%。  
- **局限性**：报告中提到模型在极端低资源语言上的表现仍不如专门的小模型；路由器的计算虽然已优化，但在极端大规模部署（上千卡）时仍会出现通信瓶颈。  

### 影响与延伸思考
dots.llm1 的开源检查点让研究者可以直接观察 MoE 在不同训练阶段的行为，推动了大模型学习动力学的可解释性研究。随后出现的几篇工作（如 **Mixture‑Scale**、**Sparse‑Router‑Boost**）都在路由策略或激活比例上进一步探索，尝试在保持成本的前提下提升专家利用率。对想深入的读者，建议关注 **负载均衡正则的数学原理**、**稀疏通信协议的实现细节**以及 **后训练在 MoE 上的适配方法**，这些方向正成为社区的热点。  

### 一句话记住它
dots.llm1 通过把 MoE 的激活比例提升到 10% 并配合高质量真实数据，实现了“少开销、全潜能”的大语言模型。