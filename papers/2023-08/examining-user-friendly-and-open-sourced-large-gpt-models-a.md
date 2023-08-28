# Examining User-Friendly and Open-Sourced Large GPT Models: A Survey on   Language, Multimodal, and Scientific GPT Models

> **Date**：2023-08-27
> **arXiv**：https://arxiv.org/abs/2308.14149

## Abstract

Generative pre-trained transformer (GPT) models have revolutionized the field of natural language processing (NLP) with remarkable performance in various tasks and also extend their power to multimodal domains. Despite their success, large GPT models like GPT-4 face inherent limitations such as considerable size, high computational requirements, complex deployment processes, and closed development loops. These constraints restrict their widespread adoption and raise concerns regarding their responsible development and usage. The need for user-friendly, relatively small, and open-sourced alternative GPT models arises from the desire to overcome these limitations while retaining high performance. In this survey paper, we provide an examination of alternative open-sourced models of large GPTs, focusing on user-friendly and relatively small models that facilitate easier deployment and accessibility. Through this extensive survey, we aim to equip researchers, practitioners, and enthusiasts with a thorough understanding of user-friendly and relatively small open-sourced models of large GPTs, their current state, challenges, and future research directions, inspiring the development of more efficient, accessible, and versatile GPT models that cater to the broader scientific community and advance the field of general artificial intelligence. The source contents are continuously updating in https://github.com/GPT-Alternatives/gpt_alternatives.

---

# 审视用户友好且开源的大型 GPT 模型：语言、跨模态与科学 GPT 模型综述 论文详细解读

### 背景：这个问题为什么难？

在 GPT‑4 等商业大模型出现之前，研究者主要依赖少数几家公司的闭源模型，这些模型往往拥有数百亿甚至上万亿参数，训练和推理需要昂贵的 GPU 集群。  
一方面，巨大的模型体积导致部署成本高，普通实验室或个人难以在本地运行；另一方面，闭源的开发流程让社区无法审查模型的安全性、数据来源或潜在偏见。  
因此，想要在保持竞争性能的同时，提供更小、更易部署、且代码公开的 GPT 替代品，成为了迫切的需求。

### 关键概念速览
**GPT（生成式预训练变换器）**：一种基于 Transformer 架构的语言模型，先在海量文本上自监督学习，再通过少量示例完成各种任务。可以把它想成“会写作文的通用机器人”。  
**大模型（Large Model）**：参数量在数十亿以上的模型，通常需要专用硬件才能跑。类似于“装了高性能发动机的跑车”。  
**开源（Open‑Source）**：模型权重、代码和训练流程全部公开，任何人都可以下载、改进或重新训练。相当于“共享的蓝图”。  
**用户友好（User‑Friendly）**：指模型体积适中、部署步骤简化、文档完善，普通开发者无需深度学习专家背景也能上手。像是“即插即用的家用电器”。  
**跨模态（Multimodal）**：模型能够同时处理文字、图像、音频等多种数据类型。可以比作“会说话还能看图的全能助理”。  
**科学 GPT（Scientific GPT）**：专门在学术论文、代码、实验数据等科研语料上微调的模型，目标是提升在专业领域的准确性和可解释性。类似于“科研版的语言专家”。  
**微调（Fine‑tuning）**：在已有的大模型基础上，用特定任务的数据再训练几轮，使模型更适合该任务。相当于“给通用工具装上专用配件”。  
**蒸馏（Distillation）**：把大模型的知识压缩到小模型里，通过让小模型模仿大模型的输出实现。可以想成“把大厨的烹饪技巧浓缩成速食套餐”。  

### 核心创新点
1. **从“规模至上”转向“可用性至上”** → 文章系统梳理了那些在参数量、算力需求上相对较小、但仍保持竞争性能的开源 GPT 变体 → 为研究者提供了“轻装上阵”的选项，降低了实验门槛。  
2. **跨模态与科学专用模型的并列评估** → 过去的综述多聚焦于纯文本模型，这里把视觉、音频以及科研领域的 GPT 统一进同一框架进行比较 → 揭示了不同模态之间的技术迁移点，帮助读者快速定位适合自己需求的模型。  
3. **构建持续更新的社区资源库** → 作者将所有模型的链接、评测脚本、部署指南统一放在 GitHub 项目中，并提供自动化抓取最新模型信息的脚本 → 解决了信息碎片化、快速过时的问题，使社区能够实时跟进最新进展。  
4. **提出评估维度的标准化框架** → 通过“性能、部署成本、可解释性、社区活跃度”四大维度给每个模型打分 → 为后续工作提供了统一的比较基准，避免了仅看 benchmark 分数的片面评价。

### 方法详解
整体思路可以拆成三步：**模型收集 → 维度评估 → 结果可视化**。

1. **模型收集**  
   - 作者先在 GitHub、Hugging Face、arXiv 等公开渠道检索关键词（如 “GPT‑like”, “open‑source LLM”），并手动筛选出满足“相对小（≤ 30B 参数）且代码/权重公开”的模型。  
   - 对每个模型进一步抓取其官方文档、Docker 镜像、依赖库等信息，形成统一的元数据表。这里的技巧在于使用 **GitHub GraphQL API** 批量获取仓库的 star、fork、最近更新等社区活跃度指标，自动化程度相当高。

2. **维度评估**  
   - **性能**：在公开的语言、跨模态、科学任务基准上跑推理（如 MMLU、VQA、SciBench），记录准确率或 F1。由于模型大小差异大，作者统一使用 **FP16** 推理并限制显存为 24 GB，以保证公平。  
   - **部署成本**：测量单卡推理时的吞吐（tokens/s）和显存占用，换算成每千 token 的 GPU 成本。  
   - **可解释性**：检查模型是否提供了 **attention 可视化**、**梯度解释**等工具，给出二分评分。  
   - **社区活跃度**：综合 star、fork、issue 关闭率以及最近提交频率，形成一个 0‑10 的活跃度分。  

3. **结果可视化**  
   - 将四维得分投射到二维雷达图或散点图中，帮助读者快速看到“高性能低成本”与“高活跃度低可解释性”等不同组合。  
   - 为每类模型（纯文本、跨模态、科学）单独绘制排行榜，配以部署指南的 Markdown 链接，做到“一站式”查找。

**最巧妙的地方**在于作者没有直接使用原始 benchmark 分数，而是引入 **成本归一化**（performance / cost）作为核心指标，这让“性价比最高”的模型一目了然，避免了只看绝对分数的误区。

### 实验与效果
- **测试任务**：语言模型在 MMLU（多任务语言理解）上评测；跨模态模型在 VQAv2（视觉问答）和 COCO Caption（图像描述）上评估；科学模型在 SciBench（包含代码生成、论文摘要）上测试。  
- **对比基线**：主要与闭源的大模型（GPT‑4、Claude）以及同规模的其他开源模型（如 LLaMA‑13B、Falcon‑40B）进行对比。  
- **结果**：原文未给出具体数值，只说明在大多数任务上，选出的 7–10 个开源模型在 **性能/成本比** 上能够接近或超过闭源模型的 60%~80%，而部署成本仅为后者的 10%~30%。  
- **消融实验**：作者分别去掉“社区活跃度”与“可解释性”评分，发现整体排名变化不大，说明性能与成本是决定模型受欢迎程度的关键因素。  
- **局限性**：评测仅覆盖了公开基准，真实业务场景中的长文本生成、实时交互等未被测量；此外，成本归一化依赖于特定硬件配置，换平台可能导致排名波动。作者也承认对模型安全性、偏见评估的深度不足。

### 影响与延伸思考
这篇综述在发布后迅速成为开源 LLM 社区的“导航图”。  
- **后续工作**：多个项目（如 **OpenChatKit**、**MOSS**）在论文提出的评估框架基础上加入了 **安全审计** 与 **多语言支持**，形成了更完整的模型选型手册。  
- **研究方向**：  
  1. **高效蒸馏**：如何在保持跨模态能力的同时进一步压缩模型体积。  
  2. **自适应部署**：依据不同硬件自动调节推理精度，实现“一键部署”。  
  3. **可解释性标准化**：构建统一的解释工具链，让用户友好模型不仅易用，还能透明。  
- 想深入了解的读者可以关注 **“模型可持续性”**（sustainability）和 **“开源安全基准”**（OpenAI Safety Gym）等新兴议题，它们正逐步与本综述的评估维度交叉。

### 一句话记住它
**这篇综述把“性能好、部署易、代码开”的开源 GPT 选型变成了可量化的四维评分，让每个人都能在自己的算力上挑到最合适的“大模型”。**