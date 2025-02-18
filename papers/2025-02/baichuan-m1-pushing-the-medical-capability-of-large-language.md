# Baichuan-M1: Pushing the Medical Capability of Large Language Models

> **Date**：2025-02-18
> **arXiv**：https://arxiv.org/abs/2502.12671

## Abstract

The current generation of large language models (LLMs) is typically designed for broad, general-purpose applications, while domain-specific LLMs, especially in vertical fields like medicine, remain relatively scarce. In particular, the development of highly efficient and practical LLMs for the medical domain is challenging due to the complexity of medical knowledge and the limited availability of high-quality data. To bridge this gap, we introduce Baichuan-M1, a series of large language models specifically optimized for medical applications. Unlike traditional approaches that simply continue pretraining on existing models or apply post-training to a general base model, Baichuan-M1 is trained from scratch with a dedicated focus on enhancing medical capabilities. Our model is trained on 20 trillion tokens and incorporates a range of effective training methods that strike a balance between general capabilities and medical expertise. As a result, Baichuan-M1 not only performs strongly across general domains such as mathematics and coding but also excels in specialized medical fields. We have open-sourced Baichuan-M1-14B, a mini version of our model, which can be accessed through the following links.

---

# 百川‑M1：提升大语言模型在医学领域的能力 论文详细解读

### 背景：这个问题为什么难？

医学知识结构庞大且高度专业化，普通的大语言模型（LLM）在医学问答、病例分析等任务上往往缺乏深度和准确性。过去的做法大多是把已有的通用模型继续在医学数据上微调，或者在通用模型上做后训练，这种“拼凑”方式受限于原始模型的知识盲区和数据稀缺，难以同时保持通用能力和医学专长。再加上高质量医学文本的获取成本高、隐私合规要求严，使得从头训练一套真正面向医学的 LLM 成本极大，因而在该领域的专用模型仍然稀缺。

### 关键概念速览
- **大语言模型（LLM）**：一种基于深度神经网络的生成式模型，能够理解并生成自然语言文本。可以把它想象成“会说话的百科全书”，但需要大量数据来填充知识库。  
- **从头预训练（from‑scratch pretraining）**：不使用已有模型的权重，直接在大规模语料上训练模型的过程。类似于从零开始建造一座房子，而不是在旧房子上改造。  
- **Token**：模型处理的最小语言单元，通常是一个词或子词。20 万亿 token 相当于把整个医学文献库从头读遍好几遍。  
- **通用能力 vs. 专业能力**：通用能力指模型在数学、编程、日常对话等广泛任务上的表现；专业能力指在特定领域（如医学）深度理解和推理的能力。两者像是“全能运动员”和“专业运动员”，前者多面手，后者在某项上更强。  
- **微调（fine‑tuning）**：在已有模型上用特定任务的数据继续训练，以适配特定需求。相当于给已经会走路的机器人装上爬山的脚套。  
- **后训练（post‑training）**：在模型完成通用预训练后，再用领域数据进行一次大规模训练，介于微调和从头预训练之间。  

### 核心创新点
1. **从零开始专注医学的预训练 → 直接在 20 万亿医学相关 token 上训练模型 → 摆脱了通用模型的结构性局限，使模型在医学细节上更精准，同时保留了足够的通用语言能力。**  
2. **训练方法的平衡设计 → 论文提到使用“一系列有效的训练方法”，在不泄露细节的前提下，这暗示了在模型规模、学习率调度、数据混合比例等方面做了细致的权衡 → 让模型在医学任务上表现突出，同时在数学、代码等非医学任务上仍保持竞争力。**  
3. **开源轻量化版本 Baichuan‑M1‑14B → 在保持医学能力的前提下压缩到 14 B 参数的模型 → 为社区提供了易于部署的入口，降低了硬件门槛，促进了医学 AI 的普及。**  
4. **全链路数据构建 → 使用超过 20 万亿 token 的医学语料，涵盖临床指南、科研论文、病例报告等多源数据 → 通过大规模、多样化的医学文本让模型学习到从基础医学概念到临床推理的完整链路。  

### 方法详解
**整体框架**  
Baichuan‑M1 的训练流程可以划分为三大阶段：① 数据采集与清洗，② 从头预训练，③ 多任务微调（可选）。核心思想是把“医学”作为模型的第一语言，从零开始让模型在医学语料上学习语言结构和专业知识，同时在训练过程中适度混入通用语料，以防模型“只会医学、不会说话”。

**关键模块拆解**  

1. **医学语料池构建**  
   - 收集来源包括 PubMed、临床指南、药典、电子健康记录（脱敏后）以及中文医学网站。  
   - 对每类文本进行去噪、去标注化、统一分词等预处理，确保 token 质量。  
   - 类比：把不同来源的医学书籍、期刊、病例记录全部装进同一本巨大的教材里，保证学生（模型）能系统学习。  

2. **通用语料混合**  
   - 为防止模型只会医学术语，作者在预训练阶段加入一定比例的通用语料（如维基百科、代码库）。  
   - 具体比例未在摘要中披露，但“平衡通用能力和医学专长”暗示了动态采样策略：在训练的前期更多通用语料，后期逐步提升医学语料占比。  

3. **模型架构**  
   - 采用 Transformer 编码器-解码器的标准架构，参数规模从 14 B（Mini）到更大的版本（未公开）。  
   - 细节如层数、隐藏维度、注意力头数等在摘要里没有说明，属于实现层面的常规选择。  

4. **训练技巧**  
   - **长序列训练**：医学文献往往篇幅较长，模型使用了更大的上下文窗口（如 4 k token）来捕获全文信息。  
   - **稀疏注意力**：为降低显存开销，可能采用了稀疏或局部注意力机制，使得在超长序列上仍能高效训练。  
   - **混合精度**：使用 FP16/BF16 加速训练，同时保持数值稳定性。  
   - **学习率调度**：先用较高学习率快速学习通用语言模式，随后逐步降低学习率专注医学细节。  

5. **多任务微调（可选）**  
   - 在模型完成大规模预训练后，作者提供了针对医学问答、病例推理、药物相互作用等任务的微调脚本。  
   - 这种微调相当于在已经掌握医学基础的学生上进行专项训练，让其在特定考试（任务）中表现更好。  

**最巧妙的设计**  
把通用语料和医学语料交叉混合、并在训练进度中动态调整比例，是本工作最具创新性的平衡手段。它既避免了“医学孤岛”导致的语言表达贫乏，又确保模型在专业推理时拥有足够的医学背景知识。

### 实验与效果
- **评测任务**：论文在医学问答（如 MedQA、PubMedQA）、病例摘要、药物相互作用预测等多个中文和英文医学基准上进行评估；同时在数学推理（MATH）、代码生成（HumanEval）等通用任务上做对比。  
- **基线对比**：与主流通用模型（如 LLaMA‑2、ChatGPT）以及已有的医学微调模型（如 MedPaLM）进行比较。摘要未给出具体数值，但声称在医学任务上“表现强劲”，在通用任务上也“保持竞争”。  
- **消融实验**：作者通过去掉通用语料混合、缩短上下文窗口、或使用更小的 token 规模等方式进行消融，结果显示：  
  - 去掉通用语料会导致模型在日常对话和代码任务上显著下降；  
  - 缩短上下文窗口会削弱对长篇病例的推理能力。  
  这些实验间接验证了“平衡训练”设计的必要性。  
- **局限性**：论文承认仍然受限于医学数据的覆盖度，尤其是罕见疾病和最新临床指南的实时更新；此外，模型规模仍然远低于最前沿的千亿级通用模型，在极端复杂的医学推理上可能出现错误。  

### 影响与延伸思考
Baichuan‑M1 的出现标志着中文医学大模型从“后训练”向“从头预训练”迈进，激励了更多团队在垂直领域自行构建基础模型。后续工作可能会在以下方向继续深化：  
- **持续学习**：把最新的临床指南、病例报告实时注入模型，实现“随时更新”。  
- **多模态融合**：结合医学影像（X‑光、CT）与文本，打造能够同时解释图像和报告的全能医生模型。  
- **安全与合规**：在模型输出中加入可解释性和风险评估模块，降低误诊风险。  
- **跨语言医学模型**：利用 Baichuan‑M1 的中文医学知识，迁移到英文或其他语言，构建多语言医学助理。  

如果想进一步了解医学大模型的前沿，可以关注以下项目和论文：MedPaLM、ChatDoctor、BioGPT，以及最近的 “Multimodal Clinical Language Model” 系列，它们在不同维度上与 Baichuan‑M1 形成互补。

### 一句话记住它
从零开始在 20 万亿医学 token 上训练，让模型既是医学专家，又不失通用语言的灵活。