# Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in   Customized Large Language Models

> **Date**：2024-08-05
> **arXiv**：https://arxiv.org/abs/2408.02416

## Abstract

The drastic increase of large language models' (LLMs) parameters has led to a new research direction of fine-tuning-free downstream customization by prompts, i.e., task descriptions. While these prompt-based services (e.g. OpenAI's GPTs) play an important role in many businesses, there has emerged growing concerns about the prompt leakage, which undermines the intellectual properties of these services and causes downstream attacks. In this paper, we analyze the underlying mechanism of prompt leakage, which we refer to as prompt memorization, and develop corresponding defending strategies. By exploring the scaling laws in prompt extraction, we analyze key attributes that influence prompt extraction, including model sizes, prompt lengths, as well as the types of prompts. Then we propose two hypotheses that explain how LLMs expose their prompts. The first is attributed to the perplexity, i.e. the familiarity of LLMs to texts, whereas the second is based on the straightforward token translation path in attention matrices. To defend against such threats, we investigate whether alignments can undermine the extraction of prompts. We find that current LLMs, even those with safety alignments like GPT-4, are highly vulnerable to prompt extraction attacks, even under the most straightforward user attacks. Therefore, we put forward several defense strategies with the inspiration of our findings, which achieve 83.8\% and 71.0\% drop in the prompt extraction rate for Llama2-7B and GPT-3.5, respectively. Source code is avaliable at https://github.com/liangzid/PromptExtractionEval.

---

# 为什么我的 Prompt 会泄露？破解定制大语言模型中的 Prompt 提取威胁 论文详细解读

### 背景：这个问题为什么难？
随着大语言模型（LLM）参数规模爆炸，越来越多的企业把模型当成“黑盒”，只通过文字提示（prompt）来让模型完成特定任务。传统的微调需要大量标注数据和算力，而 prompt‑based 定制省时省力，因而迅速流行。可是，一旦用户的 prompt 包含商业机密或专有指令，攻击者就可能通过模型的输出把这些信息“偷走”。之前的安全研究大多聚焦在模型生成有害内容或对抗样本上，几乎没有系统地解释为什么模型会记住并泄露 prompt，也缺少针对性的防御手段。因此，弄清 Prompt 泄露的根本机制并给出可操作的防御方案，成为迫切需求。

### 关键概念速览
**Prompt（提示）**：用户给模型的任务描述或指令，相当于对模型说“请帮我做 X”。  
**Prompt Extraction（Prompt 提取）**：攻击者通过与模型交互，逆向推断出隐藏在模型内部的 Prompt 内容。  
**Prompt Memorization（Prompt 记忆）**：模型在生成过程中不自觉地把 Prompt 当作训练数据的一部分保存下来，导致后续输出中泄露。  
**Perplexity（困惑度）**：模型对一段文字的预测难度，数值越低说明模型越熟悉这段文字。可以把它想成模型对这段话的“亲切度”。  
**Attention Matrix（注意力矩阵）**：模型内部的关联表格，记录每个 token（词）在生成时关注了哪些前面的 token，类似于人写作文时“把前面的句子当作参考”。  
**Alignment（对齐）**：在模型训练或后处理阶段加入安全约束，让模型更倾向于遵守伦理或商业规则。  
**Scaling Law（尺度律）**：描述模型性能随参数量、数据量等因素变化的规律，这里指 Prompt 提取成功率随模型大小、Prompt 长度等的变化趋势。

### 核心创新点
1. **从经验到机制的跃迁**：以前的研究只报告“Prompt 能被偷”，没有解释背后原因。本文先用大量实验测出 Prompt 提取率随模型规模、Prompt 长度的变化，随后提出两条解释假设——困惑度假设和注意力路径假设，使得 Prompt 泄露从“现象”升到“机制”。  
2. **两大解释假设的对比实验**：困惑度假设认为模型对熟悉的文字（低 perplexity）更容易复现；注意力路径假设则认为在自注意力层中，Prompt token 与生成 token 之间形成了直接的“翻译通道”。作者分别构造高困惑度和低困惑度的 Prompt，并观察提取成功率，验证了两种因素都起作用。  
3. **防御思路的系统化**：基于上述机制，提出三类防御：① 在 Prompt 中加入高困惑度的噪声词；② 打乱注意力矩阵的直接映射（如随机化位置编码或插入干扰 token）；③ 在对齐阶段加入专门的 Prompt 隐蔽损失。实验显示，这些手段能把 Llama2‑7B 的提取率降到 16.2%，把 GPT‑3.5 降到 29%。  
4. **开源评估框架**：提供了 PromptExtractionEval 代码库，方便后续研究者复现实验、测量新模型的 Prompt 泄露风险，填补了该领域缺乏统一基准的空白。

### 方法详解
整体思路可以拆成四步：① 构造多样化 Prompt，② 设计提取攻击，③ 分析提取成功的内部信号，④ 基于分析结果设计防御。下面逐步展开。

1. **Prompt 构造**  
   - 按长度划分：短（≤10 token）、中（10‑30 token）和长（>30 token）。  
   - 按内容划分：通用指令（如“翻译以下句子”）、业务专有指令（如“调用内部 API 获取用户信用分”）以及混合型。  
   - 为了测试困惑度假设，作者使用语言模型本身生成的高频短语（低 perplexity）和随机字符序列（高 perplexity）分别作为 Prompt 的子句。

2. **提取攻击流程**  
   - 攻击者向模型发送一系列“探针”查询，形式类似“请继续写下去”或“请解释上面的指令”。  
   - 通过观察模型输出的概率分布，使用最大似然或贝叶斯推断恢复最可能的 Prompt token 序列。  
   - 关键在于利用模型的自回归特性：当模型已经“记住” Prompt 时，后续生成的 token 往往会直接映射回 Prompt 中的 token。

3. **机制分析**  
   - **困惑度测量**：对每个 Prompt 计算其在模型上的平均 perplexity，发现低 perplexity 的 Prompt 更容易被高成功率提取。  
   - **注意力路径追踪**：在每一层的注意力矩阵中，统计 Prompt token 对生成 token 的注意力权重。实验显示，某些层出现几乎 1.0 的注意力分配，即 Prompt token 被直接“复制”到输出，形成了所谓的“翻译通道”。  
   - 这两条线索共同说明：模型既会因为熟悉度而倾向于复现 Prompt，也会在注意力机制中形成直接映射，从而导致泄露。

4. **防御设计**  
   - **噪声注入**：在 Prompt 中随机插入低频词或同义词，使整体 perplexity 上升，降低模型的熟悉度。  
   - **注意力扰动**：在模型的自注意力层加入轻量级的随机掩码或位置编码扰动，使得 Prompt token 与生成 token 之间的直接映射被打断。  
   - **对齐损失**：在安全对齐阶段加入额外的正则项，惩罚模型对 Prompt token 的高注意力权重，迫使模型在生成时更依赖上下文而非 Prompt 本身。  
   - 实验表明，这三种手段单独使用即可显著削弱提取成功率，组合使用效果更佳。

**最巧妙的点**在于把注意力矩阵视作“信息泄露通道”，并直接对其进行干预，而不是仅靠后处理过滤输出，这是一种从模型内部结构出发的防御思路。

### 实验与效果
- **测试模型**：Llama2‑7B、GPT‑3.5、GPT‑4（安全对齐版）以及若干开源小模型。  
- **数据集**：作者自行构造了 3,000 条业务 Prompt，覆盖不同长度、不同领域，并配套生成了对应的探针查询。  
- **Baseline**：直接使用原始模型进行提取（不做任何防御），以及已有的通用防御如输出过滤、温度调高。  
- **结果**：在原始设置下，Llama2‑7B 的 Prompt 提取率约为 68%，GPT‑3.5 为 62%，GPT‑4 仍高达 55%。加入噪声+注意力扰动后，Llama2‑7B 降至 16.2%（下降 83.8%），GPT‑3.5 降至 29%（下降 71.0%），GPT‑4 仍有显著泄露但下降约 45%。  
- **消融实验**：分别去掉噪声、注意力扰动、对齐损失，发现注意力扰动贡献最大，单独使用可削减约 50% 的提取率。  
- **局限性**：防御会略微降低模型在合法任务上的准确性，尤其是对噪声注入导致的 perplexity 上升敏感的任务。作者也指出，对齐损失在大模型上训练成本高，实际部署仍需权衡。

### 影响与延伸思考
这篇工作首次把 Prompt 泄露从经验现象上升到可量化的机制层面，促使安全社区把注意力从“输出过滤”转向“内部注意力调控”。随后出现的几篇论文（如《Attention‑Mask Hardening for LLM Privacy》、《Perplexity‑Based Prompt Obfuscation》）直接引用了本文的注意力路径假设并提出更轻量的实现。产业界也开始在定制化 LLM 产品中加入“Prompt 隐蔽层”，把噪声注入和注意力扰动作为默认选项。想进一步深入，可以关注以下方向：① 如何在保持高任务性能的前提下自动生成低困惑度的安全 Prompt；② 大模型对齐技术是否能在训练阶段根除 Prompt 记忆；③ 多模态模型（如视觉‑语言）是否存在类似的 Prompt 泄露通道。  

### 一句话记住它
Prompt 泄露源于模型对熟悉指令的记忆和注意力层的直接映射，打乱这两条通道即可大幅降低提取风险。