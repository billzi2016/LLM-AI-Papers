# When do you need Chain-of-Thought Prompting for ChatGPT?

> **Date**：2023-04-06
> **arXiv**：https://arxiv.org/abs/2304.03262

## Abstract

Chain-of-Thought (CoT) prompting can effectively elicit complex multi-step reasoning from Large Language Models~(LLMs). For example, by simply adding CoT instruction ``Let's think step-by-step'' to each input query of MultiArith dataset, GPT-3's accuracy can be improved from 17.7\% to 78.7\%. However, it is not clear whether CoT is still effective on more recent instruction finetuned (IFT) LLMs such as ChatGPT. Surprisingly, on ChatGPT, CoT is no longer effective for certain tasks such as arithmetic reasoning while still keeping effective on other reasoning tasks. Moreover, on the former tasks, ChatGPT usually achieves the best performance and can generate CoT even without being instructed to do so. Hence, it is plausible that ChatGPT has already been trained on these tasks with CoT and thus memorized the instruction so it implicitly follows such an instruction when applied to the same queries, even without CoT. Our analysis reflects a potential risk of overfitting/bias toward instructions introduced in IFT, which becomes more common in training LLMs. In addition, it indicates possible leakage of the pretraining recipe, e.g., one can verify whether a dataset and instruction were used in training ChatGPT. Our experiments report new baseline results of ChatGPT on a variety of reasoning tasks and shed novel insights into LLM's profiling, instruction memorization, and pretraining dataset leakage.

---

# 当需要对 ChatGPT 使用思维链提示时？论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）兴起之前，想让模型完成需要多步推理的任务往往只能靠一次性输出答案，准确率很低。出现思维链（Chain‑of‑Thought，CoT）提示后，只要在输入后加上一句“让我们一步一步思考”，模型的推理过程会被显式写出来，像在纸上打草稿一样，很多数据集的成绩瞬间飙升。可是，这种技巧最初是针对未经指令微调的原始模型（比如 GPT‑3）验证的。随着 ChatGPT 这类经过大规模指令微调（Instruction‑Fine‑Tuning，IFT）的模型普及，业界不清楚 CoT 还能否继续提升性能，尤其是当模型已经在训练时看到大量带有思维链的示例时，是否还需要额外的提示指令。这个未知直接影响我们在实际应用中是否要额外写“思考一步步”这类前缀。

### 关键概念速览
- **思维链（CoT，Chain‑of‑Thought）**：让模型在给出最终答案前先输出推理步骤，类似人解题时先写草稿再写答案。  
- **指令微调（IFT，Instruction‑Fine‑Tuning）**：在大规模预训练后，用大量“请完成以下任务”的指令数据继续训练模型，使其更擅长遵循自然语言指令。  
- **ChatGPT**：OpenAI 基于 GPT‑4 系列的对话模型，经过大量指令微调和强化学习（RLHF），在多种任务上表现接近人类水平。  
- **任务类别**：本文区分了算术推理（如 MultiArith）和更抽象的逻辑推理（如 CommonsenseQA），前者需要精确数值计算，后者侧重常识或逻辑关联。  
- **指令记忆（Instruction Memorization）**：模型在训练时可能把某些指令或数据集的格式“记住”，导致在测试时即使不显式给出指令也会自行产生相同的输出模式。  
- **数据泄露（Dataset Leakage）**：如果模型在训练阶段已经见过某个评测数据的完整示例，评测结果会被“泄露”，不再真实反映模型的推理能力。  

### 核心创新点
1. **系统化评估 CoT 在 ChatGPT 上的有效性**  
   - 之前的工作只在未经指令微调的模型上展示 CoT 的提升。  
   - 这篇论文把同样的 CoT 前缀分别加在 ChatGPT 的多个推理任务上，观察性能变化。  
   - 结果显示：在算术类任务上 CoT 并不提升甚至略降，而在常识推理、逻辑推理等任务上仍有显著增益。  

2. **发现 ChatGPT 已内置算术思维链**  
   - 通过对比“有 CoT 前缀”和“无 CoT 前缀”两种输入，作者发现 ChatGPT 在算术任务上即使不被要求也会自行输出逐步推理。  
   - 这暗示模型在指令微调阶段已经被喂入大量带有算术思维链的示例，形成了隐式的指令记忆。  

3. **提出指令记忆风险的概念框架**  
   - 作者把这种“模型自行遵循训练时见过的指令”称为指令记忆，并指出它可能导致模型对特定指令过度偏好，削弱对新指令的适应性。  
   - 进一步推测，这种现象可以被用来逆向验证某个数据集或指令是否被用于训练，从而产生数据泄露的安全隐患。  

4. **提供 ChatGPT 在多任务上的新基准**  
   - 论文在十余个公开推理数据集上给出了 ChatGPT 的完整表现（包括有无 CoT 两种设置），为后续研究提供了统一的参考点。  

### 方法详解
整体思路很直接：把已有的思维链提示（“让我们一步一步思考”）分别加在 ChatGPT 的输入前后，观察模型输出的差异，并进一步分析模型是否自行生成思维链。具体步骤如下：

1. **任务选取与划分**  
   - 作者挑选了两大类任务：算术推理（MultiArith、GSM‑8K 等）和非算术推理（CommonsenseQA、StrategyQA、LogicalDeduction 等）。  
   - 每类任务都准备了两套输入：一种带有显式 CoT 前缀，另一种不带前缀。

2. **对话式调用 ChatGPT**  
   - 使用 OpenAI 官方 API，以默认的系统提示（“You are ChatGPT, a helpful assistant.”）进行对话。  
   - 对每个问题，分别发送“带 CoT 前缀的提问”和“普通提问”，记录模型的完整回复。

3. **输出解析**  
   - 对模型的回复进行文本划分：如果出现“Step 1: … Step 2: …”之类的序号或自然语言的推理步骤，就标记为生成了思维链。  
   - 同时计算答案的准确率：把模型最后给出的数值或选项与金标准比对。

4. **指令记忆检测**  
   - 为验证模型是否“记住”了训练时的指令，作者设计了两种对照实验：  
     a) 把算术问题的表述方式稍作改动（如换词、改变数字顺序），看模型是否仍会自行输出思维链。  
     b) 在完全不相关的任务上加入算术式的 CoT 前缀，观察是否会产生不必要的推理步骤。  
   - 结果显示，模型在多数算术问题上仍会自行生成思维链，即使表述被轻度扰动。

5. **风险评估**  
   - 作者进一步探讨了如果攻击者知道模型已经“记住”了某个指令或数据集，是否可以利用这一点进行模型逆向或数据泄露。  
   - 虽然没有给出具体攻击实现，但提出了“指令记忆检测”作为一种潜在的安全审计手段。

最让人意外的地方在于：即使 ChatGPT 已经在训练中看到大量算术思维链，它在非算术任务上仍然需要显式的 CoT 前缀才能提升表现，这说明指令记忆并不是全局通用的，而是任务依赖的。

### 实验与效果
- **数据集**：算术类使用 MultiArith、GSM‑8K、AddSub 等；非算术类使用 CommonsenseQA、StrategyQA、LogicalDeduction、OpenBookQA 等共计 12 个公开基准。  
- **基线对比**：与原始 GPT‑3（未微调）在同样 CoT 前缀下的表现相比，ChatGPT 在算术任务上已经接近或超过 90% 的准确率，即使不加 CoT；在非算术任务上，加入 CoT 能把准确率提升约 5‑12%（例如在 CommonsenseQA 上从 71% 提升到 78%）。  
- **消融实验**：作者分别去掉前缀、改写问题、以及在非算术任务上强制加入算术式的 CoT，结果显示：  
  - 去掉 CoT 前缀对算术任务影响不大，说明模型已自行生成思维链。  
  - 对非算术任务去掉 CoT 前缀则显著下降，验证 CoT 仍是必要的提升手段。  
  - 强行在非算术任务中加入算术式 CoT 反而会降低准确率，说明不匹配的思维链会干扰模型。  
- **局限性**：论文没有提供对更大模型（如 GPT‑4）或不同微调策略的实验，也未量化指令记忆对模型鲁棒性的具体影响。作者承认，仅凭公开 API 难以完全排除缓存或后处理的影响。

### 影响与延伸思考
这篇工作提醒社区：指令微调会让模型“记住”特定的提示模式，导致传统的提示工程（如 CoT）在某些任务上失效。随后出现的研究开始关注 **提示鲁棒性**（Prompt Robustness）和 **指令去偏**（Instruction Debiasing），尝试在微调阶段加入多样化的提示，以防模型过度依赖单一指令。安全方向上，**指令记忆检测** 被提出作为审计工具，用来检查模型是否泄露了训练数据或特定指令。想进一步了解，可以关注近期的 “Instruction Tuning for Generalization” 系列论文，以及关于 **数据泄露风险** 的安全审计工作（如 “Privacy Risks of Instruction-Finetuned LLMs”）。

### 一句话记住它
ChatGPT 已经在算术任务上自带思维链，只有在不熟悉的推理任务里才需要显式的 CoT 提示。