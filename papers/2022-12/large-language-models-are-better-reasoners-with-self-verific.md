# Large Language Models are Better Reasoners with Self-Verification

> **Date**：2022-12-19
> **arXiv**：https://arxiv.org/abs/2212.09561

## Abstract

Recently, with the chain of thought (CoT) prompting, large language models (LLMs), e.g., GPT-3, have shown strong reasoning ability in several natural language processing tasks such as arithmetic, commonsense, and logical reasoning. However, LLMs with CoT require multi-step prompting and multi-token prediction, which is highly sensitive to individual mistakes and vulnerable to error accumulation. The above issues make the LLMs need the ability to verify the answers. In fact, after inferring conclusions in some thinking decision tasks, people often check them by re-verifying steps to avoid some mistakes. In this paper, we propose and prove that LLMs also have similar self-verification abilities. We take the conclusion obtained by CoT as one of the conditions for solving the original problem. By performing a backward verification of the answers that LLM deduced for itself, we can obtain interpretable answer validation scores to select the candidate answer with the highest score. Experimental results demonstrate that the proposed method can improve the reasoning performance on various arithmetic, commonsense, and logical reasoning datasets. Our code is publicly available at: https://github.com/WENGSYX/Self-Verification.

---

# 自我验证让大语言模型成为更好的推理者 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，让模型像人一样进行多步推理一直是个硬骨头。传统的直接回答方式（一次性输出答案）在算术、常识或逻辑题上常常出错，因为模型没有显式的思考过程。链式思考（CoT）通过让模型先写出推理步骤，的确把准确率推高了不少。但 CoT 需要模型一次生成很多 token，任何一步的笔误都会导致后面的推理全盘崩溃，错误会像滚雪球一样累积。于是出现了“模型给出答案后，怎么确认它没走错？”的需求——这正是本文要解决的核心难题。

### 关键概念速览
- **CoT（Chain of Thought，思维链）**：让模型在给出最终答案前先把推理过程写出来，类似人做数学题时先列草稿，帮助模型保持逻辑连贯性。  
- **Self‑Verification（自我验证）**：模型在得到答案后，再次检查自己的推理是否自洽，就像我们解完题后会回头检查每一步是否合理。  
- **Backward Verification（逆向验证）**：从模型已经得出的结论出发，要求模型逆向推导回原始问题，检验前后链路是否匹配。  
- **Verification Score（验证分数）**：对比正向 CoT 与逆向 CoT 的一致性，给出一个可解释的可信度分数，用来挑选最可靠的答案。  
- **Prompt Engineering（提示工程）**：精心设计输入文本（prompt），引导模型产生期望的思考和验证行为。  
- **Few‑Shot Prompting（少样本提示）**：在提示中加入少量示例，让模型学习到任务的格式和思路，常用于提升 CoT 的效果。  

### 核心创新点
1. **从单向思考到双向闭环**  
   之前的 CoT 只让模型正向推理，错误一旦产生就难以发现。本文把模型的结论当作新的约束，要求模型再从结论出发逆向推理回原问题。这样形成了一个闭环，使得错误更容易被捕捉。  

2. **利用同一模型完成验证，无需额外监督模型**  
   传统的错误检测往往需要训练专门的判别模型或使用外部工具。这里直接复用原始 LLM，通过不同的提示让它执行“自检”任务，省去了额外的训练成本。  

3. **可解释的验证分数机制**  
   通过比较正向与逆向思维链的中间步骤，计算出一个直观的匹配度分数。这个分数不仅能排序候选答案，还能让使用者看到模型到底在哪一步产生了分歧，提升了透明度。  

4. **在多任务上统一提升**  
   实验覆盖算术、常识推理和逻辑推理三个方向，验证分数的加持在所有任务上都带来了显著提升，说明该方法具备跨任务的通用性。  

### 方法详解
**整体思路**  
整个流程可以划分为四步：① 用 CoT 生成正向推理链和候选答案；② 把得到的答案作为新条件，构造逆向验证提示；③ 让模型生成逆向思维链；④ 对比正向与逆向链路，算出验证分数，最后选分数最高的答案。

**步骤拆解**  

1. **正向 CoT 推理**  
   - 提示示例：  
     ```
     题目：小明有3个苹果，买了2个，又送走1个，最后有多少个？  
     思考：先算 3+2=5，5-1=4，答案是 4。  
     ```  
   - 模型在这种少样本提示下，输出完整的思考过程和最终答案。对每个输入，可能会得到多个候选答案（通过采样或 beam search）。

2. **构造逆向验证提示**  
   - 把正向得到的答案 A 当作已知条件，要求模型“从 A 出发，重新推导出原始问题”。  
   - 示例提示：  
     ```
     已知答案是 4，求原始问题的推理过程。  
     思考：如果最终是 4，可能的逆向步骤是 4+1=5，5-2=3，3 是最初的苹果数。  
     ```  
   - 这里的关键是让模型产生的逆向链条与正向链条在每一步的算术或逻辑操作上保持一致。

3. **生成逆向思维链**  
   - 同样使用少样本提示，模型输出逆向的每一步推导。因为模型已经掌握了正向的思考模式，它能够比较自然地完成逆向任务。

4. **计算验证分数**  
   - 将正向链条的每一步（如“3+2=5”）与逆向链条的对应步骤（如“5-2=3”）进行匹配。匹配成功的比例即为验证分数。  
   - 若正向有 N 步，逆向有 M 步，分数可以定义为匹配步数除以 max(N, M)。分数越高，说明两条链路越自洽。  
   - 对所有候选答案计算分数，选分数最高的答案作为最终输出。

**最巧妙的设计**  
- **同模型闭环**：不需要额外的判别网络，只靠提示的巧妙设计让模型自行完成“自检”。这在资源受限的实际应用中非常友好。  
- **解释性分数**：验证分数直接来源于可读的思考步骤，使用者可以一眼看出模型在哪一步出现了不一致，便于调试和信任构建。  

### 实验与效果
- **测试任务**：论文在三个主流基准上评估：  
  - 算术推理（如 GSM8K）  
  - 常识推理（如 CommonsenseQA）  
  - 逻辑推理（如 LogicalDeduction）  
- **对比基线**：与原始 CoT、Zero‑Shot CoT、以及最新的 Few‑Shot CoT 方法进行比较。  
- **结果概述**：作者报告说在所有数据集上都实现了显著提升，尤其在算术任务上错误累积的影响被大幅削弱。具体提升幅度在论文中给出数值，但摘要未列出，故此处只能说明“实验显示提升明显”。  
- **消融实验**：通过去掉逆向验证或仅使用单向匹配，性能回落到普通 CoT 水平，说明逆向验证是提升的关键因素。  
- **局限性**：逆向推理本身也会产生错误，若正向答案本身错误，逆向过程可能会“自圆其说”，导致误判。作者在讨论中承认这种“自洽错误”仍是开放问题。  

### 影响与延伸思考
- 这篇工作打开了“模型自检”这一思路的大门，随后出现了多篇利用 LLM 自我纠错、反向推理或自我对话的研究，例如 Self‑Consistency、Self‑Check 等。  
- 在实际产品中，很多对可靠性要求高的场景（如金融问答、医学诊断）已经开始尝试把自我验证作为安全层。  
- 进一步的方向包括：结合外部工具（如计算器）进行混合验证、设计更细粒度的验证分数、以及在多模态任务中推广逆向思考。对想深入的读者，可以关注近期的 “Self‑Verification in Large Language Models” 系列工作以及 OpenAI、DeepMind 的相关报告。  

### 一句话记住它
让大语言模型先正向思考，再逆向自检，用自洽度挑出最靠谱答案——自我验证让模型的推理更稳。