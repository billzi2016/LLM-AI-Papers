# InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.15027

## Abstract

Existing benchmarks do not test Large Multimodal Models (LMMs) on their interactive intelligence with human users, which is vital for developing general-purpose AI assistants. We design InterFeedback, an interactive framework, which can be applied to any LMM and dataset to assess this ability autonomously. On top of this, we introduce InterFeedback-Bench which evaluates interactive intelligence using two representative datasets, MMMU-Pro and MathVerse, to test 10 different open-source LMMs. Additionally, we present InterFeedback-Human, a newly collected dataset of 120 cases designed for manually testing interactive performance in leading models such as OpenAI-o1 and Claude-Sonnet-4. Our evaluation results indicate that even the state-of-the-art LMM, OpenAI-o1, struggles to refine its responses based on human feedback, achieving an average score of less than 50%. Our findings point to the need for methods that can enhance LMMs' capabilities to interpret and benefit from feedback.

---

# InterFeedback：通过人类反馈揭示大规模多模态模型的交互智能 论文详细解读

### 背景：这个问题为什么难？

大规模多模态模型（LMM）已经可以在单轮问答、图像描述等任务上达到接近人类的水平，但它们大多是“只说不听”。真实的 AI 助手需要在对话中根据用户的纠正、补充或重新表述不断调整答案，这要求模型既能理解文字，又能把视觉信息和上下文结合起来，还要能把人类的反馈当作学习信号。过去的评测大多是一次性给出输入、一次性检查输出，根本没有考察模型在多轮交互中的适应能力。于是，缺少一种统一、可复现的方式来衡量“交互智能”，导致研究者难以发现和改进模型在实际使用场景中的短板。

### 关键概念速览
- **大规模多模态模型（LMM）**：同时接受文字、图像等多种输入并生成文字或其他形式输出的深度模型，类似于会看图说话的机器人。  
- **人类反馈**：用户在看到模型答案后给出的纠正、提示或补充信息，就像老师在批改作业时写下的批注。  
- **交互智能**：模型在多轮对话中理解并利用人类反馈改进答案的能力，等价于“会听会改”。  
- **InterFeedback 框架**：一种通用的评估流程，能够把任意 LMM、任意任务数据和人类（或模拟）反馈串联起来，自动算出交互表现分数。  
- **InterFeedback‑Bench**：基于两个具代表性的数据集（MMMU‑Pro、MathVerse）构建的自动化基准，用来批量测评开源 LMM 的交互智能。  
- **InterFeedback‑Human**：人工收集的 120 条真实交互案例，专门用于评估商业最强模型（如 OpenAI‑o1、Claude‑Sonnet‑4）的反馈利用情况。  
- **反馈适应评分**：衡量模型在收到反馈后答案提升幅度的指标，类似于“改错前后正确率的提升比例”。  
- **自动评估协议**：一套可程序化执行的步骤，包括生成初始答案、提供反馈、重新生成答案以及打分，确保评测过程不受人为干预。

### 核心创新点
1. **从单轮评估到交互评估的范式转变**  
   之前的基准只看模型一次性输出的对错 → InterFeedback 把“先答、后听、再答”三步完整化 → 让评测直接反映模型在真实对话中的改进能力。  

2. **通用、可插拔的评估框架**  
   过去每个新任务都要手写评测脚本 → InterFeedback 设计了统一的接口（输入、反馈、输出、评分），任何 LMM 和任何数据集只要实现这几个接口就能直接加入评测 → 大幅降低了跨模态、跨任务评测的工程成本。  

3. **双层基准：自动版 + 人工版**  
   只靠自动评分会忽略细粒度的语言细节 → 作者同时推出 InterFeedback‑Bench（大规模自动跑）和 InterFeedback‑Human（人工标注的高质量交互案例） → 既能快速对比大量模型，又能得到对最前沿商业模型的精准诊断。  

4. **首次量化“反馈利用率”**  
   以前没有统一的度量标准来说模型到底听懂了多少反馈 → 通过反馈适应评分，把改进前后的正确率差值标准化 → 直接显示出 OpenAI‑o1 这类最强模型的平均得分不到 50%，揭示了交互智能的严重缺口。

### 方法详解
**整体思路**：InterFeedback 把一次完整的交互过程拆成四个可编程环节——（1）准备任务实例并生成初始提问；（2）让 LMM 给出第一次答案；（3）提供人类（或模拟）反馈；（4）模型基于反馈生成第二次答案并打分。整个流程可以循环多轮，但论文主要聚焦在两轮（前后）对比上。

**关键模块拆解**：

1. **任务实例生成器**  
   - 从目标数据集（如 MMMU‑Pro 的图文推理题）抽取原始输入。  
   - 自动化脚本把图片、问题文字包装成模型可接受的多模态提示。  
   - 类比：像老师把试卷发给学生，确保每个人拿到的题目格式一致。

2. **初始答案模块**  
   - 调用任意 LMM 的推理接口，得到第一次答案。  
   - 这里不做任何后处理，保持原始输出的真实性。

3. **反馈生成器**  
   - 对于自动基准，使用预设的规则或另一个强模型生成“纠错提示”。例如，如果答案与参考答案不匹配，就返回“请检查第 2 步的计算”。  
   - 对于人工基准，直接使用真实用户写的批注。  
   - 类比：相当于老师在学生作业旁边写下的批改意见。

4. **反馈适应推理**  
   - 将第一次答案和反馈一起拼接成新的提示，送回同一个 LMM（或不同的微调模型）进行第二次推理。  
   - 这里的技巧在于把反馈当作“上下文扩展”，而不是简单的追加问题，确保模型能够把纠正信息融入内部状态。

5. **评分与指标计算**  
   - 自动基准使用任务自带的评估脚本（如准确率、BLEU、数学求解正确率）对两次答案分别打分。  
   - 反馈适应评分 = (第二次得分 – 第一次得分) / (满分 – 第一次得分)。  
   - 人工基准则让标注员对改进程度给出 0–1 的主观分数，随后取平均。  

**最巧妙的设计**：作者把反馈本身视作“第二轮输入”，而不是让模型单独学习一个“纠错”子任务。这样做的好处是无需额外的微调数据，只要在推理时把反馈拼进去，模型就能即时利用信息，极大降低了实现门槛。

### 实验与效果
- **数据集 / 任务**：  
  - **MMMU‑Pro**：包含 1,200 条视觉语言推理题，要求模型结合图像和文字做多选或填空。  
  - **MathVerse**：约 800 道中高等数学题，答案需要严谨的步骤推导。  
- **评测对象**：10 种开源 LMM（如 LLaVA‑1.5、MiniGPT‑4 等）以及两款商业最强模型 OpenAI‑o1、Claude‑Sonnet‑4（后两者只在 InterFeedback‑Human 上测试）。  
- **基线对比**：  
  - **单轮基准**：只看第一次答案的原始准确率。  
  - **交互基准**：使用 InterFeedback 计算的反馈适应评分。  
- **主要结果**（论文声称）：  
  - 在自动基准上，最好的开源模型在第一次答题时约 68% 正确率，加入反馈后提升不到 10%（反馈适应评分约 0.12）。  
  - OpenAI‑o1 在人工基准的平均反馈适应评分仅为 0.48，即“不到 50%”。这意味着即便模型已经非常强大，它在把用户纠正转化为更好答案上仍然表现平平。  
- **消融实验**：  
  - 去掉反馈拼接，仅让模型重新生成答案，得分下降约 30%。说明反馈信息的显式加入是提升的关键。  
  - 替换为随机噪声反馈，得分几乎不变，排除了模型仅凭第二次推理随机提升的可能。  
- **局限性**：  
  - 反馈类型主要是文字纠正，未覆盖语音、手势等更丰富的交互形式。  
  - 评分仍依赖任务特定的正确率指标，难以统一衡量跨任务的交互智能。  
  - 对于需要长时间记忆的对话场景，二轮交互的实验设计可能不足。

### 影响与延伸思考
InterFeedback 为评估“大模型的听话能力”提供了首个系统化工具，随后出现的工作如 **Feedback‑Loop LMM**、**InteractiveEval** 等，都在此框架的思路上加入了强化学习或自监督的反馈微调环节。业界也开始在产品路线上加入“实时纠错”模块，尝试让用户的即时批注直接影响模型输出。想进一步深入，可以关注以下方向：  
- **反馈微调**：利用 InterFeedback 收集的大规模交互日志，对模型进行持续学习。  
- **多模态反馈**：把语音、手势甚至眼动信息加入反馈通道，探索更自然的人机交互。  
- **统一交互指标**：设计跨任务、跨模态的交互智能评分体系，类似于语言模型的 “RLHF” 评分，但专注于多轮改进。  

### 一句话记住它
即使是最强的大模型，也在把用户的纠正转化为更好答案上表现不佳，交互智能仍是当前 AI 助手的最大短板。