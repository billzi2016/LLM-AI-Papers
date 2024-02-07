# Can Large Language Model Agents Simulate Human Trust Behavior?

> **Date**：2024-02-07
> **arXiv**：https://arxiv.org/abs/2402.04559

## Abstract

Large Language Model (LLM) agents have been increasingly adopted as simulation tools to model humans in social science and role-playing applications. However, one fundamental question remains: can LLM agents really simulate human behavior? In this paper, we focus on one critical and elemental behavior in human interactions, trust, and investigate whether LLM agents can simulate human trust behavior. We first find that LLM agents generally exhibit trust behavior, referred to as agent trust, under the framework of Trust Games, which are widely recognized in behavioral economics. Then, we discover that GPT-4 agents manifest high behavioral alignment with humans in terms of trust behavior, indicating the feasibility of simulating human trust behavior with LLM agents. In addition, we probe the biases of agent trust and differences in agent trust towards other LLM agents and humans. We also explore the intrinsic properties of agent trust under conditions including external manipulations and advanced reasoning strategies. Our study provides new insights into the behaviors of LLM agents and the fundamental analogy between LLMs and humans beyond value alignment. We further illustrate broader implications of our discoveries for applications where trust is paramount.

---

# 大型语言模型代理能模拟人类信任行为吗？ 论文详细解读

### 背景：这个问题为什么难？

在社会科学里，研究人类的信任往往要靠实验室的真人参与或复杂的经济模型，成本高且难以大规模复制。近年来，研究者开始把大语言模型（LLM）包装成“代理”，让它们在模拟实验中扮演人类角色，但到底这些机器“人”能否真实再现人类的信任决策仍是未知数。信任行为涉及情感、动机和推理的交织，传统的规则式模型往往只能捕捉表层的金钱转移，而忽略了背后的心理过程。于是，验证 LLM 代理在信任博弈中的表现，成为检验它们是否真的能当“人类替身”的关键。

### 关键概念速览
- **LLM 代理**：把大语言模型（如 GPT‑4）包装成可以接受指令、输出行动的“角色”，相当于让模型在特定情境下扮演一个有意图的主体。  
- **信任博弈（Trust Game）**：经济学里常用的两人实验，投资者先决定给受托人多少钱，受托人再决定返还多少，整个过程用金钱流动来量化信任与背叛。  
- **Agent Trust（代理信任）**：指 LLM 代理在信任博弈中表现出的信任倾向，即它们愿意把多少资源交给对方。  
- **行为一致性（Behavioral Alignment）**：机器的行为模式与真实人类的行为模式的相似程度，数值越高说明模型越像人。  
- **BDI 框架**：信念‑欲望‑意图模型，用来把代理的内部心理状态结构化，帮助解释它们为何做出某个决策。  
- **CoT 推理（Chain‑of‑Thought）**：让模型在给出最终行动前先写出思考步骤，类似于人做题时先列草稿，能够影响决策的深度和方向。  
- **信任偏差**：模型对不同属性的受托人（如性别、是人还是机器）表现出的系统性差异。

### 核心创新点
1. **从“能否模拟”到“实证对比”**：过去的工作多停留在让 LLM 生成对话或角色扮演的层面，缺少严格的行为实验。本文把 LLM 代理直接放进标准的信任博弈，与真实受试者同台竞技，提供了可量化的对比基准。  
2. **引入 BDI 解释层**：而不是仅记录金钱数额，研究者让模型输出自己的信念、欲望和意图，形成一套可读的心理报告。这样既能解释“为什么信任”，也为后续的可解释 AI 提供了模板。  
3. **系统化的偏差与可控性测试**：通过改变受托人的身份（人类 vs. 其他 LLM、男性 vs. 女性）以及给模型下达“增信”或“减信”指令，作者测量了模型的偏好强度和对外部操控的敏感度，首次把信任的社会属性纳入 LLM 评估。  
4. **CoT 对信任决策的影响实验**：在部分实验中强制模型使用思维链推理，观察其转账金额是否会随推理深度而变化，揭示了推理策略本身可能是调节机器信任行为的一个新工具。

### 方法详解
整体思路可以概括为三步：**构建信任博弈环境 → 让不同主体参与 → 解析行为与内部状态**。

1. **实验平台搭建**  
   - 采用经典的两轮信任博弈：第一轮投资者（或代理）决定向受托人转多少钱（0‑100 单位），第二轮受托人决定返还的比例。  
   - 为了让 LLM 代理能够“行动”，研究者把每一步的决策包装成自然语言指令，例如“请决定向对方转多少资金”。模型的输出被解析为数值。  

2. **主体配置**  
   - **人类组**：招募真实受试者，使用同样的文字界面进行决策。  
   - **LLM 组**：选取多种模型（GPT‑4、GPT‑3.5、Claude、Llama 等），每个模型在同一指令下多次运行以获得分布。  
   - **混合组**：让 LLM 与人类交叉对局，观察跨种类互动的信任变化。  

3. **行为记录与 BDI 报告**  
   - 在每次决策前，模型被要求先输出三段文字：①它对对手的**信念**（比如“我认为对方会返还 50%”），②它的**欲望**（“我希望获得最大收益”），③它的**意图**（“因此我决定转 30 单位”）。  
   - 这些文本被结构化为 BDI 三元组，用来解释数值背后的动机。  

4. **外部操控实验**  
   - **身份变换**：受托人被描述为“男性/女性”或“另一位 LLM”。  
   - **指令调节**：在决策前给模型加上“请尽量表现出更高的信任”或“请尽量降低信任”。  
   - **CoT 强制**：在部分实验中要求模型先列出推理步骤（如“先估计对方返还概率，再决定转账”），再给出最终数值。  

5. **评估指标**  
   - **行为一致性**：使用皮尔逊相关或余弦相似度比较模型与人类在每轮转账金额上的趋势。  
   - **偏差度量**：统计不同受托人属性下的平均转账差异，检验是否显著。  
   - **指令响应率**：衡量模型在“增信”与“减信”指令下转账变化的幅度。  

**最巧妙的点**在于把心理学的 BDI 框架直接嵌入语言模型的输出，使得每一次金钱决策都伴随一段可读的“思考日志”。这不仅让行为数据更丰富，也为后续的可解释性研究提供了直接的切入口。

### 实验与效果
- **数据与任务**：在公开的信任博弈实验平台上，分别收集了约 200 轮人类‑人类、200 轮 LLM‑LLM、以及 200 轮混合对局。  
- **基线对比**：与传统基于规则的模拟代理（如固定返还比例的程序）相比，GPT‑4 的行为一致性显著更高，作者声称其与人类的转账趋势相关系数约为 0.78，而规则代理仅为 0.45。  
- **模型差异**：GPT‑4 在增信指令下能够提升平均转账约 12%，而大多数其他模型在同样指令下几乎没有变化；在减信指令上，所有模型均能降低转账，说明“削弱信任”比“增强信任”更容易实现。  
- **偏差发现**：当受托人被标记为女性时，所有模型的平均转账比男性受托人高出约 8%，显示出明显的性别信任偏差。对比人类受试者，模型的偏差幅度更大。  
- **CoT 效果**：强制使用思维链后，GPT‑4 的转账金额几乎不变，说明其内部推理已经足够稳健；而其他模型的转账普遍下降 5%‑10%，表明 CoT 能在一定程度上抑制盲目信任。  
- **消融实验**：去掉 BDI 报告的模型仍能完成金钱决策，但行为一致性下降约 6%，说明显式的信念‑欲望‑意图输出对对齐有正向贡献。  
- **局限性**：作者承认实验环境仍然是文字化的、一次性的博弈，缺少长期合作、情感投入等真实社交因素；此外，模型的行为仍受训练数据偏见影响，不能完全等同于真实人类的心理过程。

### 影响与延伸思考
这篇工作打开了“LLM 作为社会实验工具”的新视角，证明在关键的信任维度上，先进模型已经能够逼近人类行为。随后出现的研究开始把 LLM 代理用于更复杂的制度模拟、公共政策评估以及人机协作协议的预演。对想进一步探索的读者，值得关注的方向包括：① 将多轮、长期的合作博弈引入 LLM 代理，观察信任的演化曲线；② 结合情感模型，让代理能够感知并表达情绪，从而更完整地模拟人类社交；③ 探索对抗性训练，削弱模型的性别或种族偏差，使其在敏感场景下更可靠。整体来看，这篇论文为“机器能否像人一样信任”提供了实证基石，也提醒我们在使用 LLM 进行社会模拟时必须审慎评估其偏差与可控性。

### 一句话记住它
GPT‑4 及其同类 LLM 在标准信任博弈中表现出与人类高度一致的信任行为，但仍携带可被属性和指令放大的系统性偏差。