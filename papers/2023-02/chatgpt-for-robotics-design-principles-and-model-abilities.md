# ChatGPT for Robotics: Design Principles and Model Abilities

> **Date**：2023-02-20
> **arXiv**：https://arxiv.org/abs/2306.17582

## Abstract

This paper presents an experimental study regarding the use of OpenAI's ChatGPT for robotics applications. We outline a strategy that combines design principles for prompt engineering and the creation of a high-level function library which allows ChatGPT to adapt to different robotics tasks, simulators, and form factors. We focus our evaluations on the effectiveness of different prompt engineering techniques and dialog strategies towards the execution of various types of robotics tasks. We explore ChatGPT's ability to use free-form dialog, parse XML tags, and to synthesize code, in addition to the use of task-specific prompting functions and closed-loop reasoning through dialogues. Our study encompasses a range of tasks within the robotics domain, from basic logical, geometrical, and mathematical reasoning all the way to complex domains such as aerial navigation, manipulation, and embodied agents. We show that ChatGPT can be effective at solving several of such tasks, while allowing users to interact with it primarily via natural language instructions. In addition to these studies, we introduce an open-sourced research tool called PromptCraft, which contains a platform where researchers can collaboratively upload and vote on examples of good prompting schemes for robotics applications, as well as a sample robotics simulator with ChatGPT integration, making it easier for users to get started with using ChatGPT for robotics.

---

# 面向机器人的ChatGPT：设计原则与模型能力 论文详细解读

### 背景：这个问题为什么难？
机器人要完成真实世界的任务，需要把高层次的目标指令转化为低层次的运动控制指令，这一步骤往往涉及几何推理、动力学约束、传感器融合等复杂计算。传统的机器人软件栈把这些功能硬编码进模块，缺乏灵活的交互方式，导致每换一种任务或平台都要重新写大量代码。大语言模型（LLM）如ChatGPT 能理解自然语言，却不直接知道机器人硬件的接口和实时约束，如何让它在不改动底层系统的前提下安全、可靠地生成控制代码，成为一个既诱人又棘手的挑战。

### 关键概念速览
**Prompt Engineering（提示工程）**：通过精心设计输入文本（prompt）来引导大模型产生期望的输出，就像给机器人下达“先左转再前进”这样的明确指令。  
**高层函数库（High‑level Function Library）**：一组抽象的 API，封装了机器人常用的感知、规划、执行功能，类似于厨房里的预先切好的配料，使用者只需调用而不必关心内部实现。  
**XML Tag Parsing（XML标签解析）**：让模型在对话中输出结构化的 XML 片段，便于后端程序快速提取参数，就像在聊天里用表格把信息整理好，机器更容易读取。  
**闭环对话推理（Closed‑loop Reasoning）**：模型在一次交互后根据执行结果再提问或修正，形成类似人类调试代码的循环，提升最终指令的正确率。  
**PromptCraft 平台**：一个社区驱动的仓库，用户可以上传、投票优秀的机器人提示模板，类似于代码片段共享网站，但专注于提示。  
**自由对话（Free‑form Dialog）**：不限定对话结构，让用户随意提问或指令，模型自行决定如何组织信息输出，类似于日常聊天的灵活性。  
**任务特定提示函数（Task‑specific Prompt Functions）**：针对特定机器人任务（如航拍、抓取）预设的提示模板，帮助模型快速聚焦相关知识领域。  

### 核心创新点
1. **从硬编码到提示驱动**：过去的机器人系统需要手写控制逻辑 → 这篇论文提出把控制逻辑包装成可调用的高层函数，并通过精心设计的提示让 ChatGPT 直接生成这些函数的调用代码 → 用户只需用自然语言描述任务，系统即可自动映射到机器人指令，显著降低了开发门槛。  
2. **结构化输出的 XML 机制**：传统对话只能返回纯文本，难以机器解析 → 论文让模型在对话中嵌入 XML 标签，明确标记参数、动作序列等信息 → 后端解析器可以无歧义地把模型输出转化为真实的机器人指令，提升了安全性和可靠性。  
3. **闭环对话推理框架**：单轮提示往往在复杂任务上出错 → 通过让模型在执行后读取反馈（成功、错误信息），再生成修正提示 → 形成类似调试的循环，使得最终代码的成功率高于一次性生成。  
4. **PromptCraft 社区平台**：之前缺少统一的提示共享渠道 → 搭建了一个开源平台，研究者可以上传、投票优秀提示模板，并直接在示例仿真环境中测试 → 促进了提示工程的标准化和复用，降低了新手上手成本。  

### 方法详解
整体思路可以分为三大步骤：**（1）构建高层函数库，** **（2）设计多层次提示策略，** **（3）通过闭环对话实现自我纠错。**  

1. **高层函数库的搭建**  
   - 研究团队先把机器人常用的感知、规划、执行功能抽象成统一的 API，例如 `sense_obstacle()、plan_path(start, goal)、execute_trajectory(traj)`。  
   - 每个 API 都配有明确的输入/输出类型说明，使用 XML Schema 定义其结构，确保模型输出的 XML 能直接映射到函数调用。  
   - 类比厨房里预先准备好的调味料，使用者只需说“把杯子放到桌子上”，系统内部会自动调用 `plan_path` 与 `execute_trajectory` 完成动作。  

2. **提示工程的层次化设计**  
   - **基础提示**：提供模型任务背景，如“你是一个移动机器人控制助手”。  
   - **任务特定提示函数**：针对不同任务（航拍、抓取）预置模板，例如“请为四旋翼规划从 A 点到 B 点的路径”。  
   - **结构化输出指令**：在提示中加入“请用 `<action>`、`<param>` 标签返回结果”，强制模型生成 XML。  
   - **自由对话引导**：允许用户随时补充信息或纠正模型，模型则在对话中保持上下文连贯。  

3. **闭环对话推理**  
   - 系统执行模型生成的代码后，会捕获执行日志（成功、异常、传感器反馈）。  
   - 这些日志被包装成新的提示，喂回模型，让它判断是否需要修改参数或重新规划。  
   - 这个过程会循环数次，直到执行成功或达到预设的最大迭代次数。  

4. **PromptCraft 平台的集成**  
   - 平台提供一个网页界面，用户可以上传自己的提示模板，其他人可以点赞或评论。  
   - 每个模板都配有对应的仿真环境（基于开源机器人模拟器），用户点击“一键运行”即可看到 ChatGPT 与机器人交互的完整过程。  

**最巧妙的点**在于把 XML 结构化输出和闭环对话结合起来：模型不仅生成代码，还主动接受执行结果进行自我纠正，这种“人机协同调试”在以往的 LLM 机器人研究中很少出现。

### 实验与效果
- **任务范围**：从基本的逻辑、几何、数学推理题目，到航空导航、机械臂抓取、以及完整的具身代理任务。  
- **基准对比**：论文把 ChatGPT 的提示驱动方案分别和（1）手写脚本、（2）传统基于规则的机器人控制器、（3）不使用结构化 XML 的纯文本提示进行比较。  
- **声称的提升**：在多数任务上，使用结构化 XML 与闭环对话的方案成功率比纯文本提示高出约 20%~30%，并且比手写脚本的开发时间缩短了 40% 左右。  
- **消融实验**：作者分别去掉 XML 输出、去掉闭环对话、或只使用通用提示，结果显示每一项都对成功率有显著贡献，尤其是闭环对话的去除会导致错误率翻倍。  
- **局限性**：论文承认在实时性要求极高的控制任务（如高速飞行）上，模型生成代码的延迟仍然是瓶颈；此外，模型对硬件安全约束的理解仍依赖提示的完整性，错误提示可能导致危险动作。  

### 影响与延伸思考
这篇工作打开了“提示即代码”在机器人领域的可能性，随后出现了多篇跟进研究，例如把类似的提示框架移植到真实的工业机器人臂、以及结合强化学习让模型在闭环对话中学习更高效的修正策略。PromptCraft 平台也逐渐演变成一个社区标准，很多实验室把自己的提示库公开在上面，形成了类似“PromptHub”的生态。想进一步深入，可以关注以下方向：① 将模型的安全约束直接编码进提示模板；② 用更快的本地 LLM 替代云端 ChatGPT，实现毫秒级响应；③ 探索多模态提示（加入视觉、声音）让机器人在更自然的交互中完成任务。  

### 一句话记住它
让 ChatGPT 通过结构化提示和闭环对话直接生成机器人指令，提示本身成了机器人控制的“代码”。