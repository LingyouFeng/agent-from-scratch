# agent-from-scratch

从零开始的Harness Agent

## DONE

### 1. api层

- 自定义的消息格式类型，方便接入各api厂商
- 接入openai协议api，支持流式/非流式输出

### 2. agent层

#### 2.1 agent循环
- 基本agentloop循环（思考模式）

#### 2.2 工具
- 5类内置工具：read/write/glob/edit/bash
- 工具执行流程：（1）参数校验（2）权限校验（3）PreTool（4）工具执行（5）PostTool
- 5种权限校验：（1）硬拒绝（2）工作区外拒绝（3）读允许（4）写询问（5）bash询问

#### 2.3 Hooks
- PreTool
- PostTool
- Stop

### 3. harness层
- 实现AgentEvent、Agent管理agentLoop

### 4. tui层
- 实现基于Textual的终端界面


## TODO
- 兼容不同模型的思考模式
- status bar
- session and store（local）
- audit
- skill
- mcp
- ui（Tauri + React）
- context compression
- memory
- multi-agent
