# agent-from-scratch

从零开始的Harness Agent

## DONE

### 1. API LAYER

- 自定义的消息格式类型，方便接入各api厂商
- 接入openai协议api，支持流式/非流式输出

### 2. AGENT LAYER

#### 2.1 agent loop
- agent loop with output events
- status bar(eg. system time, todo list)

#### 2.2 Tools
- 5类内置工具：read/write/glob/edit/bash
- 工具执行流程：（1）参数校验（2）权限校验（3）PreTool（4）工具执行（5）PostTool
- 5种权限校验：（1）硬拒绝（2）工作区外拒绝（3）读允许（4）写询问（5）bash询问

#### 2.3 Hooks
- PreTool
- PostTool
- Stop

### 3. harness层
- 实现AgentEvent、Agent管理agentLoop
- session and store(local)

### 4. TUI LAYER
- Textual based tui

## TODO
- 兼容不同模型的思考模式
- audit
- skill
- mcp
- context compression
- memory
- multi-agent
