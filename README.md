# agent-from-scratch

从零开始的Harness Agent

## DONE

### 1. API LAYER

- 自定义的消息格式类型，方便接入各api厂商
- 接入openai协议api，支持流式/非流式输出


### 2. AGENT LAYER

#### 2.1 Agent Loop
- agent loop with output events
- status bar(eg. system time, todo list)

#### 2.2 Hooks
- PreTool
- PostTool
- Stop


### 3. HARNESS LAYER

#### 3.1 Tools
- 8 builtin tools: read, write, glob, edit, bash, ls, todo, load_skill
- 工具执行流程：（1）参数校验（2）权限校验（3）PreTool（4）工具执行（5）PostTool
- 5种权限校验：（1）硬拒绝（2）工作区外拒绝（3）读允许（4）写询问（5）bash询问

### 3.2 session
- store local

### 3.3 skill


### 4. TUI LAYER
- Textual based tui
- session sidebar
- streamed chat area
- model select & token usage & thinking level

## TODO
- 兼容不同模型的思考模式
- audit
- mcp
- context compression
- memory
- multi-agent
