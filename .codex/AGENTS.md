# Codex CLI 中转站（API Relay）接入指南

> **受众**：AI Agent / 开发者。目标：用「自定义 `model_provider`」的方式，让 Codex CLI 走一个 OpenAI 兼容的中转站 API，而不是官方 OpenAI。
> 本文先讲**通用方法**，再把 **maolao 中转站**作为完整示例。复制通用步骤、替换占位符即可用于任意中转站。

---

## TL;DR

Codex CLI 的中转站配置有**三个铁律**，违反任何一条都会导致「以为配了但没生效」：

1. **provider 必须定义在全局** `~/.codex/config.toml` —— 写在项目级 `.codex/config.toml` 里的 `model_providers` 会被 Codex **静默忽略**（`codex doctor` 会报警）。
2. **定义 ≠ 激活**：`[model_providers.xxx]` 只是定义；还要在某处写 `model_provider = "xxx"` 才会真正使用它（用 profile 来做）。
3. **key 通过环境变量注入**：provider 定义里写 `env_key = "<VAR_NAME>"`，然后 `export <VAR_NAME>=sk-...`。Codex 自身的 ChatGPT 登录态**不会**喂给中转站。

一句话启动：`codex --profile <id>`（profile 负责激活 provider）。

---

## 1. Codex 配置加载规则（必须先懂，否则白配）

| 来源 | 路径 | 能否定义/激活 provider | 说明 |
|------|------|------------------------|------|
| 全局 base | `~/.codex/config.toml` | ✅ **唯一可靠的定义处** | provider 定义 + 默认 model/信任级别都放这 |
| Profile | `~/.codex/<name>.config.toml` | ✅ 用来**激活** provider | `codex --profile <name>` 把它叠加到 base 上 |
| 命令行覆盖 | `-c key=value` | ✅ 临时 | 如 `-c model_provider="xxx"` |
| **项目级** | `<proj>/.codex/config.toml` | ❌ **`model_providers` 被忽略** | 只能放 `model` / `service_tier` 等非 provider 字段 |

**注意事项**：

- `--profile` 只对**运行时命令**生效：`codex`、`codex exec`、`codex review`、`codex resume` …；**不对 `codex doctor` 生效**。别用 doctor 验证 profile。
- 环境变量 `CODEX_HOME` 决定上面所有 `~/.codex` 的实际位置（本项目为 `/root/.codex`）。
- 项目级 `.codex/AGENTS.md`（本文件）会被读作项目指令；但**不要**在这里放 provider 定义。

---

## 2. 通用配置步骤（4 步，适用于任意中转站）

设中转站标识为 `<id>`（自定义，如 `maolao`/`openrouter`/`deepbricks`），key 环境变量为 `<VAR>`。

### 步骤 1：在全局 config 定义 provider

编辑 `~/.codex/config.toml`，添加：

```toml
[model_providers.<id>]
name = "<显示名>"
base_url = "<中转站 OpenAI 兼容端点，通常以 /v1 结尾>"
env_key = "<VAR>"                 # Codex 从这个环境变量读 key
wire_api = "responses"            # 见下方说明，另一个选择是 "chat"
```

### 步骤 2：建 profile 来激活它

新建 `~/.codex/<id>.config.toml`：

```toml
# Usage: codex --profile <id>
# Key:   export <VAR>=sk-...
model_provider = "<id>"           # 这一行 = 激活
model = "<中转站支持的模型 id>"
model_reasoning_effort = "medium" # low|medium|high|xhigh，可选
```

### 步骤 3：注入 key（务必持久化）

```bash
echo 'export <VAR>=sk-你的key' >> ~/.bashrc   # 或 ~/.zshrc
source ~/.bashrc
```

> 只在终端临时 `export` 而不写进 rc，是新 shell 里 key 丢失、调用失败的常见原因。

### 步骤 4：使用 + 验证

```bash
codex --profile <id>              # 正式启动
codex --profile <id> exec "ping"  # 非交互验证：观察输出里的 provider 行
```

---

## 3. 关键字段说明

### `env_key`
Codex 把该环境变量的值作为 `Authorization: Bearer <值>` 发给中转站。与 Codex 本身的 ChatGPT OAuth 登录**互不影响**——切到自定义 provider 后只走 `env_key`。

### `wire_api`
中转站要兼容 OpenAI 的哪种端点：

- `"responses"` —— OpenAI 较新的 `/responses` 端点（gpt-5.x / o 系列默认走这个）。
- `"chat"` —— 经典 `/chat/completions`，**兼容性最好**。

不确定就先试 `responses`；若报 model 不存在 / 路由 404 / 端点错误，改成 `"chat"` 再试。

### `model`
必须填**中转站实际支持**的模型 id（不一定是 OpenAI 官方名，有些中转站会改名）。在中转站后台查可用模型列表。

---

## 4. 完整示例：maolao 中转站

实际跑通的配置（占位：`<id>=maolao`，`<VAR>=MAOLAO_API_KEY`）。

### `~/.codex/config.toml`（节选，provider 定义）
```toml
[model_providers.maolao]
name = "Maolao API"
base_url = "https://maolaoapi.com/v1"
env_key = "MAOLAO_API_KEY"
wire_api = "responses"
```

### `~/.codex/maolao.config.toml`（profile，激活 provider）
```toml
# CLI-only profile for the Maolao relay.
# Usage:  codex --profile maolao
# Key:    export MAOLAO_API_KEY=sk-...   (read via env_key in ~/.codex/config.toml)
model_provider = "maolao"
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"
```

### key（持久化到 shell）
```bash
echo 'export MAOLAO_API_KEY=sk-你的maolao_key' >> ~/.bashrc
source ~/.bashrc
```

### 使用
```bash
cd /root/max/VISTA-Skill
codex --profile maolao
```

---

## 5. 验证清单（怎么确认真的通了）

```bash
# A. 看是否切到中转站 provider（应输出 "provider: maolao"）
codex --profile maolao exec "say hi" 2>&1 | grep -i provider

# B. base config 健康度（parse ok、无 env_key 相关报错）
codex doctor 2>&1 | grep -iE 'parse|config.*load|error'

# C. 实际发一条消息，收到模型回复即通
codex --profile maolao
```

**预期信号**：

- 配对但缺 key → `Missing environment variable: MAOLAO_API_KEY`（说明 provider 已激活，只差 key）
- `provider: maolao` → 链路切换成功
- 收到模型回复 → 全通

---

## 6. 常见坑（均已踩过）

| 现象 | 原因 | 解决 |
|------|------|------|
| 配了 `.codex/config.toml` 却没生效，`codex doctor` 报 `Ignored unsupported project-local config keys ... model_providers` | Codex 不读项目级 provider 定义 | 把 `[model_providers.*]` 搬到**全局** `~/.codex/config.toml` |
| 定义了 provider，但实际还走 `openai` | 只定义没激活 | profile 里写 `model_provider = "<id>"`，或全局顶层写（后者会全局生效） |
| `provider: maolao` 但调用 401 | key 错 / 无权限 / 未 export | 检查 `echo $<VAR>`，确认写进了 `~/.bashrc` |
| model 报 404 / 路由错误 / wire_api 报错 | 中转站不支持 `/responses` | `wire_api` 从 `"responses"` 改 `"chat"` |
| 每次开新终端就调不通 | key 只临时 export | `echo 'export ...' >> ~/.bashrc` 持久化 |
| `codex --profile <id> doctor` 不反映 profile | `--profile` 不对 doctor 生效 | 用 `codex --profile <id> exec ...` 验证 |

---

## 7. 进阶：进目录自动切换中转站

嫌每次 `--profile maolao` 麻烦，可用 **direnv** 在进入项目时自动激活：

```bash
# apt install direnv  并在 ~/.bashrc 加：eval "$(direnv hook bash)"
cd /root/max/VISTA-Skill
echo 'export CODEX_PROFILE=maolao' > .envrc    # 或 export CODEX_PROFILE；具体以本机 Codex 版本支持为准
direnv allow
```

> 注：`CODEX_PROFILE` 环境变量是否被自动当作 `--profile`，取决于 Codex 版本；不可靠时退回 `codex --profile maolao` 或 shell alias。
