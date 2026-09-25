# AuK Studio

本地运行 AuK / AuK-Flash 的语音生成与编辑工作台：FastAPI 后端 + 单文件前端 + 独立模型子进程，双击批处理即可使用，不依赖 Gradio。

包名：`auk-studio`（分发包 / 仓库名）。导入包名仍是 `auk`（`src/auk/`），两者不是一回事。

上游模型本身的文档（论文介绍、基准、权重下载、CLI / Gradio / ComfyUI / Python API、微调）已迁到 [docs/AuK-MODEL.md](docs/AuK-MODEL.md)。本仓库自带 `src/auk/`，**不需要**再克隆官方仓库。

## 能做什么

- **9 个任务视图**：声音克隆、指令 TTS、语音内容编辑、歌词编辑、语速/音量/音调、情感编辑、音色/口音/耳语/非语言声、语音与音乐分离、音频增强；另有功能中心、历史记录、模型与设置。
- **官方句式**：Web 路径不经过 Prompt Enhancer，所以页面自己按 `src/auk/infer/pe.config.yaml` 与 Cookbook 的官方模板拼指令、算目标时长，避免把用户随手写的散文原样喂给模型。内置官方素材案例可一键载入。
- **长文本分段**：`/api/plan` 在不加载模型的前提下预告「分几段、大约多久」，生成时按段回报进度。
- **结果可核对**：每张结果卡带播放器、实际时长 / 目标时长 / 耗时 / seed，可与原音频 A/B 对比，也可带参数重跑；历史页记录本浏览器的生成结果。
- **显存策略开关** + 顶部 CPU/内存/显存资源条 + 深浅色主题 + 取消等待按钮。
- **全链路日志**：服务进程、模型子进程、启动器三份日志，靠 8 位日志编号串起来。

## 环境要求

| 项 | 结论 |
| --- | --- |
| Python | 3.10（`.python-version` 锁的就是它；`requires-python >=3.10`） |
| PyTorch | 2.7.x。uv 会按 `pytorch-cu126` 索引取 CUDA 12.6 轮子（`pyproject.toml` 已配好）；用 pip 装到的是 PyPI 默认轮子，要指定 CUDA 就自己加 `--index-url` |
| 显存 8GB | 可以跑。默认**分页**策略，实测卡在 RTX 3060 Ti 8GB |
| 显存 14GB+ | 才可能用**全常驻**策略（bf16 权重约 12.8GB：Qwen 8.1 + DiT 3.1 + fp32 VAE 0.6，再加 CUDA context 与激活）；30GB 以上宽裕 |
| 内存 / 页面文件 | 本机实测一次分页加载约提交 16.9GB 虚拟内存，可用提交低于 18GB 时启动器会警告。建议 32GB 内存，或把页面文件调大 |
| 磁盘 | 代码仓库约 30MB；权重另计（Base 与 Flash 各一份 safetensors） |

## 安装

```bash
# 1) 建环境（.venv 放在项目根目录，bat 认这个位置）
uv venv --python 3.10                  # 或 python -m venv .venv
.venv\Scripts\activate                 # Windows；Linux/macOS 用 source .venv/bin/activate

# 2) 装 Web 层依赖 + 模型包（可编辑安装，把 src/auk 注册进环境）
uv pip install -r requirements-web.txt
uv pip install -e .
```

用 pip 替换上面的 `uv pip` 也完全可以。不要用 `uv sync`：Web 层依赖写在 `requirements-web.txt`，不在 `[project.dependencies]` 里，`uv sync` 装不出 fastapi。

需要官方 Gradio / Prompt Enhancer / ComfyUI 的额外依赖时再装 `.[gradio]` 或 `.[comfyui]` 这些 extras；本工作台不依赖它们。

**改过包名之后要重装**：本包从 `auk` 改名为 `auk-studio`，旧的可编辑安装会在环境里留下 `auk-0.1.0.dist-info` 与 `__editable__.auk-0.1.0.pth`。二者并存时 `import auk` 可能指向旧路径，请删掉这两个文件后再跑一次 `uv pip install -e .`。

## 准备权重

推荐保持官方目录结构（该目录内容不入库，只留 `.gitkeep`）：

```text
ckpts/
├─ AuK/          config.yaml + auk_base.safetensors
├─ AuK-Flash/    config.yaml + auk_flash.safetensors
└─ Qwen2.5-Omni-3B/
```

下载地址与镜像见 [docs/AuK-MODEL.md → Download the weights](docs/AuK-MODEL.md#download-the-weights)。路径不同时用环境变量指过去：

```powershell
$env:AUK_BASE_CKPT="D:\AuK\ckpts\AuK\auk_base.safetensors"
$env:AUK_BASE_CONFIG="D:\AuK\ckpts\AuK\config.yaml"
$env:AUK_FLASH_CKPT="D:\AuK\ckpts\AuK-Flash\auk_flash.safetensors"
$env:AUK_FLASH_CONFIG="D:\AuK\ckpts\AuK-Flash\config.yaml"
```

## 启动

双击 `启动AuK-Studio.bat`：它先检查 `.venv` 是否存在、`import fastapi, uvicorn, pydantic, auk` 能否通过，再交给 `run.py`。命令行启动等价：

```bash
python run.py                 # 默认 127.0.0.1:8080，就绪后自动打开浏览器
python run.py --no-browser    # 不打开浏览器
python run.py --stop          # 只清理端口上的实例及其模型子进程
```

然后打开 `http://127.0.0.1:8080`，健康检查 `http://127.0.0.1:8080/api/health`。

启动器语义（`run.py`）：

- 端口上已经是 AuK 服务 → **结束旧实例、用当前代码重启**（若旧实例跑的是被改之前的 `backend/app.py`，会额外提示）。
- 端口上是不相干的进程 → 报错退出，不去结束它；想并行就换端口（`AUK_PORT=8081`）。
- 服务进程被 Job Object 包住：关掉控制台、Ctrl+C、异常退出时，模型子进程一并回收，不留占显存的孤儿。
- 首次使用不必先加载模型，页面里点「加载模型」即可；换模型或换显存策略需要重新加载（约 20s）。

## 显存策略

默认**分页加载**：编码器逐层上卡，慢一些但 8GB 卡装得下。切成**全常驻**可省掉逐层搬运，代价是权重必须整个进显存——实测门槛 14GB 以上。显存不够时页面会先把状态标成警告；硬切过去会在加载时报「显存不足」并提示切回分页。

命令行定死默认值用 `AUK_RESIDENT=1`，但页面里的选择优先级更高。

## 目录结构

```text
auk-studio/
├─ run.py                      启动器：单实例、进程树回收、启动器日志
├─ 启动AuK-Studio.bat / 停止AuK-Studio.bat
├─ backend/
│  ├─ app.py                   路由 + ModelSupervisor + spawn 模型子进程
│  └─ logging_setup.py         日志骨架与请求编号
├─ frontend/index.html         单文件工作台，无构建步骤
├─ src/auk/                    上游模型代码（infer/、model/、train/、pe.config.yaml）
├─ comfyui/                    ComfyUI 自定义节点与工作流
├─ assets/demo-input-audio/    官方案例素材，后端挂在 /demo-audio
├─ ckpts/                      权重目录（内容不入库）
├─ uploads/ outputs/ logs/     运行时目录（不入库）
├─ docs/                       AuK-MODEL、COOKBOOK、COMFYUI、FINETUNING、CONTRIBUTING
└─ pyproject.toml  uv.lock  requirements-web.txt  .env.example
```

## 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `AUK_HOST` | `127.0.0.1` | 监听地址。工作台无鉴权且能删文件，别随意暴露到局域网 |
| `AUK_PORT` | `8080` | 端口；被非 AuK 进程占用时启动器会拒绝启动 |
| `AUK_OPEN_BROWSER` | `1` | 设 `0` 不自动开浏览器（等价 `--no-browser`） |
| `AUK_BASE_CKPT` / `AUK_BASE_CONFIG` | `./ckpts/AuK/…` | Base 权重与配置 |
| `AUK_FLASH_CKPT` / `AUK_FLASH_CONFIG` | `./ckpts/AuK-Flash/…` | Flash 权重与配置 |
| `AUK_RESIDENT` | `0` | `1` = 默认全常驻，页面选择仍优先 |
| `AUK_MODEL_TIMEOUT` | `0` | 加载/生成时限（秒）。`0` = 不限时，用页面的「取消等待」中止 |
| `AUK_LOG_DIR` | `./logs` | 日志目录；并行实例请分开设 |
| `AUK_LOG_LEVEL` | `INFO` | 日志级别 |
| `AUK_DEBUG` | 未设 | 设 `1` 时控制台只报 INFO，日志文件里保留 DEBUG 明细 |

`.env.example` 列的是**上游 Prompt Enhancer** 需要的 LLM / 云 ASR 密钥，本仓库代码不会自动加载 `.env`，需要时请自行导出到环境。

## 日志与故障定位

所有报错都落到 `logs/`（已 git 忽略，单文件满 10MB 自动滚动，最多留 5 份）：

| 文件 | 谁在写 | 里面有什么 |
| --- | --- | --- |
| `logs/launcher.log` | `run.py` | 端口占用、替换旧实例、uvicorn 异常退出 |
| `logs/server.log` | FastAPI 服务进程 | 每个请求一行（含耗时）、4xx 的用户可见提示、5xx 的堆栈与内存/显存快照、浏览器上报的错误 |
| `logs/worker.log` | 模型子进程 | 导入 torch → 加载权重 → 逐段推理 → 落盘，每个阶段的起止与显存 |

每个请求分配一个 8 位日志编号（响应头 `X-Auk-Rid`，5xx 提示里写作「日志编号 xxxxxxxx」），同一个编号会同时出现在 `server.log` 和 `worker.log`，`grep` 一次即可串起两个进程。模型进程被系统杀掉时（原生崩溃、提交内存耗尽），服务进程会把「当时在做什么 + 已等待多久 + 内存/显存读数 + 模型日志末尾 20 行」一并写进 `server.log`，所以一个文件就够定位。

```bash
AUK_DEBUG=1 python run.py               # 控制台安静，日志文件留 DEBUG 明细
AUK_LOG_DIR=D:\tmp\logs python run.py   # 第二个实例换目录，避免互相滚动
AUK_MODEL_TIMEOUT=600 python run.py     # 给模型操作加硬时限
```

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 服务与模型状态（页面启动探测） |
| GET | `/api/system` | CPU / 内存 / 显存读数 |
| POST | `/api/model/load` | `model=base\|flash`、`resident=0\|1` |
| POST | `/api/model/unload` | 卸载模型并回收显存 |
| POST | `/api/model/abort` | 取消当前加载/生成（页面的「取消等待」） |
| GET | `/api/model/status` | 加载/生成进度 |
| POST | `/api/plan` | `text`、`ref_name`、`ref_seconds`、`ref_range`、`emotion` → 分段预览 |
| POST | `/api/generate` | 见下方说明 |
| GET | `/api/audio/{job_id}` | 结果 WAV |
| GET | `/api/input/{name}` | 回读已上传的输入音频（A/B 对比、重跑） |
| POST | `/api/log/client` | 前端错误上报，写进 `server.log` |
| GET | `/demo-audio/*`、`/*` | 静态素材与页面 |

`POST /api/generate` 表单字段：`instruction`（必填）、`model`（`base`/`flash`，默认 `flash`）、`gen_seconds`、`emotion`、`seed`、`task`（页面声明官方任务名，用于后处理门控）、`text`、`input_name`、`ref_range`、`input_duration`、`audio`（文件）。`gen_seconds=0` 表示「与输入音频等长」，纯文本生成必须给正数。服务端把指令转成官方 message 格式后调用 `engine.generate(...)`，结果存为 WAV 并返回 `/api/audio/{job_id}`。

## 常见问题

- **改了后端没反应**：`run.py` 用 `reload=False`，重启才生效（再跑一次 bat 就会替换旧实例）。前端改动用 Ctrl+F5 强刷。
- **加载报 `WinError 1455`（页面文件太小）**：多半是残留的模型子进程还在占用提交内存。`python run.py --stop` 清场后重试。
- **想在局域网访问**：除了 `AUK_HOST=0.0.0.0`，还要改 `backend/app.py` 里的 CORS `allow_origins`。工作台没有鉴权且能删文件，请只在可信网络开启。
- **文件留存**：`uploads/` 超过 7 天自动清理（A/B 对比与重跑依赖它），`outputs/` 不自动删除，需要手动清空。
- **人声分离响度**：官方行为是「限到 −14 LUFS 再封顶 0.95 峰值」，环境里没装 `pyloudnorm` 时退化为只封顶峰值；装一次 `pip install pyloudnorm` 即可解锁完整行为。

## 关于上游 AuK 模型

- [docs/AuK-MODEL.md](docs/AuK-MODEL.md) — 模型介绍、性能基准、架构、支持任务、权重下载、CLI / Gradio / ComfyUI / Python API 用法、微调与引用条目。
- [docs/COOKBOOK.md](docs/COOKBOOK.md) — 各任务的官方指令模板与示例。
- [docs/COMFYUI.md](docs/COMFYUI.md) · [docs/FINETUNING.md](docs/FINETUNING.md) · [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- 许可：代码与模型权重由上游以 **MIT License** 发布，详见 [`LICENSE`](LICENSE)；`auk-studio` 这层 Web 工作台沿用同一许可。

## 后续可扩展

1. 真正的任务队列（避免多个 GPU 推理同时抢显存）
2. WebSocket/SSE 实时任务进度
3. 历史记录数据库
4. 多模型/多 GPU worker
5. 音频波形解析与可视化
6. 用户登录与权限
7. Prompt Enhancer 接入
