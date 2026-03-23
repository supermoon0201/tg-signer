# 默认 Docker 镜像瘦身设计

日期：2026-03-23

## 背景

当前根目录 [`docker-compose.yml`](c:/Dev/Code/py-workspace/tg-signer/docker-compose.yml) 默认构建 [`docker/Run.Dockerfile`](c:/Dev/Code/py-workspace/tg-signer/docker/Run.Dockerfile)，并以源码运行方式启动 CLI 任务：

- 默认命令是 `run nebula -n 1000`
- 默认路径并不需要 WebUI
- 当前镜像却无条件安装了 `".[speedup,gui]"`
- 当前项目基础依赖仍包含完整的 `opencv-python`
- 当前运行镜像还安装了 `ffmpeg` 与一组图形相关系统库

这会让默认 `docker compose up -d` 产出的运行镜像承担了不属于默认场景的依赖成本，导致镜像体积过大。

## 目标

- 让默认 `docker compose up -d` 构建出的镜像只服务于 CLI/任务执行场景
- 从默认镜像中移除 WebUI 依赖
- 在不破坏 GIF/视频验证码识别链路的前提下，收缩 OpenCV 相关依赖
- 保持当前“源码运行”的 Docker 使用方式，不切换到 PyInstaller 产物镜像
- 更新文档，使默认镜像的职责边界明确

## 非目标

- 不改造默认 compose 为 PyInstaller 多阶段镜像
- 不删除项目中的 `webgui` 命令或 [`tg_signer/webui`](c:/Dev/Code/py-workspace/tg-signer/tg_signer/webui) 模块
- 不在这次工作中设计新的 WebUI 专用 compose 文件或镜像产品线
- 不为了追求极限瘦身而无验证地删除 `ffmpeg` 或视频解码所需运行库

## 当前问题归因

默认镜像体积偏大的主要原因不是构建上下文，而是运行时依赖集合过宽：

1. [`docker/Run.Dockerfile`](c:/Dev/Code/py-workspace/tg-signer/docker/Run.Dockerfile) 安装了 `".[speedup,gui]"`，会把 `nicegui` 及其依赖装入默认镜像。
2. [`pyproject.toml`](c:/Dev/Code/py-workspace/tg-signer/pyproject.toml) 把 `opencv-python` 放在基础依赖中，导致默认安装路径拉入完整 OpenCV 包。
3. 运行镜像额外安装了 `ffmpeg` 与图形库，但当前代码只看到 [`tg_signer/ai_tools.py`](c:/Dev/Code/py-workspace/tg-signer/tg_signer/ai_tools.py) 使用 `cv2.VideoCapture` 与 `cv2.cvtColor` 处理 GIF/视频帧，没有 GUI 窗口类调用。

## 设计概览

### 1. 默认镜像职责收窄

默认 compose 的产物定义为“轻量 CLI 运行镜像”：

- 继续使用源码运行模式
- 继续支持 `run`、签到、消息处理、AI 识图、GIF/视频验证码识别
- 不默认包含 WebUI 依赖

这意味着默认镜像只覆盖当前 compose 的主路径，不再为未启用的 `webgui` 能力付出体积成本。

### 2. Python 依赖收缩

默认方向是将项目基础依赖中的完整 OpenCV 切换为 headless 版本，并保持 WebUI 依赖继续作为可选 extra：

- 将基础依赖从 `opencv-python` 调整为 `opencv-python-headless`
- 保留 `gui = ["nicegui"]` 作为显式可选能力
- 默认 Docker 安装路径从 `".[speedup,gui]"` 改为 `".[speedup]"`

选择 `opencv-python-headless` 的依据：

- 当前代码没有使用 OpenCV 的窗口显示、键盘交互或桌面 GUI API
- 容器默认场景是无显示环境
- headless 包通常能显著降低镜像体积，并避免不必要的图形依赖

如果测试结果表明视频解码链路仍需要个别系统库，则只保留最小必需集合，不恢复整套 GUI 依赖。

### 3. Docker 运行时依赖收缩

[`docker/Run.Dockerfile`](c:/Dev/Code/py-workspace/tg-signer/docker/Run.Dockerfile) 将按“最小可运行”原则调整：

- 移除默认安装中的 `gui` extra
- 按 headless OpenCV 的实际需要减少图形相关系统包
- `ffmpeg` 不先验删除，只有在验证确认不需要后才移除
- 保留时区、证书与 Python 运行必需项

实现阶段的目标不是预设一个最小列表，而是通过测试反推出最小可接受列表。

### 4. 文档与默认行为对齐

文档将明确以下约定：

- 默认 `docker compose up -d` 是轻量 CLI 镜像
- 默认镜像不包含 WebUI 依赖
- 若未来需要 WebUI，应走显式的独立依赖路径，而不是复用默认镜像

需要同步更新的文档至少包括：

- [`README.md`](c:/Dev/Code/py-workspace/tg-signer/README.md)
- [`docker/README.md`](c:/Dev/Code/py-workspace/tg-signer/docker/README.md)

## 兼容性约束

本次收缩必须保持以下行为不变：

- 默认 compose 仍能运行现有 CLI 任务
- `speedup` 相关加速能力保持可用
- AI 识图能力保持可用
- GIF 验证码识别中的 PIL 路径保持可用
- GIF/视频回退到 OpenCV 解码的路径保持可用

WebUI 不属于默认镜像兼容目标，但项目级能力仍需保留，不能在代码层移除 `webgui` 命令。

## 测试策略

实现将先补测试，再改实现，重点覆盖依赖边界而不是只做静态文件修改。

计划补充的测试重点：

1. `recognize_gif_code` 在 PIL 可正常打开图片时仍能完成帧提取与后续处理。
2. `recognize_gif_code` 在 PIL 无法识别时，能够回退到 `cv2.VideoCapture` 路径。
3. OpenCV 回退路径只依赖当前代码实际使用的 API，不依赖 GUI 功能。
4. `webgui` 仍作为可选能力保留，不应被默认 Docker 路径误装。

测试通过后再修改：

- [`pyproject.toml`](c:/Dev/Code/py-workspace/tg-signer/pyproject.toml)
- [`docker/Run.Dockerfile`](c:/Dev/Code/py-workspace/tg-signer/docker/Run.Dockerfile)
- 相关文档

## 风险与处置

### 风险 1：headless OpenCV 与视频解码链路存在差异

处置：

- 先用测试锁定 `cv2.VideoCapture` 回退路径的行为
- 不假设 `ffmpeg` 可删
- 若运行时仍需少量视频/编解码系统库，仅保留已验证的最小集合

### 风险 2：项目级依赖调整影响非 Docker 用户

处置：

- 基于代码使用面确认未使用 OpenCV GUI API
- 若测试或实际验证显示需要保留差异化安装路径，再在计划阶段评估是否拆分为额外依赖而不是直接全局替换

### 风险 3：文档与默认行为继续不一致

处置：

- 把默认 compose 的职责写进 README 与 Docker README
- 在 Dockerfile 注释中明确默认镜像不包含 WebUI

## 预期改动范围

高概率涉及的文件：

- [`pyproject.toml`](c:/Dev/Code/py-workspace/tg-signer/pyproject.toml)
- [`docker/Run.Dockerfile`](c:/Dev/Code/py-workspace/tg-signer/docker/Run.Dockerfile)
- [`README.md`](c:/Dev/Code/py-workspace/tg-signer/README.md)
- [`docker/README.md`](c:/Dev/Code/py-workspace/tg-signer/docker/README.md)
- 与 `ai_tools` 相关的测试文件

## 验收标准

- 默认 compose 构建链路不再安装 `nicegui`
- 默认 Docker 安装路径与“无 WebUI 的 CLI 镜像”职责一致
- OpenCV 依赖从完整桌面版收缩到适合容器场景的最小方案，且 GIF/视频验证码识别保持可用
- 文档明确说明默认镜像不包含 WebUI
- 所有新增与受影响测试通过
