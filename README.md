# awesome-on-device-LLM

端侧 LLM 推理与量化工程资源库，收集端侧大模型推理加速、量化及软硬件协同相关的论文、博客和技术文档。

关注的设备：Jetson / Orin、高通手机、台式机、独显笔记本和小型 AI 工作站。

## 文档入口

| 想做什么 | 去哪里 |
| --- | --- |
| 浏览全部资源，按类型或标签查阅 | [INDEX.md](INDEX.md) |
| 在浏览器中筛选、编辑、按网址收录和阅读笔记 | [本地前端](#本地前端) |
| 用 Excel 筛选、排序 | [catalog.csv](catalog.csv) |
| 用 Obsidian 浏览、筛选和编辑资源属性 | [资源库.base](资源库.base)，使用方法见下文 |
| 用自己的 Zotero 阅读论文与按标签筛选 | [Zotero 联动](ZOTERO.md)，本机入口为 `.local/zotero/index.md` |
| 查看或新增标签 | [tags.yaml](tags.yaml) |
| 配置环境、迁移到新设备、运行测试 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 新建条目或笔记 | [templates/](templates/) |
| 让 agent 代为收录和阅读 | [resource-curator skill](skills/resource-curator/SKILL.md) |

## 快速开始

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在仓库根目录运行（以下命令均在仓库根目录执行）：

```powershell
uv sync --locked                           # 准备 Python 环境，首次需要联网
uv run python scripts/catalog.py search --tag 量化 --tag orin   # 试一下搜索
```

## 本地前端

```powershell
uv run --locked python scripts/web.py --open
```

默认打开 **http://127.0.0.1:8765**，终端按 `Ctrl+C` 停止。端口被占用时加 `--port 8766`。无需 Node.js、前端构建、数据库或云账号；准备好 Python 依赖后，浏览和编辑可离线使用。

- **检索**：侧栏可按中文名、规范键或别名找到标签；多选标签取交集，可叠加类型、阅读状态和元数据关键词。关键词不搜索笔记正文。
- **编辑**：点击资源标题，再点“编辑元数据”。常用字段和可选的“代码仓库”网址直接填写；其他链接、附件和自定义公开字段在“高级字段”中以 JSON 编辑。ID 和类型固定，不在网页中重命名或移动目录。
- **保存**：只修改表单中变动的字段，保留未改动的自定义字段及其 YAML 类型、卡片正文、独立笔记和附件路径。保存前复用目录校验，成功后自动更新 `INDEX.md`、`catalog.csv`。YAML 注释和排版可能重新整理；没有第二份可写元数据或浏览器数据库。
- **外部修改**：Obsidian 或其他编辑器改过文件后，点“刷新仓库”更新列表，点详情中的“重新读取文件”更新当前资源。过时表单保存会报告冲突并保留输入；先复制修改，再重新读取并合并，不会直接覆盖新版本。
- **网址收录**：点“收录资源”，粘贴公开 HTTP(S) 网页地址。先预览来源声明的标题、作者等信息，并跨分类按 DOI、arXiv、来源网址和标题查重；确认 ID、类型、中文收藏理由及标签后才写入。默认 `unread`，不会自动标记已读、猜测标签或把外文摘要写成中文收藏理由。
- **阅读笔记**：详情页“研究笔记”渲染已有 `notes.md`，支持表格、代码块、行内/块级公式、Mermaid 图和 `resources/` 下的本地配图。笔记只读，仍在 Markdown 文件中维护；打开笔记不会改变阅读状态。

公式使用本地 KaTeX 的 HTML/CSS 排版，并附带供辅助技术使用的 MathML；不依赖浏览器原生 MathML 完成视觉排版，也不访问 CDN。行内支持 `$x^2$` 和 `\(x^2\)`，块级支持 `$$...$$` 和 `\[...\]`，块内可换行。反引号、代码块和转义的分隔符不参与公式渲染；不支持的公式会保留 LaTeX 源码并提示错误。

网址预览只解析返回的 HTML 元数据，不执行网页 JavaScript，不登录网站，不下载 PDF，也不核实全文、撤稿状态或最新版本。受限页面、纯 PDF 或无法读取的来源可转入明确标注的手动填写流程；未知作者、年份等应留空，不能推测。新收录不创建空笔记。

网址预览需要联网；预览本身不写仓库，点击确认后才创建资源卡并更新索引。遇到重复论文应打开已有条目，不通过换 ID 绕过去重。Web 工作台不会下载附件或同步 Zotero，附件字段只记录来源和预期路径；Zotero 联动需单独执行[导入与核验流程](ZOTERO.md#准备导入与核验)。

原始 HTML 不执行，远程图片不自动加载，SVG 附件仅下载；缺失附件显示错误，不影响元数据保存。新标签仍需先维护 `tags.yaml`，前端不绕过词表。

服务只监听本机，不是多人协作或公网部署服务。不要通过反向代理或端口转发对外开放；同一资源库同时只运行一个写入服务。

## 添加一条资源

收录不要求已经阅读，也不要求已下载 PDF 或已同步 Zotero。下面三步彼此独立。

### 1. 仅收录到仓库

可使用上面的 Web 工作台预览并确认收录，也可手动创建。手动添加前，先按 DOI、arXiv ID、网址和标题在全库查重；已有论文应补充原条目，不另建版本副本。以新增论文 `my-paper` 为例：

1. 创建条目目录，复制资源卡模板：

   ```powershell
   New-Item -ItemType Directory -Path papers/my-paper
   Copy-Item templates/resource.md papers/my-paper/resource.md
   ```

2. 编辑 `resource.md` 开头的 YAML frontmatter：核实标题、来源网址、版本及可选作者和年份，填写中文收藏理由和规范标签，并把 `id` 改成目录名 `my-paper`。新条目默认 `status: unread`；卡片正文可选，不作为索引元数据。
3. 如需保存原文，按[附件说明](#下载-pdf-或网页快照)记录来源和路径，将核对过版本的文件保存到该位置。只有附件记录、不含本地文件时，应保留这一缺口，不宣称原文已保存。
4. 手动建卡或在 Obsidian 中修改后，重新生成索引；Web 确认收录已自动完成此步骤：

   ```powershell
   uv run --locked python scripts/catalog.py build
   ```

校验通过、索引出现新条目，即完成收录。此时不必创建 `notes.md`。

命名规则：

- 条目 ID 在整个仓库中唯一，只用小写英文、数字和连字符，且必须与目录名一致。
- 条目放在对应类型的目录下，`type` 随目录而定：

  | 目录 | `type` | 收录内容 |
  | --- | --- | --- |
  | `papers/` | `paper` | 论文、综述、学术技术报告（不限 PDF/HTML） |
  | `blogs/` | `blog` | 博客、技术解析、工程实践文章 |
  | `docs/` | `doc` | 官方文档、API 文档、部署指南、硬件手册 |

- 同一篇论文的 PDF、网页、代码链接放进同一个条目。独立解读它的博客另建条目，并在 `links` 中互相关联。

### 2. 可选：联动自己的 Zotero

1. 首次使用时打开并登录 Zotero，配置只读 API、已认证 MCP，并运行 `zotero_sync.py configure`，详见 [Zotero 配置](ZOTERO.md#先连接自己的-zotero)。
2. 运行 `uv run --locked python scripts/zotero_sync.py prepare`，检查计划。**当前处理全库论文，没有单篇参数；`--download` 也可能下载其他条目的缺失 PDF。** 仅想下载新论文时，先单独保存文件，不用全库下载命令代替。
3. 由已连接认证 MCP 的 agent 核对账号、复用或创建顶层集合 `awesome-on-device-LLM`，再执行 `.local/zotero/import.js`，保存实际回执。
4. 核验回执并生成个人阅读入口：

   ```powershell
   uv run --locked python scripts/zotero_sync.py record .local/zotero/receipt.json
   uv run --locked python scripts/zotero_sync.py index
   ```

确认论文身份、附件版本和 PDF 实际可读；只生成计划不算同步完成。账号配置、下载授权和回执格式以 [ZOTERO.md](ZOTERO.md) 为准。Zotero 中的元数据、批注、笔记和阅读状态不反向同步到仓库。

### 3. 开始阅读后再写笔记

将资源卡的 `status` 改为 `reading`，复制独立笔记模板：

```powershell
Copy-Item templates/notes.md papers/my-paper/notes.md
```

在卡片正文添加相对链接 `[阅读笔记](notes.md)`；没有笔记时不添加该链接。笔记开头记录来源版本和阅读范围，正文按需删改模板。用户读完后再改为 `read`，agent 生成分析不自动代表用户已读。修改状态或笔记后运行 `uv run --locked python scripts/catalog.py build`。

## 生成索引

```powershell
uv run python scripts/catalog.py build
```

`build` 先校验所有条目，再生成 `INDEX.md`（分类表和标签交叉索引）与 `catalog.csv`。

> **只维护 `resource.md` 的 frontmatter、可选卡片正文和独立的 `notes.md`，不要直接编辑 `INDEX.md` 和 `catalog.csv`。** 生成文件每次都会被覆盖。

- 笔记：存在 `notes.md` 时，索引自动生成链接；不存在则显示 `—`。
- 离线：脚本不检查链接是否可访问，不下载附件，也不要求附件已存在。
- 公共索引与 CSV 不包含个人 Zotero 链接，也不读取本机绑定或下载状态。登录自己的 Zotero 账号并完成 [联动流程](ZOTERO.md) 后，运行 `uv run --locked python scripts/zotero_sync.py index` 生成 `.local/zotero/index.md`，在本机打开它进入自己的条目或 PDF。该文件不入 Git，不要复制作者的关联。

## 用 Obsidian 管理资源

1. 在 Obsidian 中将仓库根目录打开为 vault，启用核心插件 **Bases**。
2. 打开根目录 [资源库.base](资源库.base)，使用 **全部资源 / 阅读队列 / 主题专题 / 设备专题 / 稀疏注意力** 视图浏览、筛选资源。
3. 在 Bases 中编辑 `status`、`summary`、`tags` 等简单属性，编辑后点击其他单元格或工具栏完成保存；通过“打开卡片”列进入资源卡，已有笔记可通过正文中的 `[阅读笔记](notes.md)` 打开。新建笔记后再给卡片添加这个相对链接。

   笔记编辑视图中可用 `Ctrl+单击` 跟随链接；在阅读视图中直接单击。

Bases 只纳入 `papers/`、`blogs/`、`docs/` 下的 `resource.md`，不会把 `notes.md`、模板或附件当作资源。元数据只保存于资源卡开头的 YAML frontmatter；正文可选，不参与目录脚本的元数据搜索与索引。`links`、`attachments` 等嵌套 YAML 在源码模式中编辑，避免把结构改成普通字符串。

Obsidian 的编辑界面不替代仓库校验：标签仍须使用 `tags.yaml` 的规范键，状态仍只能是 `unread` / `reading` / `read`。修改后运行 `check` 或 `build` 检查，GitHub 的 `INDEX.md` 与 CSV 必须运行 `build` 才会更新。公共卡片不保存个人 Zotero 绑定；个人入口仍用 `.local/zotero/index.md`。

`.obsidian/` 是各设备自己的设置目录，被 Git 忽略；根目录 `.base` 文件随仓库共享。“主题专题”展示带 `topic/*` 标签的资源，“设备专题”展示带 `hardware/*` 标签的资源；可在视图的“筛选”中按具体规范标签继续收窄。公式列只用于展示，修改标签应使用“标签”列。

视图的筛选、排序和列设置会保存到共享的 `资源库.base`，并可能随 Git 影响其他设备。不要仅凭“全部资源”的名称判断结果是否完整；若曾添加 `topic/inference` 等筛选，要恢复总览，应清除该视图自己的筛选条件，保留 Base 顶层限定资源卡目录的条件。临时专题建议复制视图并改名，避免改窄公共总览。

“稀疏注意力”按现有元数据聚合：须带 `system/sparse-inference` 标签，且标题或摘要包含 `attention`、`注意力` 或独立的 `KV`（英文不区分大小写）。该规则用于筛选候选，不等同于新增了专用标签；仅有权重或神经元稀疏化标签的条目不会因此自动纳入。

已有条目已从独立元数据文件迁移为 `resource.md` frontmatter，保留原有字段、ID、阅读状态、标签、自定义字段、附件相对路径和独立笔记；附件不移动。迁移后没有旧格式读取回退，也不维护第二份 YAML 元数据源。

## 搜索

```powershell
# 只校验，不写文件
uv run python scripts/catalog.py check

# 同时带有“量化”和“Jetson Orin”标签（中文名和英文别名均可）
uv run python scripts/catalog.py search --tag 量化 --tag orin

# 已读的 KV cache 论文
uv run python scripts/catalog.py search --tag kv-cache --type paper --status read

# 在元数据中搜索关键字（不搜索笔记正文）
uv run python scripts/catalog.py search --query AWQ
```

多个 `--tag` 以及类型、状态、关键字之间取交集。

## 条目结构与元数据

每个条目是分类目录下的一层子目录：

| 路径 | 内容 | 是否入 Git |
| --- | --- | --- |
| `<分类>/<id>/resource.md` | 资源卡，YAML frontmatter 为唯一元数据源（必需）；正文可选 | 是 |
| `<分类>/<id>/notes.md` | 阅读笔记（可选） | 是 |
| `<分类>/<id>/resources/` | 本地 PDF、网页快照（配图子目录除外） | 否 |
| `<分类>/<id>/resources/assets/` | 笔记引用的图片 | 是 |

分类只按资源类型划分。量化、异构、具体硬件等一律用标签表达，不建主题子目录。

`resource.md` 开头的 YAML frontmatter 字段（以 `---` 开始和结束）：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `id`、`title`、`type`、`url` | 必填 | 身份、标题、资源类型、主要来源 |
| `status` | 必填 | `unread` / `reading` / `read`，只表示阅读进度 |
| `summary` | 必填，可为空字符串 | 收藏理由，建议一句话；留空时写 `summary: ""` |
| `tags` | 必填，可为空列表 | 规范标签组成的字符串列表；无标签时写 `tags: []`，不填中文名或别名 |
| `links` | 可选 | 名称到网址的映射，如 `code`、`project`、`related-paper` |
| `links.code` | 可选，嵌套在 `links` 内 | 代码仓库的完整 HTTP(S) 网址，网页显示为“代码仓库”；没有或未核实时省略 |
| `attachments` | 可选 | 下载来源 `url` 和相对路径 `path` |
| `authors` | 可选 | 字符串列表，如 `["Alice", "Bob"]`；每个姓名须为非空字符串 |
| `year` | 可选 | `1` 至 `9999` 的整数，如 `2024`；不能写成字符串 `"2024"` 或布尔值 |
| `venue`、`doi` | 可选 | 非空字符串；信息未知或不适用时省略字段 |

YAML 中的 `summary:`、`tags:` 或显式的 `null` 都不是上述空字符串、空列表，会校验失败。可选字段的信息未知时直接省略，不用 `null` 占位。

代码仓库沿用 `links.code`，不另建 `code_url` 或中文顶层字段。网页收录和编辑表单提供独立输入框；清空它只移除代码链接，保留其他链接。“其他相关链接”编辑框不再填写 `code`，清空该框也不会删除代码仓库网址。Obsidian 中可用源码模式编辑：

```yaml
links:
  code: https://github.com/owner/project
```

上述网址仅为格式示例，实际填写前需核对仓库与资源的对应关系。CSV 继续将它保存在 `links` 列的 JSON 对象中。

其他公开的自定义字段可被关键字搜索，但不会增加 CSV 列。`metadata_path` 和 `notes_path` 是脚本加载时计算的**保留字段**，不能用作自定义字段：同名元数据值会被派生路径覆盖，不参与后续搜索或导出；原始 `resource.md` 不会被改写。`metadata_path` 指向资源卡，`notes_path` 仅在独立的 `notes.md` 存在时提供。外部笔记链接应放入 `links` 或其他非保留的自定义字段。

**个人 Zotero 状态不是公开扩展字段**：禁止顶层 `zotero` 以及附件中的 `zotero_item_key`、`zotero_attachment_key`、`local_status`，即使值为空也会校验失败。账号配置、条目/附件关联及可用状态保存在 `.local/zotero/`；个人下载时间 `downloaded_on` 不写进共享元数据。历史 `source_zotero_*` 仅追溯来源，不作为当前绑定，来源和版本事实仍可保留。

## 标签

标签格式为 `前缀/名称`。规范键、中文名称和搜索别名都在 [tags.yaml](tags.yaml) 中维护。新标签必须先加入词表再使用，脚本会拒绝未知标签和重复标签。

- 规范键的两段都由小写英文字母或数字组成，可用单个连字符连接，例如 `target/kv-cache`、`precision/w4a16`；不能使用大写、下划线、空格、中文、首尾连字符或连续连字符。`precision/INT4`、`target/kv_cache` 均不合法。
- 每个标签的 `label` 必须是非空字符串；`aliases` 可省略，填写时必须是非空字符串组成的列表，也可为空列表 `[]`。
- 不同标签的规范键、`label` 和 `aliases` 共用大小写不敏感的唯一命名空间。例如一个标签已有别名 `quantization`，另一个标签就不能使用 `QUANTIZATION` 作为 `label` 或 `aliases` 中的值；其他标签的规范键也不能用作 `label` 或别名。

**开源**使用规范标签 `availability/open-source`，命令行可用中文“开源”或别名 `open-source`、`opensource` 搜索。仅在确认相关代码已公开、许可证允许使用、修改和再分发后标注；只有 GitHub 链接、待发布仓库或无明确开源许可，不足以打标。填写代码仓库网址不会自动添加此标签，也不表示代码已运行或性能已复现。既有资源不会仅因含代码链接被批量标为开源。

代码核验记录保存在资源卡的 `code_review` 扩展字段：`checked_on` 是核验日期，`evidence` 是来源链接，`note` 说明对应关系、实现范围和缺口；核实许可后记录 `license`、`license_url`。代码地址仍只以 `links.code` 为准。

| `code_review.status` | 含义 | 是否添加“开源”标签 |
| --- | --- | --- |
| `open-source` | 已核实对应实现和适用开源许可；具体覆盖范围见 `note` | 是 |
| `source-available` | 找到公开实现，但开源许可依据不足或不满足条件 | 否 |
| `not-found` | 本次在所查来源中未找到可确认的实现 | 否 |
| `unverified` | 链接不可用、对应关系或实现范围等仍无法核实 | 否 |

这些状态不是顶层的阅读进度 `status`；“未找到”不等于证明代码不存在。历史版本、只公开部分组件、仅有许可证徽章等情况会单独注明，不能把基础框架或同名项目当作论文实现，也不能把许可核验当作运行复现。

| 前缀 | 含义 | 示例 |
| --- | --- | --- |
| `topic/` | 研究主题 | 量化、推理加速、内存优化、编译优化、性能评测 |
| `architecture/` | 模型架构 | `architecture/moe` |
| `target/` | 量化对象 | 权重、激活、KV 缓存 |
| `scheme/` | 量化方案 | 仅权重、权重激活联合、混合精度、PTQ、QAT |
| `granularity/` | 量化粒度 | 逐张量、逐通道、分组、逐 Token |
| `precision/` | 位宽组合 | W4A16、W8A8、W4A4、KV4 |
| `method/` | 具体方法 | AWQ、GPTQ、SmoothQuant、KIVI、KVQuant |
| `transform/` | 数值变换 | `transform/rotation` |
| `system/` | 系统技术 | 异构计算、卸载、算子融合、流水线、动态调度、近数据处理、专家缓存 |
| `stage/` | 推理阶段 | prefill、decode |
| `workload/` | 工作负载 | 在线服务、离线批量、长上下文 |
| `memory/` | 内存架构 | 统一内存、独立显存 |
| `hardware/` | 设备 | Jetson Orin、高通、NVIDIA 独显、Apple Silicon |
| `backend/` | 计算后端 | CPU、NPU、CUDA、Metal、Vulkan |
| `evidence/` | 来源提供的实现或评估证据 | 模拟量化、压缩权重、量化内核、端到端实现、体系结构仿真 |
| `availability/` | 代码可用性 | 开源（`availability/open-source`） |

## 本地资料与笔记配图

每个条目的 `resources/` 目录用来存放资料。PDF、网页快照被 Git 忽略，笔记配图放在 `assets/` 子目录并随 Git 同步；来源版本、图号和 PDF 页码直接写在笔记图注中。

```text
<分类>/<id>/resources/
├── paper.pdf               # 下载的 PDF 或网页快照（不入 Git）
└── assets/                 # 笔记引用的图片（入 Git）
    └── figure-05-xxx.png
```

### 下载 PDF 或网页快照

在 `resource.md` 的 YAML frontmatter 中记录来源，再把文件保存到 `resources/`：

```yaml
attachments:
  - url: https://example.org/paper.pdf
    path: resources/paper.pdf
```

- `path` 必须以 `resources/` 开头，使用正斜杠，不能是本机绝对路径。
- 只记录来源即可，不要求文件已下载，也不改变阅读状态。

### 在笔记中引用原图

- 原图放在 `resources/assets/`，笔记用相对路径引用，例如 `![说明](resources/assets/figure-05-xxx.png)`。
- 图注注明来源版本、图号和 PDF 页码；来源 PDF 或公开网址在条目元数据、笔记中保留，关键结论仍用文字说明。


### 换设备时

克隆仓库会恢复元数据、笔记、索引和笔记内图片，**不会恢复 PDF 或个人 Zotero 绑定**。元数据里存在附件路径、`check` / `build` 通过，都不代表原文文件存在。

- 有个人备份：将对应版本的 PDF 复制回资源卡 `attachments[].path` 指定的位置。
- Zotero 中有可读原件：从 Zotero 复制文件到该位置，不移动或删除 Zotero 存储目录中的原件。
- 两边都没有：从附件记录中的已核实来源重新下载，检查 PDF 内容与版本；无法取得全文时保留缺失说明，不用未知版本替代。

`zotero_sync.py prepare --download` 不是仓库 PDF 的全量恢复命令：它处理全库，且 Zotero 已有可读附件时会直接复用，不自动复制回仓库。详细条件见 [Zotero 准备流程](ZOTERO.md#1-准备本地计划)。

PDF 用个人备份或 Zotero 文件同步跨设备保存；需要仓库中的本地副本时仍须按上述路径复制。`.local/zotero/` 单独按账号管理，不能把他人的绑定当作自己的。`resources/assets/` 配图可正常提交，不要用 `git add -f` 提交 PDF。

## 用 agent 收录和阅读

[resource-curator skill](skills/resource-curator/SKILL.md) 指导 agent 收录、去重、打标签，并按需用 `templates/notes.md` 写笔记。调用示例：

```text
使用 $resource-curator 收录这篇论文：https://……，先不阅读。
使用 $resource-curator 收录这个博客，推荐标签给我审阅后再应用：https://……
使用 $resource-curator 阅读这篇论文，重点分析 Orin 上的量化实现，按默认模板写笔记：https://……
```

默认行为：

- 只收录，状态为 `unread`。
- 阅读分析不会自动把状态改成 `read`，阅读进度由你自己维护。
- 选择“审阅标签”时，agent 只给出建议，不会提前写入索引。

新设备上需要先安装 skill，安装方法见 [CONTRIBUTING.md](CONTRIBUTING.md#安装-agent-skill)。

## 笔记润色

笔记初稿可以用 [humanizer](https://github.com/op7418/Humanizer-zh) 润色，减少生硬的机器腔。
