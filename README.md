# awesome-on-device-LLM

端侧 LLM 推理与量化工程资源库，收集端侧大模型推理加速、量化及软硬件协同相关的论文、博客和技术文档。

关注的设备：Jetson / Orin、高通手机、台式机、独显笔记本和小型 AI 工作站。

## 文档入口

| 想做什么 | 去哪里 |
| --- | --- |
| 浏览全部资源，按类型或标签查阅 | [INDEX.md](INDEX.md) |
| 用 Excel 筛选、排序 | [catalog.csv](catalog.csv) |
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

## 添加一条资源

以新增论文 `my-paper` 为例。

1. 创建条目目录，复制元数据模板：

   ```powershell
   New-Item -ItemType Directory -Path papers/my-paper
   Copy-Item templates/meta.yaml papers/my-paper/meta.yaml
   ```

2. 编辑 `meta.yaml`：填写标题、网址、收藏理由和标签，并把 `id` 改成目录名 `my-paper`。
3. 开始阅读时复制笔记模板，按内容删改章节，并将 `status` 改为 `reading`：

   ```powershell
   Copy-Item templates/notes.md papers/my-paper/notes.md
   ```

4. 读完后将 `status` 改为 `read`。
5. 重新生成索引：

   ```powershell
   uv run python scripts/catalog.py build
   ```

命名规则：

- 条目 ID 在整个仓库中唯一，只用小写英文、数字和连字符，且必须与目录名一致。
- 条目放在对应类型的目录下，`type` 随目录而定：

  | 目录 | `type` | 收录内容 |
  | --- | --- | --- |
  | `papers/` | `paper` | 论文、综述、学术技术报告（不限 PDF/HTML） |
  | `blogs/` | `blog` | 博客、技术解析、工程实践文章 |
  | `docs/` | `doc` | 官方文档、API 文档、部署指南、硬件手册 |

- 同一篇论文的 PDF、网页、代码链接放进同一个条目。独立解读它的博客另建条目，并在 `links` 中互相关联。

## 生成索引

```powershell
uv run python scripts/catalog.py build
```

`build` 先校验所有条目，再生成 `INDEX.md`（分类表和标签交叉索引）与 `catalog.csv`。

> **只维护 `meta.yaml` 和 `notes.md`，不要直接编辑 `INDEX.md` 和 `catalog.csv`。** 它们每次都会被覆盖。

- 笔记：存在 `notes.md` 时，索引自动生成链接；不存在则显示 `—`。
- 离线：脚本不检查链接是否可访问，不下载附件，也不要求附件已存在。
- 公共索引与 CSV 不包含个人 Zotero 链接，也不读取本机绑定或下载状态。登录自己的 Zotero 账号并完成 [联动流程](ZOTERO.md) 后，运行 `uv run --locked python scripts/zotero_sync.py index` 生成 `.local/zotero/index.md`，在本机打开它进入自己的条目或 PDF。该文件不入 Git，不要复制作者的关联。

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
| `<分类>/<id>/meta.yaml` | 元数据（必需） | 是 |
| `<分类>/<id>/notes.md` | 阅读笔记（可选） | 是 |
| `<分类>/<id>/resources/` | 本地资料：PDF、网页快照、笔记引用的原图 | 否 |
| `<分类>/<id>/resources/asset-sources.yaml` | 原图的来源、页码和裁剪信息 | 是 |

分类只按资源类型划分。量化、异构、具体硬件等一律用标签表达，不建主题子目录。

`meta.yaml` 字段：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `id`、`title`、`type`、`url` | 必填 | 身份、标题、资源类型、主要来源 |
| `status` | 必填 | `unread` / `reading` / `read`，只表示阅读进度 |
| `summary` | 必填，可为空 | 收藏理由，建议一句话 |
| `tags` | 必填，可为空 | 使用 `tags.yaml` 中的规范键 |
| `links` | 可选 | 名称到网址的映射，如 `code`、`project`、`related-paper` |
| `attachments` | 可选 | 下载来源 `url` 和相对路径 `path` |
| `authors`、`year`、`venue`、`doi` | 可选 | 论文信息，不适用时省略 |

其他公开的自定义字段会保留，可被关键字搜索，但不会增加 CSV 列。**个人 Zotero 状态不是公开扩展字段**：禁止顶层 `zotero` 以及附件中的 `zotero_item_key`、`zotero_attachment_key`、`local_status`，即使值为空也会校验失败。账号配置、条目/附件关联及可用状态保存在 `.local/zotero/`；个人下载时间 `downloaded_on` 不写进共享元数据。历史 `source_zotero_*` 仅追溯来源，不作为当前绑定，来源和版本事实仍可保留。

## 标签

标签格式为 `前缀/名称`。规范键、中文名称和搜索别名都在 [tags.yaml](tags.yaml) 中维护。新标签必须先加入词表再使用，脚本会拒绝未知标签和重复标签。

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
| `publication/` | 发表状态 | `publication/withdrawn`（已撤稿） |

## 本地资料与笔记配图

每个条目的 `resources/` 目录用来存放本地资料。整个目录被 Git 忽略，只有 `asset-sources.yaml` 例外。

```text
<分类>/<id>/resources/
├── paper.pdf               # 下载的 PDF 或网页快照（不入 Git）
├── assets/                 # 笔记引用的原图（不入 Git）
│   └── figure-05-xxx.png
└── asset-sources.yaml      # 原图来源（入 Git）
```

### 下载 PDF 或网页快照

在 `meta.yaml` 中记录来源，再把文件保存到 `resources/`：

```yaml
attachments:
  - url: https://example.org/paper.pdf
    path: resources/paper.pdf
```

- `path` 必须以 `resources/` 开头，使用正斜杠，不能是本机绝对路径。
- 只记录来源即可，不要求文件已下载，也不改变阅读状态。

### 在笔记中引用原图

- 原图放在 `resources/assets/`，笔记用相对路径引用，例如 `![说明](resources/assets/figure-05-xxx.png)`。
- 图注中写明论文版本、图表编号和 PDF 页码。
- 在 `resources/asset-sources.yaml` 记录原 PDF、下载网址、页码、裁剪坐标和单位，方便在新设备上重新生成。
- 笔记中的关键结论仍要用文字写出，缺少本地图片时也能读懂。

### 换设备时

克隆仓库会恢复元数据、笔记、索引和 `asset-sources.yaml`，**不会**恢复 PDF 和原图。需要时按 `attachments` 的来源重新下载，原图按 `asset-sources.yaml` 重新裁剪。无法重新获取的资料请自行备份。

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
