# 端侧 LLM 推理与量化工程资源库

收集端侧大模型推理加速、量化及软硬件协同相关的论文、博客和技术文档。重点关注 Jetson / Orin、高通手机、台式机、独显笔记本和小型 AI 工作站。

**[浏览资源索引](INDEX.md)** · [表格索引](catalog.csv) · [标签词表](tags.yaml)

## 添加一条资源

每条资源一个目录，`meta.yaml` 保存网址和标签；开始阅读后再创建 `notes.md`。以下以新增一篇论文为例，命令在仓库根目录运行：

```powershell
New-Item -ItemType Directory -Path papers/my-paper
Copy-Item templates/meta.yaml papers/my-paper/meta.yaml
```

编辑新建的 `meta.yaml`，填写标题、网址、收藏理由和标签。将 `id` 改成目录名 `my-paper`；`papers/` 的 `type` 为 `paper`，`blogs/` 为 `blog`，`docs/` 为 `doc`。条目 ID 在整个仓库中唯一，使用小写英文、数字和连字符。

开始阅读时复制笔记模板，按资源内容删改章节：

```powershell
Copy-Item templates/notes.md papers/my-paper/notes.md
```

将 `status` 改成 `reading`，读完改成 `read`。索引自动链接已存在的 `notes.md`；没有笔记时显示 `—`。

## 生成索引和搜索

使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 环境和依赖。安装 uv 后，在仓库根目录运行：

```powershell
uv sync --locked
```

uv 会在本机准备 `.venv`，无需手动激活。项目支持 Python 3.10 及以上版本，`.python-version` 将默认开发环境固定在 Python 3.11。新增条目、修改元数据或增加笔记后运行：

```powershell
uv run python scripts/catalog.py build
```

这会先校验所有条目，再生成 `INDEX.md` 和 `catalog.csv`。**人工只维护元数据和笔记，不直接编辑生成的索引。** `INDEX.md` 包含分类表和按标签的交叉索引；CSV 使用 UTF-8 BOM，方便 Excel 导入，标签以分号分隔，链接和附件字段使用 JSON。

以下命令在仓库根目录运行。`uv run` 会自动同步项目环境，不需要指定 Windows 或 Linux 的虚拟环境路径。

```powershell
# 只检查格式、分类、ID、标签、网址格式和附件路径，不写文件
uv run python scripts/catalog.py check

# 同时包含“量化”和“Jetson Orin”的条目；支持中文名称和英文别名
uv run python scripts/catalog.py search --tag 量化 --tag orin

# 查找已读的 KV cache 论文
uv run python scripts/catalog.py search --tag kv-cache --type paper --status read

# 搜索元数据关键字，不搜索笔记正文
uv run python scripts/catalog.py search --query AWQ
```

多个 `--tag` 以及类型、状态、关键字之间均取交集。索引脚本本身离线运行，不验证链接是否可访问、不下载附件，也不要求本机存在附件。uv 首次准备环境或安装依赖时需要网络。

### 依赖与跨设备迁移

`pyproject.toml` 声明依赖，`uv.lock` 锁定解析结果，两者与 `.python-version` 一起上传 Git；各设备的 `.venv/` 不上传。在新设备克隆仓库后运行 `uv sync --locked`，即可按锁文件准备环境。

增加依赖用 `uv add 包名`，升级已锁定依赖用 `uv lock --upgrade-package 包名`，随后运行 `uv sync --locked` 和测试。无需另行维护 `requirements.txt`。

## 目录与元数据约定

```text
papers/<id>/meta.yaml       论文、综述、学术技术报告（不限 PDF/HTML）
blogs/<id>/meta.yaml        博客、技术解析、工程实践文章
docs/<id>/meta.yaml         官方文档、API 文档、部署指南、硬件手册
<分类>/<id>/notes.md        可选的阅读笔记
<分类>/<id>/resource/       本地下载资料，不上传 Git
templates/                 元数据和笔记模板
scripts/catalog.py         校验、索引生成与搜索
tests/                     脚本测试
pyproject.toml             项目与依赖声明
uv.lock                    跨设备依赖锁文件
.python-version            默认 Python 版本
```

目录只按资源类型区分。量化、异构、具体硬件等通过标签表达，不再创建主题子目录。每条记录直接位于分类目录下一层。同一篇论文的 PDF、网页和代码链接放在一个条目中；独立解读它的博客可另建条目，在 `links` 中关联。

| 字段 | 用途 |
| --- | --- |
| `id`、`title`、`type`、`url` | 必填的身份、标题、资源类型和主要来源 |
| `status` | 必填：`unread` / `reading` / `read`，仅表示阅读进度 |
| `summary` | 必填字符串，可为空；建议写一句收藏理由 |
| `tags` | 必填列表，可为空；使用词表里的规范键 |
| `links` | 可选，名称到网址的映射，如 `code`、`project`、`related-paper` |
| `attachments` | 可选，下载来源 `url` 和条目内的相对 `path` |
| `authors`、`year`、`venue`、`doi` | 可选的论文信息；不适用时省略 |

其他扩展字段会保留在元数据中，可被关键字搜索，但不会自动增加 CSV 列。

## 标签：对象、方法、硬件和实现证据分开

在 `tags.yaml` 中维护规范标签、中文名称和搜索别名。新标签先加入词表，再用于条目；脚本会拒绝未知标签和重复标签。

- `topic/*`：量化、推理、内存、编译、评测。
- `target/*`、`scheme/*`、`granularity/*`、`precision/*`：量化对象、方案、粒度和精度。
- `method/*`：AWQ、GPTQ、SmoothQuant 等方法。
- `system/*`、`stage/*`：异构、卸载、融合及 prefill / decode。
- `memory/*`、`hardware/*`、`backend/*`：内存架构、设备及计算后端。
- `evidence/*`：来源中提供的模拟量化、压缩权重、量化内核、端到端实现证据。

只标注来源实际涉及的内容。AWQ 的“激活感知”不等于激活被量化；`precision/w4a16` 也不等于所有算子都使用该精度。提供 kernel 不等于端到端加速。`evidence/*` 描述来源提供的实现，不表示本人复现成功；本人实测写入笔记，附测试条件。

## 下载 PDF 或网页快照

在对应条目中创建 `resource/` 并保存资料；`.gitignore` 的 `**/resource/` 规则会忽略所有这类目录。例如在元数据中记录：

```yaml
attachments:
  - url: https://example.org/paper.pdf
    path: resource/paper.pdf
```

附件路径使用正斜杠，必须位于该条目的 `resource/` 中，不能使用本机绝对路径。记录附件来源不要求文件已下载，也不会改变阅读状态。

**跨设备克隆会恢复元数据、笔记和索引，不会恢复本地附件。** 新设备可按 `attachments` 中的来源重新下载；无法重新下载的资料应单独备份。不要用 `git add -f` 强制提交下载文件。忽略规则不会自动取消已经被 Git 跟踪的文件。

## 检查脚本

```powershell
uv run python -m unittest discover -s tests -v
```
