# 维护指南

本文面向维护者：环境配置、依赖管理、跨设备迁移、测试和 skill 安装。日常添加资源请看 [README](README.md)。

以下命令均在仓库根目录运行。

## 环境

项目使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 环境和依赖。

```powershell
uv sync --locked
```

- uv 会在本机创建 `.venv`，无需手动激活；`uv run` 会自动使用它，不需要指定 Windows 或 Linux 的虚拟环境路径。
- 支持 Python 3.10 及以上，`.python-version` 把默认开发环境固定在 3.11。
- 首次准备环境或安装依赖时需要联网；索引脚本本身离线运行。

## 依赖与跨设备迁移

下面这些文件要上传 Git：

| 文件 | 作用 |
| --- | --- |
| `pyproject.toml` | 声明依赖 |
| `uv.lock` | 锁定解析结果 |
| `.python-version` | 固定默认 Python 版本 |

各设备的 `.venv/` 不上传。

**在新设备上**：克隆仓库后运行 `uv sync --locked`，即可按锁文件还原环境。

**增加依赖**：

```powershell
uv add 包名
```

**升级已锁定的依赖**：

```powershell
uv lock --upgrade-package 包名
uv sync --locked
uv run python -m unittest discover -s tests
```

无需另行维护 `requirements.txt`。

## 测试

```powershell
uv run python -m unittest discover -s tests
```


## 项目文件

| 路径 | 作用 |
| --- | --- |
| `papers/`、`blogs/`、`docs/` | 资源条目，按类型分类 |
| `templates/` | `meta.yaml` 和 `notes.md` 模板 |
| `tags.yaml` | 标签词表：规范键、中文名称、搜索别名 |
| `scripts/catalog.py` | 校验、索引生成与搜索 |
| `scripts/zotero_sync.py`、`scripts/zotero_import.js` | 生成带账号约束的 Zotero MCP 同步计划，核验回执并保存本地关联，见 [ZOTERO.md](ZOTERO.md) |
| `skills/resource-curator/` | agent 收录与阅读 skill 的源码 |
| `INDEX.md`、`catalog.csv` | 自动生成，不手工编辑 |
| `.local/zotero/config.json` | 本机 API 源地址与当前 Zotero 用户 ID，不入 Git，不含 token |
| `.local/zotero/bindings.json` | 按 `user:<id>` 隔离的个人条目/附件关联与可用状态，不入 Git |
| `.local/zotero/plan.json`、`import.js`、`receipt.json` | 本机待执行计划、MCP 脚本和实际回执，不入 Git |
| `.local/zotero/index.md` | 当前用户的个人 Zotero 阅读入口，不入 Git |
| `pyproject.toml`、`uv.lock`、`.python-version` | 环境与依赖 |

## 生成文件的格式

`catalog.csv` 的约定：

- 编码为 UTF-8 BOM，方便 Excel 直接打开。
- 标签以分号分隔。
- 链接和附件字段使用 JSON。
- 固定列依次为 `id,title,type,url,status,summary,tags,authors,year,venue,doi,links,attachments,metadata_path,notes_path`。
- 不再提供 `zotero_item_uri`、`zotero_pdf_uri` 列，也不保留空兼容列；使用这些旧列的下游脚本需迁移。公共 Markdown 同样没有 Zotero 列。
- 公共构建不读取 `.local/zotero/`，不受个人绑定、下载状态或本机 PDF 是否存在影响。

`INDEX.md` 顶部有自动生成的提示。需要修改内容时，改 `meta.yaml` 后重新运行 `build`。

公开元数据禁止顶层 `zotero`，附件禁止 `zotero_item_key`、`zotero_attachment_key`、`local_status`；校验时即使值为空也会拒绝。个人下载时间 `downloaded_on` 也不应写进共享元数据。可核实的 `downloaded_from`、`downloaded_version` 及历史 `source_zotero_*` 可作为来源事实保留，但不是当前账号绑定。

个人 Zotero 联动仅支持已登录账号的个人库：各用户独立运行 `configure`，再按 `prepare`（下载须显式 `--download`）、已认证 MCP、`record`、`index` 的顺序执行，完整命令见 [ZOTERO.md](ZOTERO.md)。空库必须配置自己的正整数 `--user-id`。不同账号不能共用绑定；同一账号换设备先完成 Zotero 数据/文件同步，再重新核验并生成个人索引。不要复制作者的绑定、旧计划或回执。

## 本地资料的 Git 规则

`.gitignore` 忽略每个条目 `resources/` 下的本地资料，只放行笔记配图目录 `assets/`：

```gitignore
**/resources/*
!**/resources/assets/
```

- 笔记配图通过普通 `git add` 提交；不要用 `git add -f` 强制提交下载的 PDF 或网页快照。
- 忽略规则不会取消已经被 Git 跟踪的文件。如果误提交，需要先执行 `git rm --cached <文件>`。
- 脚本不检查附件是否存在，缺文件不会导致 `check` 或 `build` 失败。
- 图片来源版本、图号和 PDF 页码写在笔记图注中，来源 PDF 或公开网址保留在条目元数据、笔记中。

## 安装 agent skill

skill 源码在 `skills/resource-curator/`，随仓库同步。agent 使用的是本机安装的副本，位置是 `$CODEX_HOME/skills/`；未设置 `CODEX_HOME` 时为 `~/.codex/skills/`。

- 新设备：把 `skills/resource-curator/` 复制到上述目录。
- 修改仓库里的 skill 后：重新复制一次，覆盖本机副本。
- 只在仓库内修改源码，不要直接改本机副本，否则两份内容会分叉。

## 提交前检查

```powershell
uv run python scripts/catalog.py build
uv run python -m unittest discover -s tests
```

- `build` 会先校验所有条目，失败时按报错提示修改。
- 新增标签时，先在 `tags.yaml` 中加入词条，再用于条目；不同标签不能共用相同的名称或别名。
- 确认 `git status` 中没有出现 `resources/` 下的 PDF 或网页快照；`resources/assets/` 下的笔记配图可以提交。
