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
| `skills/resource-curator/` | agent 收录与阅读 skill 的源码 |
| `INDEX.md`、`catalog.csv` | 自动生成，不手工编辑 |
| `pyproject.toml`、`uv.lock`、`.python-version` | 环境与依赖 |

## 生成文件的格式

`catalog.csv` 的约定：

- 编码为 UTF-8 BOM，方便 Excel 直接打开。
- 标签以分号分隔。
- 链接和附件字段使用 JSON。

`INDEX.md` 顶部有自动生成的提示。需要修改内容时，改 `meta.yaml` 后重新运行 `build`。

## 本地资料的 Git 规则

`.gitignore` 忽略每个条目 `resources/` 下的所有文件，只放行 `asset-sources.yaml`：

```gitignore
**/resources/*
!**/resources/asset-sources.yaml
```

- 不要用 `git add -f` 强制提交下载的 PDF 或原图。
- 忽略规则不会取消已经被 Git 跟踪的文件。如果误提交，需要先执行 `git rm --cached <文件>`。
- 脚本不检查附件是否存在，缺文件不会导致 `check` 或 `build` 失败。

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
- 确认 `git status` 中没有出现 `resources/` 下的 PDF 或图片。
