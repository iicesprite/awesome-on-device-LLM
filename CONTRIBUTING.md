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

前端相关回归在 `tests/test_web.py`、`tests/test_web_store.py`、`tests/test_note_render.py` 和 `tests/test_resource_import.py`，覆盖 HTTP 写入边界、并发编辑冲突、字段保留、失败回滚、路径隔离、公式分隔符与代码边界、笔记渲染及来源查重。Windows 未开启符号链接权限时，相关用例会明确跳过；目录联接另有 Windows 专用用例。

## 本地前端开发

```powershell
uv run --locked python scripts/web.py
# 对隔离副本做写入验证，避免改动真实条目
uv run --locked python scripts/web.py --root C:/path/to/catalog-copy --port 8766
```

默认地址为 `http://127.0.0.1:8765`；端口占用时换用 `--port`，`--port 0` 可分配空闲端口，以终端打印的地址为准。`--open` 自动打开浏览器，终端按 `Ctrl+C` 停止。`--root` 指定完整的资源库根目录，不能只指向某篇论文目录。

`web/` 是原生 HTML/CSS/JavaScript，不需要 npm 安装或打包。修改页面后刷新浏览器；修改 Python 后重启服务。更改后除了单元测试，还应在浏览器实际检查多标签交集、编辑保存与冲突、网址预览/收录/重复拦截，以及笔记图片、公式和 Mermaid 图，包含窄屏布局。

验证时区分两类操作：网址预览允许联网但不应写文件；确认收录或保存元数据会修改资源卡并自动重建索引。检查已有/缺失 PDF 的展示，不将附件记录当作下载成功；笔记是只读渲染，编辑仍在 Markdown 文件中完成。Web 收录不触发 Zotero 联动，不能以网页保存成功代替 MCP 回执核验。

- `scripts/web.py` 仅绑定 `127.0.0.1`，检查 Host、Origin 和写入令牌；没有账号体系，不支持对外提供服务。
- `web_store.py` 直接读取 `resource.md`，复用 `catalog.py` 校验和索引渲染。保存使用字段 patch、revision 检查、同目录临时文件替换与失败回滚；不能用前端 JSON 投影覆盖整个 YAML。单进程锁不协调另一个服务进程或外部编辑器，避免多实例写入和保存同时运行 CLI `build`。
- 遇到外部编辑冲突，网页保留输入；先复制修改，再重新读取并合并。文件替换与失败回滚不等于跨文件原子事务，若报告回滚未完成，先检查资源卡及两个索引、解决权限问题，再重新校验和构建，不盲目重试覆盖。
- `resource_import.py` 只抓取公网 HTML，DNS 地址固定到连接，逐跳检查重定向；单次上限 15 秒、2 MiB、5 次重定向。不使用浏览器登录态或环境代理，不抓取内网地址。
- `note_render.py` 使用 `markdown-it-py` 和 `mdit-py-plugins` 解析 Markdown 与公式边界，原始 HTML 禁用，LaTeX 作为转义后的文本片段传给前端。`app.js` 调用本地 KaTeX，使用 `htmlAndMathml` 输出、`trust: false`，限制宏展开次数和显式尺寸；每个公式使用独立宏状态，不能借公式注入链接、图片或 HTML。不再使用 Python 的 `latex2mathml` 转换或自定义 MathML 白名单。本地图片/附件仍只通过当前资源的 `resources/` 路由访问，拒绝符号链接、Windows 目录联接和路径穿越，不暴露 `.local/` 或 Git 文件。

离线 Mermaid 使用 npm 发布包 [mermaid 11.12.1](https://registry.npmjs.org/mermaid/11.12.1) 中原样提取的 `dist/mermaid.min.js`，许可证保存在 `web/vendor/mermaid.LICENSE`。文件 SHA-256 为 `198b19442f86f0b46a17e56d3abf744c3a28a3427e66fa8dada3b589a77babcc`；升级时核对发布包完整性、保留许可证和组件内置声明，并实际检查严格安全模式下的图表渲染。不要改用运行时 CDN。

离线 KaTeX 使用 [katex 0.19.1](https://registry.npmjs.org/katex/0.19.1) 发布包，在核对包的 SHA-512 integrity 后原样提取 `dist/katex.min.js`、`dist/katex.min.css`、`dist/fonts/` 和许可证到 `web/vendor/katex/`。JS 的 SHA-256 为 `fa140498b8cfc8302cbe5e7cb1bdbc1f952bc38b3fa258ede09e73edbf9a74bc`，CSS 为 `00e0b094171b07f58422075f469b4feac2b2f7bce1aeec943c08da4d697f4dcf`。升级时脚本、样式、字体必须来自同一版本，保留 `LICENSE`；字体目录保持在 CSS 同级，只通过限定文件名的静态路由提供。运行不需要 Node.js。

公式变更后用浏览器检查 `$...$`、`$$...$$`、`\(...\)`、`\[...\]`，以及代码不被误解析、错误公式不阻断后文、重复切换页签不重复渲染。还应检查本地字体响应、窄屏长公式的横向滚动，以及 Mermaid 图和笔记配图没有被公式样式影响。后端解析变更须重启服务后再刷新页面，不能只刷新仍运行旧 Python 模块的服务。


## GitHub Pages 开发与部署

`site/` 是独立的 VitePress 默认主题站点，Node.js 22+ 构建；`web/` 仍为原有本地 Python 工作台，两者不相互替代。站点依赖及版本由 `site/package.json`、`site/package-lock.json` 管理，使用 `npm ci --prefix site` 安装，不提交 `node_modules`。

```powershell
uv run --locked python -m unittest discover -s tests
npm ci --prefix site
npm run build --prefix site
npm run preview --prefix site
```

构建命令先调用 `site/prepare.mjs`，通过 uv 运行 `scripts/site_export.py`，再构建 VitePress。导出结果 `site/.generated/`、缓存和 `site/.vitepress/dist/` 都被 Git 忽略，不手工维护。`resource.md` 和独立笔记仍为源文件，分类目录 `docs/` 不是 VitePress 站点目录。

验证项目子路径（PowerShell）：

```powershell
$env:SITE_BASE = "/awesome-on-device-LLM/"
npm run build --prefix site
npm run preview --prefix site
# 完成后恢复本地默认根路径；改变 base 需要重新构建
Remove-Item Env:SITE_BASE
```

默认本地 `base` 为 `/`；构建时记录到生成配置中，preview 自动使用对应路径。GitHub Actions 对普通项目使用 `/<仓库名>/`，对 `<用户>.github.io` 仓库使用 `/`。部署到自定义域名根目录时，需要相应调整工作流的 `SITE_BASE`。

导出仅包含白名单元数据，未知扩展字段与附件列表不进入公开 JSON。笔记标签从资源卡派生；本地链接只映射到已知资源/笔记页面。原始 HTML 禁用，Vue 模板插值保持为文本，公式复用本地工作台的四种分隔符解析，浏览器通过 KaTeX 排版。仅复制笔记实际引用的 `resources/assets/` 栅格图，缺图显示提示；不复制 PDF、个人配置、任意本地文件或符号链接目标。图表脚本从已锁定的本地 Mermaid 组件复制，不使用运行时 CDN。

`tests/test_site_export.py` 覆盖只读源文件、输出目录保护、公开字段、注入、图片与 PDF 边界、相对链接和 base。除了测试，还需用实际生产构建检查标签交集、可分享网址、同页“阅读笔记”导航、中文全文搜索与命中跳转、SPA 切页后的公式/图表、深色模式和窄屏。重新构建后重启 preview 服务，避免服务仍缓存旧产物。

发布工作流只上传 `site/.vitepress/dist`，无写回资源权限。`main` 推送和手动执行可构建，只有 `main` 可以部署到 `github-pages` 环境；PR 不发布。首次需在仓库设置启用 Pages 的 GitHub Actions 来源。本地构建不会自动提交、推送或修改 GitHub 设置。

## 项目文件

| 路径 | 作用 |
| --- | --- |
| `papers/`、`blogs/`、`docs/` | 资源条目，按类型分类 |
| `templates/` | `resource.md` 资源卡和独立的 `notes.md` 模板 |
| `tags.yaml` | 标签词表：规范键、中文名称、搜索别名 |
| `scripts/catalog.py` | 校验、索引生成与搜索 |
| `scripts/web.py`、`scripts/web_store.py` | 本机 HTTP 服务、元数据读写与索引更新 |
| `scripts/resource_import.py`、`scripts/note_render.py` | 公网来源预览查重、安全笔记渲染 |
| `web/` | 浏览器界面及离线 Mermaid 组件 |
| `site/`、`.github/workflows/pages.yml` | VitePress 只读站点、锁定的 Node 依赖与 GitHub Pages 部署 |
| `scripts/site_export.py`、`tests/test_site_export.py` | 静态内容导出及隐私/路由边界回归 |
| `scripts/zotero_sync.py`、`scripts/zotero_import.js` | 生成带账号约束的 Zotero MCP 同步计划，核验回执并保存本地关联，见 [ZOTERO.md](ZOTERO.md) |
| `skills/resource-curator/` | agent 收录与阅读 skill 的源码 |
| `INDEX.md`、`catalog.csv` | 自动生成，不手工编辑 |
| `资源库.base` | 共享的 Obsidian Bases 视图，只收录分类目录下的 `resource.md` |
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

`INDEX.md` 顶部有自动生成的提示。手动或通过 Obsidian 修改 `resource.md` 的 YAML frontmatter 后需运行 `build`；Web 成功创建或保存元数据会自动更新两个索引。新收录时同时创建零字节 `notes.md`，并在资源卡正文添加 `[阅读笔记](notes.md)`；空文件不表示已阅读，开始撰写时再按模板填写。历史条目允许缺少笔记，不自动补建或覆盖已有内容。`metadata_path` 指向资源卡，`notes_path` 由独立笔记是否存在派生；卡片正文不作为索引元数据。`check`、`build` 仍只校验及生成索引，不替收录者创建笔记或改写资源卡。

每条资源只有这一份 frontmatter 元数据，不再维护独立 `meta.yaml`，也没有旧格式回退。迁移保留原字段、ID、状态、标签、自定义字段、附件相对路径和已有笔记，附件不移动。Obsidian 使用方法见 [README](README.md#用-obsidian-管理资源)；简单属性可在 Bases 编辑，嵌套的 `links`、`attachments` 在 YAML 源码中编辑，仍须通过 `check` / `build` 的标签与状态校验。

公开元数据禁止顶层 `zotero`，附件禁止 `zotero_item_key`、`zotero_attachment_key`、`local_status`；校验时即使值为空也会拒绝。个人下载时间 `downloaded_on` 也不应写进共享元数据。可核实的 `downloaded_from`、`downloaded_version` 及历史 `source_zotero_*` 可作为来源事实保留，但不是当前账号绑定。

个人 Zotero 联动仅支持已登录账号的个人库：各用户独立运行 `configure`，再按 `prepare`（下载须显式 `--download`）、已认证 MCP、`record`、`index` 的顺序执行，完整命令见 [ZOTERO.md](ZOTERO.md)。空库必须配置自己的正整数 `--user-id`。不同账号不能共用绑定；同一账号换设备先完成 Zotero 数据/文件同步，再重新核验并生成个人索引。不要复制作者的绑定、旧计划或回执。

`prepare` 及 `prepare --download` 都面向全库论文，没有单篇过滤参数；若只有一篇获得下载授权，先按其附件记录单独保存文件。Zotero 已有可读 PDF 时，`--download` 不会将它复制回仓库。附件恢复与失效绑定处理见 [ZOTERO.md](ZOTERO.md#修改或删除-zotero-内容后)。

## 本地资料的 Git 规则

`.gitignore` 忽略每个条目 `resources/` 下的本地资料，只放行笔记配图目录 `assets/`：

```gitignore
**/resources/*
!**/resources/assets/
```

- 笔记配图通过普通 `git add` 提交；不要用 `git add -f` 强制提交下载的 PDF 或网页快照。
- 忽略规则不会取消已经被 Git 跟踪的文件。如果误提交，需要先执行 `git rm --cached <文件>`。
- `catalog.py check/build` 不检查附件是否存在；Web 打开附件、Zotero 联动会检查实际文件。目录校验通过不代表 PDF 已下载。
- 图片来源版本、图号和 PDF 页码写在笔记图注中，来源 PDF 或公开网址保留在条目元数据、笔记中。
- `.obsidian/` 是本机 vault 设置，不入 Git；根目录 `资源库.base` 需要提交，供各设备共享视图。
- Bases 的筛选、排序和列设置也属于共享视图修改；提交前检查是否误将临时筛选留在“全部资源”中，不删除顶层的资源卡范围过滤。

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
