# AGENTS.md

端侧 LLM 推理与量化的学习资源库：每条资源一个目录，`resource.md` 的 YAML frontmatter 是唯一元数据源，新收录时创建独立的空 `notes.md`，索引由脚本生成。`scripts/catalog.py` 负责校验、构建与搜索，`scripts/web.py` 和 `web/` 提供本地前端，`scripts/zotero_sync.py` 和 `scripts/zotero_import.js` 负责 Zotero 联动，流程见 [ZOTERO.md](ZOTERO.md)。

用户文档在 [README.md](README.md)，维护流程在 [CONTRIBUTING.md](CONTRIBUTING.md)。本文只写 agent 容易出错的约束。

## 语言与环境

- 文档、笔记、`summary` 用中文；标签键、ID、命令用英文。
- 默认 Windows + PowerShell，仓库根目录运行命令。
- 用 uv，不手动激活 `.venv`，不用 `pip install`。依赖变更见 CONTRIBUTING。
- `site/` 是独立的 VitePress 只读站点，使用 Node.js 22+ 与 `npm ci --prefix site`；Python 导出仍通过 uv。`docs/` 是资源分类，不能当作站点源码目录。

## 命令

```powershell
uv run --locked python scripts/catalog.py check    # 只校验，不写文件
uv run --locked python scripts/catalog.py build    # 校验并重新生成 INDEX.md、catalog.csv
uv run --locked python scripts/catalog.py search --tag 量化 --tag orin
uv run --locked python scripts/web.py --open       # 启动本机前端，不需要 Node.js
uv run --locked python -m unittest discover -s tests   # 仅在修改 scripts/ 时需要
npm run build --prefix site                        # 生成静态资料并构建 GitHub Pages
```

- 手动或通过 Obsidian 修改 `resource.md`、`notes.md`、`tags.yaml` 后运行 `build`；Web 成功创建或保存元数据会自动更新索引，不与其写入同时运行 CLI `build`。
- 校验失败时按报错修改数据，不要为了通过校验去放宽 `scripts/catalog.py`。
- Pages 的 `site/.generated/` 和 `site/.vitepress/dist/` 是忽略的生成物，不手工编辑。元数据仍来自资源卡，笔记沿用其标签；不发布 PDF、私人配置或第二份可写元数据。改站点后实际验证生产构建、项目子路径、中文搜索与笔记渲染。

## 目录

```text
papers/ blogs/ docs/           资源条目，目录即类型
<分类>/<id>/resource.md        必需，开头是 --- 包围的 YAML frontmatter
<分类>/<id>/notes.md           新收录时创建空文件，开始阅读后填写；历史条目可缺省
<分类>/<id>/resources/         本地 PDF、网页快照，被 Git 忽略
<分类>/<id>/resources/assets/  笔记配图，随 Git 同步
templates/                     resource.md、notes.md 模板，以模板为准
tags.yaml                      标签词表
skills/resource-curator/       收录与阅读 skill 的源码
资源库.base                    共享 Obsidian Bases 视图，只纳入分类目录下的 resource.md
scripts/web.py                 仅监听本机的 HTTP 服务
scripts/web_store.py           资源卡读写、冲突检查与索引更新
scripts/resource_import.py     公网 HTML 元数据预览与查重
scripts/note_render.py         只读笔记的安全渲染
web/                          浏览器界面及本地 Mermaid 组件
scripts/site_export.py          只读导出公开静态页面、检索数据与必要配图
site/                         VitePress 默认主题与资源检索首页
.github/workflows/pages.yml    GitHub Pages 构建与发布（仅 main 部署）
```

## 硬性规则

**生成文件**

- `INDEX.md` 和 `catalog.csv` 是生成物，不手工编辑。

**条目**

- `id` 是小写英文、数字、连字符，与目录名一致，全库唯一（三个分类目录之间也不能重复）。
- `type` 必须与所在目录一致：`papers/` → `paper`，`blogs/` → `blog`，`docs/` → `doc`。
- 必填字段：`id`、`title`、`type`、`url`、`status`、`summary`、`tags`。可选的 `authors`、`year`、`venue`、`doi` 不确定时直接省略，不要猜。
- 不创建独立 `meta.yaml`，不保留双份可写元数据或旧格式回退。资源卡正文可选，不作为索引元数据；修改 frontmatter 时保留已有正文、自定义字段与附件相对路径。
- `metadata_path`、`notes_path` 是派生保留字段，分别指向资源卡和存在时的独立笔记，不写入 frontmatter。
- `status` 只表示用户的阅读进度：`unread` / `reading` / `read`。新条目默认 `unread`。agent 写了分析也不代表用户已读，不要改成 `read`。
- 同一篇论文的不同版本、发表页和代码放进同一条目，用 `links` 关联。添加前先按 DOI、arXiv ID、网址、标题在三个分类里查重。
- 代码链接要实际核对后再写进 `links`。
- Web 预览只是页面声明的元数据，不证明全文已读、版本最新或撤稿状态已核实。确认收录前仍需核对来源；Web 不下载 PDF、不调用 Zotero，确认新建时创建空笔记及跳转链接。
- Web 和 Obsidian 编辑同一份资源卡。遇到 revision 冲突，保留输入、重新读取并合并，不绕过检查强行覆盖。只运行一个 Web 写入服务，不把单进程锁当作跨进程保护。

**标签**

- `tags` 只能用 `tags.yaml` 里的规范键（形如 `target/kv-cache`），不能写中文名或别名。
- 需要新标签时先在 `tags.yaml` 加词条（`label`、`aliases`），再用于条目。中文名和别名不能与其他标签冲突。
- 打标签要有证据：
  - 激活感知的权重量化不等于量化了激活。
  - KV 卸载或淘汰不等于 KV 量化。
  - `hardware/*` 和 `backend/*` 只标来源实际支持或评测的设备与后端。
  - `evidence/*` 只标实际检查过的实现，论文里的加速宣称不算。
- 用户要求先审阅标签时，建议存放在 `resource.md` frontmatter 的 `tag_review` 扩展字段，不写入 `tags`，也不改词表。格式见 skill。

**附件与本地资料**

- `attachments[].path` 必须是 `resources/<文件名>`：正斜杠、相对路径，不含 `..`、盘符或绝对路径。
- 附件只记录来源，不要求文件存在。下载后不代表已读。
- 笔记里用相对路径引用：`resources/xxx.pdf`、`resources/assets/xxx.png`。
- 不用 `git add -f` 提交 `resources/` 下的 PDF。
- 个人 Zotero 绑定、账号和可用状态只存 `.local/zotero/`，不得放入公共资源卡；具体禁止字段见模板与 ZOTERO。
- Zotero `prepare` 遍历全库论文，无单篇参数；执行全库下载或 MCP 写入前确认授权范围。`--download` 不会把已在 Zotero 可读的 PDF 自动复制回仓库，不是单篇下载或全量恢复工具。

**Git**

- 不主动 `commit` 或 `push`，除非用户明确要求。
- 不修改其他条目，除非用户要求。已有的笔记、状态和用户确认过的标签要保留。
- `.obsidian/` 是忽略的本机设置，根目录 `.base` 视图随 Git 共享；不要将笔记或模板纳入资源视图。

## 写笔记

- 从 `templates/notes.md` 出发，标题改成资源题名，删掉不适用的章节和 HTML 注释提示。已有笔记先通读再整合，不整体覆盖。
- 新收录时在同目录创建零字节 `notes.md`，并在 `resource.md` 正文添加 `[阅读笔记](notes.md)`；不填模板或占位内容，不改变阅读状态。历史条目可缺少笔记，不批量补建或覆盖已有内容；笔记独立保存，不迁入资源卡。
- 开头写来源版本、阅读范围和本次关注问题。页码一律注明是 PDF 页码还是论文页码。
- 数字要带单位、基线、设备、模型、batch、输入输出长度，以及 prefill 或 decode 条件。
- 区分三类内容：作者报告的结果、你的解释、本人实测。没运行过的东西不要写复现结果。
- 不把 kernel 加速当端到端收益，不把云端结果直接当端侧结论。
- 论文中有内部不一致、撤稿或证据缺口时要明确写出。
- 用自己的话总结，不整段复制正文。
- 引用原图时：图注写论文版本、图号、PDF 页码，来源 PDF 或公开网址保留在条目元数据、笔记中；关键结论仍用文字说明。

## 收录与阅读流程

收录、去重、标签审阅和阅读分析的完整流程见 [skills/resource-curator/SKILL.md](skills/resource-curator/SKILL.md)，处理这类请求时先读它。要点：

- 用户给出链接但没说明时，只收录，不阅读。
- 打开来源核对题名、作者、版本、撤稿或更正，不凭记忆拼网址。
- 无法访问全文时，只保存已核实的元数据，并说明实际读到的范围。
- 不运行论文代码、不装额外依赖、不用 GPU，除非用户另行授权。

## 修改脚本或模板

- `scripts/catalog.py` 的校验规则、`templates/` 和 `skills/resource-curator/SKILL.md` 需保持一致。改其中一个，要检查另外两个是否需要同步。
- 修改 `scripts/` 后运行测试。`tests/test_catalog.py` 和 `tests/test_zotero_sync.py` 分别覆盖目录脚本与 Zotero 联动；`tests/test_web*.py`、`tests/test_resource_import.py` 覆盖本地前端后端。
  修改 `web/` 后用浏览器实际验证受影响的操作。写入测试使用临时资源库，不改现有条目；前端不得建立第二份元数据源或放宽 `catalog.py` 校验。
- 修改 `skills/resource-curator/` 后，需要提醒用户重新复制到本机 skill 目录，见 CONTRIBUTING。

## 交付时说明

- 改动了哪些路径。
- `check` 或 `build` 是否通过。
- 待用户审阅的标签，以及真正影响使用的缺口（如 PDF 未下载、全文未读完）。
