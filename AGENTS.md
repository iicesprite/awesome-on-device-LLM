# AGENTS.md

端侧 LLM 推理与量化的学习资源库：每条资源一个目录，元数据是 YAML，阅读笔记是 Markdown，索引由脚本生成。没有应用代码，只有 `scripts/catalog.py` 一个脚本。

用户文档在 [README.md](README.md)，维护流程在 [CONTRIBUTING.md](CONTRIBUTING.md)。本文只写 agent 容易出错的约束。

## 语言与环境

- 文档、笔记、`summary` 用中文；标签键、ID、命令用英文。
- 默认 Windows + PowerShell，仓库根目录运行命令。
- 用 uv，不手动激活 `.venv`，不用 `pip install`。依赖变更见 CONTRIBUTING。

## 命令

```powershell
uv run --locked python scripts/catalog.py check    # 只校验，不写文件
uv run --locked python scripts/catalog.py build    # 校验并重新生成 INDEX.md、catalog.csv
uv run --locked python scripts/catalog.py search --tag 量化 --tag orin
uv run --locked python -m unittest discover -s tests   # 仅在修改 scripts/ 时需要
```

- 改了 `meta.yaml`、`notes.md` 或 `tags.yaml` 后，运行 `build`。
- 校验失败时按报错修改数据，不要为了通过校验去放宽 `scripts/catalog.py`。

## 目录

```text
papers/ blogs/ docs/           资源条目，目录即类型
<分类>/<id>/meta.yaml          必需
<分类>/<id>/notes.md           可选，阅读后才创建
<分类>/<id>/resources/         本地 PDF、网页快照、assets/ 原图；整体被 Git 忽略
<分类>/<id>/resources/asset-sources.yaml   原图来源，唯一入 Git 的文件
templates/                     meta.yaml、notes.md 模板，以模板为准
tags.yaml                      标签词表
skills/resource-curator/       收录与阅读 skill 的源码
```

## 硬性规则

**生成文件**

- `INDEX.md` 和 `catalog.csv` 是生成物，不手工编辑。

**条目**

- `id` 是小写英文、数字、连字符，与目录名一致，全库唯一（三个分类目录之间也不能重复）。
- `type` 必须与所在目录一致：`papers/` → `paper`，`blogs/` → `blog`，`docs/` → `doc`。
- 必填字段：`id`、`title`、`type`、`url`、`status`、`summary`、`tags`。可选的 `authors`、`year`、`venue`、`doi` 不确定时直接省略，不要猜。
- `status` 只表示用户的阅读进度：`unread` / `reading` / `read`。新条目默认 `unread`。agent 写了分析也不代表用户已读，不要改成 `read`。
- 同一篇论文的不同版本、发表页和代码放进同一条目，用 `links` 关联。添加前先按 DOI、arXiv ID、网址、标题在三个分类里查重。
- 代码链接要实际核对后再写进 `links`。

**标签**

- `tags` 只能用 `tags.yaml` 里的规范键（形如 `target/kv-cache`），不能写中文名或别名。
- 需要新标签时先在 `tags.yaml` 加词条（`label`、`aliases`），再用于条目。中文名和别名不能与其他标签冲突。
- 打标签要有证据：
  - 激活感知的权重量化不等于量化了激活。
  - KV 卸载或淘汰不等于 KV 量化。
  - `hardware/*` 和 `backend/*` 只标来源实际支持或评测的设备与后端。
  - `evidence/*` 只标实际检查过的实现，论文里的加速宣称不算。
- 用户要求先审阅标签时，建议存放在 `meta.yaml` 的 `tag_review` 扩展字段，不写入 `tags`，也不改词表。格式见 skill。

**附件与本地资料**

- `attachments[].path` 必须是 `resources/<文件名>`：正斜杠、相对路径，不含 `..`、盘符或绝对路径。
- 附件只记录来源，不要求文件存在。下载后不代表已读。
- 笔记里用相对路径引用：`resources/xxx.pdf`、`resources/assets/xxx.png`。
- 不用 `git add -f` 提交 `resources/` 下的 PDF 或图片。

**Git**

- 不主动 `commit` 或 `push`，除非用户明确要求。
- 不修改其他条目，除非用户要求。已有的笔记、状态和用户确认过的标签要保留。

## 写笔记

- 从 `templates/notes.md` 出发，标题改成资源题名，删掉不适用的章节和 HTML 注释提示。已有笔记先通读再整合，不整体覆盖。
- 开头写来源版本、阅读范围、阅读日期。页码一律注明是 PDF 页码还是论文页码。
- 数字要带单位、基线、设备、模型、batch、输入输出长度，以及 prefill 或 decode 条件。
- 区分三类内容：作者报告的结果、你的解释、本人实测。没运行过的东西不要写复现结果。
- 不把 kernel 加速当端到端收益，不把云端结果直接当端侧结论。
- 论文中有内部不一致、撤稿或证据缺口时要明确写出。
- 用自己的话总结，不整段复制正文。
- 引用原图时：图注写论文版本、图号、PDF 页码；在 `resources/asset-sources.yaml` 记录来源 PDF、网址、页码、裁剪坐标；关键结论仍用文字说明。

## 收录与阅读流程

收录、去重、标签审阅和阅读分析的完整流程见 [skills/resource-curator/SKILL.md](skills/resource-curator/SKILL.md)，处理这类请求时先读它。要点：

- 用户给出链接但没说明时，只收录，不阅读。
- 打开来源核对题名、作者、版本、撤稿或更正，不凭记忆拼网址。
- 无法访问全文时，只保存已核实的元数据，并说明实际读到的范围。
- 不运行论文代码、不装额外依赖、不用 GPU，除非用户另行授权。

## 修改脚本或模板

- `scripts/catalog.py` 的校验规则、`templates/` 和 `skills/resource-curator/SKILL.md` 需保持一致。改其中一个，要检查另外两个是否需要同步。
- 修改 `scripts/catalog.py` 后运行测试。`tests/` 被 `.gitignore` 忽略，只存在于本机。
- 修改 `skills/resource-curator/` 后，需要提醒用户重新复制到本机 skill 目录，见 CONTRIBUTING。

## 交付时说明

- 改动了哪些路径。
- `check` 或 `build` 是否通过。
- 待用户审阅的标签，以及真正影响使用的缺口（如 PDF 未下载、全文未读完）。
