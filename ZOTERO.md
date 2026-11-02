# 用自己的 Zotero 阅读仓库论文

公开仓库只保存资源身份、来源、版本、标签和阅读笔记；每个人独立选择自己的 Zotero 个人库、保存关联并生成个人索引。公共 `INDEX.md` 和 `catalog.csv` 不含个人 Zotero 链接，也不读取 `.local/`。不要复制作者或其他用户的绑定来使用。

同步目标是个人库中的 `awesome-on-device-LLM` 集合。已有论文、标签、笔记、其他集合和阅读状态保留；仓库的 PDF 复制为 Zotero 存储附件，不移动原件。**目前只支持已登录 Zotero 账号的个人库，不支持群组库或未登录的本地库。** 正整数账号 ID 是账号隔离的必要条件，不是本地路由 `0`，也不是插件内部的 libraryID。

仓库已更名为 `awesome-on-device-LLM`；Zotero 集合名和导入标记仍保留 `awesome-on-device-LLM`，避免更名影响已有同步记录。

## 先连接自己的 Zotero

1. 打开并登录自己的 Zotero，安装、配置自己的 LLM-for-Zotero 插件和 MCP 客户端。
2. 使用插件给出的认证配置连接 MCP。默认本机端点为：

   ```text
   http://127.0.0.1:23119/llm-for-zotero/mcp
   ```

   实际写入只能通过**已认证 MCP 的 `zotero_script` 工具**进行，按工具要求声明 `access: privileged`、`effect: write`。使用自己安装的插件、配置及 token；不要提交凭据、粘贴到聊天、关闭认证或直接修改 Zotero SQLite。客户端显示连接不代表当前聊天已加载工具，必要时重新加载会话。
3. 本地只读 API 默认使用 `http://127.0.0.1:23119`。在 Zotero 设置中启用允许本机应用访问的 API；它与 MCP 写入端点不是同一个 URL。配置中不保存 token。

## 配置与账号隔离

以下命令在仓库根目录执行：

```powershell
uv run --locked python scripts/zotero_sync.py configure
```

非空个人库通过 `/api/users/0/items?format=json&limit=1` 返回的 `library.id` 发现当前账号，再保存正整数用户 ID。首次运行 `prepare` 时也可自动完成这一步。**空库没有可用于发现账号的条目，必须明确提供自己的 Zotero 用户 ID**：

```powershell
uv run --locked python scripts/zotero_sync.py configure --user-id 12345678
```

`12345678` 仅为示例，须换成自己的账号 ID。脚本通过 `/api/users/<id>/items?format=json&limit=1` 验证归属，不能把其他人的 ID 当成配置。需更改本机端口时，可使用：

```powershell
uv run --locked python scripts/zotero_sync.py configure --api-base http://127.0.0.1:23119 --user-id 12345678
```

`--api-base` 只接受本机回环 HTTP(S) 源地址，不带 `/api` 后缀、路径、凭据或 token。配置保存到 `.local/zotero/config.json`，格式为：

```json
{"api_base":"http://127.0.0.1:23119","user_id":12345678}
```

发现账号后，所有 API 读取都使用明确的 `/api/users/<正整数ID>/...`，不再以 `users/0` 代表任意当前账号。切换账号后，原配置应安全失败；为新账号重新执行 `configure` 和 `prepare`。任何 MCP 写入前都必须核对 `Zotero.Users.getCurrentUserID() === plan.library.id`，不匹配时不能创建集合、条目或附件。导入模板也会在自身写入前重新检查。不要为绕过检查而手改计划或回执中的账号。

## 准备、导入与核验

### 1. 准备本地计划

```powershell
uv run --locked python scripts/zotero_sync.py prepare
```

默认只读 Zotero 和已有本地资料，**不下载、不写入 Zotero、不改公开元数据**。按当前账号的本地绑定、DOI/arXiv ID、规范化完整标题查找论文；歧义、身份不符或失效绑定需先处理，不能猜测替换。复用 PDF 前检查本机实体文件，不能只看附件条目存在。

允许从公开来源补充缺失 PDF 时，必须显式选择：

```powershell
uv run --locked python scripts/zotero_sync.py prepare --download
```

下载只针对需要补充的公网 HTTP(S) 附件，优先使用已记录的 `downloaded_from`，不覆盖可读文件；检查 PDF 文件头后才保存。下载失败逐文件报告，不阻塞其他论文；先前仅导入元数据的条目可随后用 `--download` 补齐。版本不明的历史来源不会自动重建，需先确认版本或手动补充确切文件。缺文件不影响公共目录构建。已撤稿且无正文的论文仍可保留已核实的元数据与撤稿标签；不要用另一篇论文或不明版本冒充正文。PDF、网页快照及原图不入 Git。

计划写到 `.local/zotero/plan.json`，可执行模板写到 `.local/zotero/import.js`。计划包含明确的 `library: {"type":"user","id":12345678}`、集合及论文列表。执行前检查目标账号、论文身份、来源版本、标签和待复制文件；准备计划不代表已经同步。

### 2. 通过已认证 MCP 写入

让已连接自己 Zotero MCP 的 agent 按顺序执行：

1. 阅读 `.local/zotero/plan.json`，先通过**只读 MCP 调用**取得 `Zotero.Users.getCurrentUserID()`，与 `plan.library.id` 比较。不相等或无法确认时停止，不执行后续写入。
2. 通过 MCP 的 `library_update`（`kind: collection`、`action: create`）在已核对的个人库创建 `awesome-on-device-LLM` 集合；已有则复用。不要在未核对账号时预先建集合。
3. 阅读并通过 `zotero_script` 执行 `.local/zotero/import.js`。脚本再次核对计划账号，查找已有集合，然后创建缺失条目、增量添加规范标签及复制缺失 PDF。源码模板为 `scripts/zotero_import.js`。集合创建不由导入模板负责。

任一步骤之间如果切换了账号，必须停止并重新核对。不得以直接改库、未认证接口或其他脚本执行入口替代。

保留工具返回的 actionId，以便按 MCP 的恢复机制处理误操作。部分导入失败时保留已完成工作，重新 `prepare` 后继续，不要删掉已有条目重跑。新建条目的仓库 ID 标记用于防重复。

### 3. 保存回执并验证

将 MCP 返回的 JSON **对象**保存为 `.local/zotero/receipt.json`。结构为：

```json
{
  "library": {"type":"user","id":12345678},
  "items": [
    {
      "id":"example-paper",
      "key":"ABCD1234",
      "files":[{"relative_path":"resources/paper.pdf","key":"PDF12345"}]
    }
  ]
}
```

账号、条目和附件 key 必须来自实际执行结果，示例不是可用回执。旧版无账号计划、裸数组回执均已废弃；不要复用，重新准备并取得新回执。

```powershell
uv run --locked python scripts/zotero_sync.py record .local/zotero/receipt.json
uv run --locked python scripts/zotero_sync.py index
```

`record` 核对当前配置、计划与回执的账号和预期附件，再从 Zotero 读取论文身份、规范标签、附件父条目和本机文件。全部通过后才保存当前账号的本地绑定，**从不回写 `meta.yaml`**。不要手工伪造成功回执。`index` 重新生成 `.local/zotero/index.md`；不需要为个人关联执行 `catalog.py build`。

## 本地状态与个人索引

以下文件均在 `.local/zotero/`，由 Git 忽略：

| 文件 | 用途 |
| --- | --- |
| `config.json` | 本机只读 API 源地址与当前用户 ID，不含认证凭据 |
| `bindings.json` | 按用户隔离的论文/附件关联、附件可用状态及必要的历史迁移信息 |
| `plan.json`、`import.js` | 当前账号的待执行计划及 MCP 导入脚本 |
| `receipt.json` | 实际 MCP 执行回执，需经 `record` 验证 |
| `index.md` | 当前用户的 Zotero 条目和阅读 PDF 链接 |

绑定文件使用以下分层结构；同一个资源 ID 在不同用户库里可以有不同 key：

```json
{
  "version": 1,
  "libraries": {
    "user:12345678": {
      "library": {"type":"user","id":12345678},
      "entries": {
        "example-paper": {
          "item_key":"ABCD1234",
          "reading_attachment_key":"PDF12345",
          "attachments": {
            "resources/paper.pdf":{"key":"PDF12345","status":"available"}
          }
        }
      }
    }
  }
}
```

必要时本地条目还包含 `legacy_metadata`，仅用于保留迁移历史。不要将这份示例或别人的实际绑定当成自己的关联。附件缺失与下载时间也是个人本地状态，不写入共享元数据；`downloaded_from`、`downloaded_version` 等可核实的来源/版本事实仍可保留。已有 `source_zotero_*` 只描述历史来源，**从不作为当前绑定或自动恢复依据**。公开元数据禁止顶层 `zotero` 及附件中的 `zotero_item_key`、`zotero_attachment_key`、`local_status`、`downloaded_on`。

在本机 Markdown 查看器中打开 `.local/zotero/index.md`，可跳转 Zotero 条目或阅读 PDF；需要安装 Zotero 且允许 `zotero://` 协议。公开网页可能屏蔽该协议，所以个人入口不放进公共索引或 CSV。Zotero 集合内仍可通过 `topic/quantization` 等标签筛选。

## 多用户、换设备与维护

- 每位用户自行配置自己的账号、MCP 与 token，独立执行 `prepare`、MCP 导入、`record`、`index`。克隆仓库不应带来作者的个人绑定。
- **同一账号换设备**：先登录同一 Zotero 账号，完成 Zotero 数据及文件同步，在新设备配置本机 API/MCP，再重新 `prepare`、经 MCP 补齐、`record` 和 `index`。不复制旧设备的计划、回执或个人索引；本地关联由当前库重新核验。若保留同账号的本地绑定备份，也必须重新核验，不能将旧索引作为成功证据。
- **换账号**：重新 `configure`，为目标账号生成全新计划和回执；绑定按 `user:<id>` 分开保存，不能挪用其他账号的 key。
- 仓库现有绑定的移出由本次仓库迁移完成，公共构建不会从旧 `meta.yaml` 自动恢复关联；发现私有字段时会直接拒绝。
- 标签只增量添加：仓库删掉某标签不会自动删除 Zotero 原标签，也不反向覆盖仓库。
- 不同 PDF 版本分别保留；未绑定附件必须同时核对来源与版本标题或文件名，不能仅凭同 URL 复用。已有本地绑定明确指向同一附件且来源、版本一致的多个路径可保留；不据此自动合并其他版本，不替换有批注的旧 PDF。PDF 跨设备传输由 Zotero 文件同步或个人备份负责，Git 不保存它们。

本流程不包含笔记/批注同步，也不自动标记已读。是否完成同步以实际 MCP 回执及 `record` 核验结果为准，不以计划生成、文档示例或索引存在为准。
