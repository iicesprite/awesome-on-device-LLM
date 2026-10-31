// Template: zotero_sync.py prepends a fresh plan. Execute only through the
// authenticated Zotero MCP zotero_script tool, never via Node.js or SQLite.
// Adds tags/collection membership; preserves existing metadata, tags and files.
function assertAccount() {
  if (plan.library?.type !== 'user' || !Number.isSafeInteger(plan.library.id) ||
      plan.library.id <= 0 || Zotero.Users.getCurrentUserID() !== plan.library.id) {
    throw new Error('The logged-in Zotero account does not match this plan; prepare again');
  }
}
assertAccount(); // Must precede every possible write, including item creation.
const libraryID = Zotero.Libraries.userLibraryID;
const collections = await Zotero.Collections.getByLibrary(libraryID);
const collection = collections.find(c => c.name === plan.collection && !c.parentID);
if (!collection) throw new Error('Create the named collection with authenticated MCP library_update first');
const receipt = {library: plan.library, items: []};
async function readablePDF(attachment, parentID) {
  if (!attachment || attachment.deleted || !attachment.isAttachment() ||
      attachment.parentID !== parentID || attachment.attachmentContentType !== 'application/pdf') return false;
  const path = await attachment.getFilePathAsync();
  return Boolean(path && Zotero.File.pathToFile(path).exists());
}
for (const p of plan.papers) {
  assertAccount();
  let item = p.key ? await Zotero.Items.getByLibraryAndKeyAsync(libraryID, p.key) : null;
  if (p.key && (!item || item.deleted || !item.isRegularItem())) {
    throw new Error(`Stale item binding: ${p.id}`);
  }
  if (!item) {
    // A retry after partial success must reuse the item we already created.
    const marker = `awesome-LLM-infra: ${p.id}`;
    const search = new Zotero.Search();
    search.libraryID = libraryID;
    search.addCondition('extra', 'contains', marker);
    const candidates = (await Zotero.Items.getAsync(await search.search()))
      .filter(x => x.isRegularItem() && !x.deleted &&
        x.getField('extra').split('\n').includes(marker));
    if (candidates.length > 1) throw new Error(`Ambiguous retry: ${p.id}`);
    item = candidates[0];
    if (!item) {
      const type = p.venue ? (/IEEE Transactions/i.test(p.venue) ? 'journalArticle' : 'conferencePaper') : 'preprint';
      item = new Zotero.Item(type);
      item.libraryID = libraryID;
      item.setField('title', p.title);
      item.setField('url', p.url);
      if (p.year) item.setField('date', String(p.year));
      item.setField('extra', marker + (p.doi ? '\nDOI: ' + p.doi : ''));
      if (p.venue) item.setField(type === 'journalArticle' ? 'publicationTitle' : 'proceedingsTitle', p.venue);
      item.setCreators(p.authors.map(name => ({name, creatorType: 'author'})));
      assertAccount();
      await item.saveTx();
      env.recordCreatedItem(item);
    }
  }
  env.snapshot(item);
  for (const tag of p.tags) item.addTag(tag, 0);
  item.addToCollection(collection.id);
  assertAccount();
  await item.saveTx();
  const files = [];
  const usedAttachments = new Set();
  for (const f of p.files) {
    let attachment = f.key ? await Zotero.Items.getByLibraryAndKeyAsync(libraryID, f.key) : null;
    if (f.key && !await readablePDF(attachment, item.id)) {
      throw new Error(`Stale or unavailable PDF binding: ${p.id} ${f.relative_path}`);
    }
    if (!attachment && p.files.filter(other => other.url === f.url && other.title === f.title).length === 1) {
      // Exact version title plus source, not source alone. Never collapse two
      // known versions sharing a changing/unversioned URL.
      const candidates = [];
      const children = await Zotero.Items.getAsync(item.getAttachments());
      for (const child of children) {
        if (!usedAttachments.has(child.key) && child.getField('url') === f.url &&
            child.getField('title') === f.title &&
            await readablePDF(child, item.id)) candidates.push(child);
      }
      if (candidates.length > 1) throw new Error(`Ambiguous PDF retry: ${p.id} ${f.relative_path}`);
      attachment = candidates[0];
    }
    if (!attachment) {
      if (!f.file || !Zotero.File.pathToFile(f.file).exists()) {
        throw new Error(`Local PDF unavailable: ${p.id} ${f.relative_path}`);
      }
      assertAccount();
      attachment = await Zotero.Attachments.importFromFile({file: f.file, parentItemID: item.id});
      env.recordCreatedItem(attachment);
      attachment.setField('title', f.title);
      attachment.setField('url', f.url);
      assertAccount();
      await attachment.saveTx();
    }
    if (!await readablePDF(attachment, item.id)) throw new Error(`PDF unavailable: ${p.id}`);
    usedAttachments.add(attachment.key);
    files.push({relative_path: f.relative_path, key: attachment.key});
  }
  receipt.items.push({id: p.id, key: item.key, files});
}
assertAccount();
return JSON.stringify(receipt);
