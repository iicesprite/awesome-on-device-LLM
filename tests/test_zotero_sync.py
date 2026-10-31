from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import zotero_sync as sync


ACCOUNT = 12345
ITEM = 'ITEM0001'
PDF = 'PDF00001'


def item(key, title, doi='', tags=()):
    return {'key': key, 'library': {'type': 'user', 'id': ACCOUNT, 'name': 'Test user'},
            'data': {'itemType': 'journalArticle', 'title': title, 'DOI': doi,
                     'tags': [{'tag': tag} for tag in tags]}}


class ZoteroSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local = self.root / '.local' / 'zotero'
        self.config = {'api_base': 'http://127.0.0.1:23119', 'user_id': ACCOUNT}
        self.active_account = ACCOUNT
        self.entries, self.items, self.children, self.file_paths = [], [], {}, {}
        scripts = self.root / 'scripts'
        scripts.mkdir()
        shutil.copyfile(Path(sync.__file__).with_name('zotero_import.js'), scripts / 'zotero_import.js')
        for target, value in [('ROOT', self.root), ('LOCAL', self.local)]:
            patcher = patch.object(sync, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        catalog_patch = patch.object(sync.catalog, 'load_catalog', side_effect=lambda root: (self.entries, {}, {}))
        catalog_patch.start()
        self.addCleanup(catalog_patch.stop)
        api_patch = patch.object(sync, 'get', side_effect=self.api)
        self.get = api_patch.start()
        self.addCleanup(api_patch.stop)
        sync.save_json(self.local / 'config.json', self.config)

    def api(self, route, config):
        if config['user_id'] not in (0, self.active_account):
            raise HTTPError('http://localhost/', 400, 'Wrong account', {}, None)
        if route == '/items?format=json&limit=1':
            return self.items[:1]
        if route.startswith('/items/top?'):
            return self.items
        if '/children?' in route:
            return self.children.get(route.split('/')[2], [])
        key = route.split('/')[2].split('?')[0]
        if route.endswith('/file/view/url'):
            return self.file_paths.get(key, self.root / 'absent.pdf').as_uri()
        for value in self.items + [child for children in self.children.values() for child in children]:
            if value['key'] == key:
                return value
        raise HTTPError('http://localhost/', 404, 'Missing item', {}, None)

    def paper(self, identity='a', attachments=None):
        folder = self.root / 'papers' / identity
        folder.mkdir(parents=True)
        (folder / 'meta.yaml').write_text('public metadata must not change\n', encoding='utf-8')
        entry = {'id': identity, 'type': 'paper', 'title': f'Paper {identity}',
                 'url': f'https://example.org/{identity}', 'tags': ['topic/inference'],
                 'metadata_path': f'papers/{identity}/meta.yaml', 'notes_path': '',
                 'attachments': attachments or []}
        self.entries.append(entry)
        return entry

    def pdf(self, path, parent=ITEM, key=PDF, url='https://example.org/a.pdf', title='v1'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'%PDF-1.7\ncontent')
        self.file_paths[key] = path
        self.children.setdefault(parent, []).append({
            'key': key, 'data': {'itemType': 'attachment', 'parentItem': parent,
                                  'contentType': 'application/pdf', 'url': url,
                                  'title': title, 'filename': path.name}})

    def bind(self, entries, account=ACCOUNT):
        path = self.local / 'bindings.json'
        data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'version': 1, 'libraries': {}}
        data['libraries'][f'user:{account}'] = {'library': {'type': 'user', 'id': account}, 'entries': entries}
        sync.save_json(path, data)

    def receipt(self, rows, account=ACCOUNT):
        path = self.local / 'receipt.json'
        sync.save_json(path, {'library': {'type': 'user', 'id': account}, 'items': rows})
        return path

    def prepare_record_fixture(self):
        entry = self.paper(attachments=[{'path': 'resources/a.pdf', 'url': 'https://example.org/a.pdf', 'version': 'v1'},
                                        {'path': 'resources/old.pdf', 'url': 'https://example.org/old.pdf', 'version': 'old'}])
        self.items.append(item(ITEM, entry['title'], tags=entry['tags']))
        self.pdf(self.root / 'papers/a/resources/a.pdf')
        self.bind({'a': {'item_key': ITEM, 'legacy_metadata': {'library_id': 0},
                         'attachments': {'resources/a.pdf': {'key': PDF, 'status': 'available',
                                                              'downloaded_on': '2026-01-01',
                                                              'legacy_metadata': {'source': 'old'}}}}})
        sync.prepare()
        return entry, self.receipt([{'id': 'a', 'key': ITEM, 'files': [{'relative_path': 'resources/a.pdf', 'key': PDF}]}])

    def assert_record_rejected_without_writes(self, receipt):
        before = {path: path.read_bytes() for path in (self.local / 'bindings.json', self.local / 'index.md') if path.exists()}
        with self.assertRaises((ValueError, HTTPError)):
            sync.record(receipt)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_title_unicode_punctuation_and_case(self):
        self.assertEqual(sync.title_key('ＡＷＱ: A-B'), sync.title_key('awq a b'))

    def test_doi_and_arxiv_version_identity(self):
        self.assertIn('10.1234/abc', sync.identities({'doi': '10.1234/ABC'}))
        self.assertEqual(sync.identities({'url': 'https://arxiv.org/pdf/2605.05819v1'}),
                         sync.identities({'url': 'https://arxiv.org/abs/2605.05819v2'}))
        self.assertIn('arxiv:cs/9901001', sync.identities({'extra': 'arXiv:cs/9901001v2'}))

    def test_related_candidates_are_not_paper_identity(self):
        entry = {'id': 'a', 'title': 'Published paper', 'doi': '10.1234/published',
                 'links': {'related-version-candidate': 'https://arxiv.org/abs/2501.14794'}}
        candidate = item(ITEM, 'Unconfirmed related paper')
        candidate['data']['url'] = 'https://arxiv.org/abs/2501.14794'
        self.assertIsNone(sync.match(entry, [candidate]))
        self.assertEqual(sync.identities({'links': {'arxiv': 'https://arxiv.org/html/2501.14794v1'}}),
                         {'arxiv:2501.14794'})

    def test_binding_is_local_and_stale_binding_fails(self):
        entry = {'id': 'a', 'title': 'A', 'zotero': {'item_key': 'AUTHOR01'}}
        self.assertIsNone(sync.match(entry, []))
        self.assertEqual(sync.match(entry, [item(ITEM, 'A')], {'item_key': ITEM})['key'], ITEM)
        for items in ([], [item(ITEM, 'Different paper')]):
            with self.assertRaises(ValueError):
                sync.match(entry, items, {'item_key': ITEM})

    def test_doi_precedes_title_and_conflicting_title_is_not_reused(self):
        entry = {'id': 'a', 'title': 'New name', 'doi': '10.1234/abc'}
        candidates = [item('OTHER001', 'New name', '10.1234/wrong'), item(ITEM, 'Old name', '10.1234/ABC')]
        self.assertEqual(sync.match(entry, candidates)['key'], ITEM)
        self.assertIsNone(sync.match(entry, candidates[:1]))
        with self.assertRaises(ValueError):
            sync.match(entry, candidates + [item('OTHER002', 'Other', '10.1234/abc')])

    def test_exact_full_title_and_ambiguity(self):
        entry = {'id': 'a', 'title': 'A full title'}
        self.assertIsNone(sync.match(entry, [item(ITEM, 'A full title extended')]))
        with self.assertRaises(ValueError):
            sync.match(entry, [item(ITEM, entry['title']), item('OTHER001', entry['title'])])

    def test_api_origin_only_accepts_loopback_without_secrets(self):
        for value in ('http://127.0.0.1:23119', 'http://localhost:23119/', 'https://[::1]:23119'):
            self.assertEqual(sync.api_base(value), value.rstrip('/'))
        for value in ('https://example.org', 'http://127.0.0.1/api', 'http://name:secret@localhost',
                      'http://localhost?token=x', 'http://localhost#fragment', 'file:///tmp/x',
                      'http://localhost:bad', 'http://localhost /'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sync.api_base(value)

    def test_empty_library_needs_explicit_positive_user_id(self):
        (self.local / 'config.json').unlink()
        with self.assertRaises(ValueError):
            sync.configure()
        self.assertFalse((self.local / 'config.json').exists())
        self.assertEqual(sync.configure(user_id=ACCOUNT), self.config)
        for account in (0, -1, ACCOUNT + 1):
            before = (self.local / 'config.json').read_bytes()
            with self.assertRaises((ValueError, HTTPError)):
                sync.configure(user_id=account)
            self.assertEqual((self.local / 'config.json').read_bytes(), before)

    def test_discovery_uses_library_id_then_numeric_validation(self):
        self.items.append(item(ITEM, 'A'))
        self.assertEqual(sync.configure(), self.config)

    def test_account_separation_and_index_switch_does_not_reuse_old_links(self):
        entry = self.paper()
        self.bind({'a': {'item_key': 'AUTHOR01', 'attachments': {}}}, account=ACCOUNT - 1)
        (self.local / 'index.md').write_text('old author link', encoding='utf-8')
        sync.save_json(self.local / 'config.json', {**self.config, 'user_id': ACCOUNT - 1})
        sync.configure(user_id=ACCOUNT)
        self.assertNotIn('old author link', (self.local / 'index.md').read_text(encoding='utf-8'))
        plan = sync.prepare()
        self.assertIsNone(plan['papers'][0]['key'])
        sync.index()
        text = (self.local / 'index.md').read_text(encoding='utf-8')
        self.assertNotIn('AUTHOR01', text)
        self.assertIn(entry['title'], text)

    def test_wrong_active_account_cannot_load_bindings_or_replace_index(self):
        self.paper()
        self.bind({'a': {'item_key': ITEM}})
        (self.local / 'index.md').write_text('keep index', encoding='utf-8')
        self.active_account = ACCOUNT + 1
        with self.assertRaises(HTTPError):
            sync.load_bindings(self.config)
        with self.assertRaises(HTTPError):
            sync.index()
        self.assertEqual((self.local / 'index.md').read_text(encoding='utf-8'), 'keep index')

    def test_prepare_includes_metadata_only_and_partial_papers(self):
        self.paper('empty')
        entry = self.paper('partial', [{'path': 'resources/yes.pdf', 'url': 'https://example.org/yes.pdf'},
                                       {'path': 'resources/no.pdf', 'url': 'https://example.org/no.pdf'}])
        path = self.root / 'papers/partial/resources/yes.pdf'
        path.parent.mkdir()
        path.write_bytes(b'%PDF-1.7')
        entry['zotero'] = {'item_key': 'AUTHOR01'}
        plan = sync.prepare()
        jobs = {row['id']: row for row in plan['papers']}
        self.assertEqual(plan['library'], {'type': 'user', 'id': ACCOUNT})
        self.assertEqual(jobs['empty']['files'], [])
        self.assertEqual(len(jobs['partial']['files']), 1)
        self.assertEqual(jobs['partial']['missing_files'][0]['relative_path'], 'resources/no.pdf')
        self.assertTrue(jobs['partial']['missing_files'][0]['reason'])
        self.assertIsNone(jobs['partial']['key'])
        self.assertFalse((self.local / 'bindings.json').exists())

    def test_actual_file_required_for_zotero_attachment(self):
        missing = self.root / 'missing.pdf'
        self.file_paths[PDF] = missing
        self.assertFalse(sync.attachment_available(PDF, self.config))
        missing.write_bytes(b'%PDF-')
        self.assertTrue(sync.attachment_available(PDF, self.config))

    def test_same_url_does_not_revive_missing_or_collapse_versions(self):
        url = 'https://arxiv.org/pdf/2605.05819'
        entry = self.paper(attachments=[{'path': 'resources/old.pdf', 'url': url},
                                        {'path': 'resources/new.pdf', 'url': url, 'version': 'v2'}])
        self.items.append(item(ITEM, entry['title']))
        self.pdf(self.root / 'zotero/new.pdf', url=url, title='v2')
        self.bind({'a': {'item_key': ITEM, 'attachments': {'resources/old.pdf': {'status': 'missing'}}}})
        plan = sync.prepare()
        self.assertEqual([f["relative_path"] for f in plan['papers'][0]['files']], ['resources/new.pdf'])
        self.assertEqual(plan['papers'][0]['files'][0]['key'], PDF)
        self.assertEqual([f["relative_path"] for f in plan['papers'][0]['missing_files']], ['resources/old.pdf'])

    def test_verified_same_source_bindings_can_share_existing_attachment(self):
        url = 'https://example.org/a.pdf'
        paths = ('resources/first-copy.pdf', 'resources/second-copy.pdf')
        entry = self.paper(attachments=[{'path': path, 'url': url, 'version': 'v1'} for path in paths])
        self.items.append(item(ITEM, entry['title'], tags=entry['tags']))
        self.pdf(self.root / 'zotero/a.pdf', url=url)
        self.bind({'a': {'item_key': ITEM, 'attachments': {
            path: {'key': PDF, 'status': 'available'} for path in paths}}})
        plan = sync.prepare()
        self.assertEqual([file['key'] for file in plan['papers'][0]['files']], [PDF, PDF])
        sync.record(self.receipt([{'id': 'a', 'key': ITEM, 'files': [
            {'relative_path': path, 'key': PDF} for path in paths]}]))
        saved = json.loads((self.local / 'bindings.json').read_text(encoding='utf-8'))
        attachments = saved['libraries'][f'user:{ACCOUNT}']['entries']['a']['attachments']
        self.assertEqual({path: value['key'] for path, value in attachments.items()}, dict.fromkeys(paths, PDF))
        entry['attachments'][1]['version'] = 'v2'
        with self.assertRaises(ValueError):
            sync.prepare()

    def test_metadata_only_binding_can_later_download_its_pdf(self):
        entry = self.paper(attachments=[{'path': 'resources/a.pdf', 'url': 'https://example.org/a.pdf'}])
        self.items.append(item(ITEM, entry['title'], tags=entry['tags']))
        sync.prepare()
        sync.record(self.receipt([{'id': 'a', 'key': ITEM, 'files': []}]))
        def download(url, path):
            path.parent.mkdir(parents=True)
            path.write_bytes(b'%PDF-1.7')
        with patch.object(sync, 'download_pdf', side_effect=download):
            plan = sync.prepare(download=True)
        self.assertEqual([file['relative_path'] for file in plan['papers'][0]['files']], ['resources/a.pdf'])
        self.assertEqual(plan['papers'][0]['missing_files'], [])

    def test_record_rejects_merging_unbound_versions_into_one_attachment(self):
        entry = self.paper(attachments=[
            {'path': 'resources/old.pdf', 'url': 'https://example.org/old.pdf', 'version': 'v1'},
            {'path': 'resources/new.pdf', 'url': 'https://example.org/new.pdf', 'version': 'v2'},
        ])
        self.items.append(item(ITEM, entry['title'], tags=entry['tags']))
        for attachment in entry['attachments']:
            path = self.root / 'papers/a' / attachment['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'%PDF-1.7')
        self.pdf(self.root / 'zotero/a.pdf')
        self.bind({})
        sync.prepare()
        receipt = self.receipt([{'id': 'a', 'key': ITEM, 'files': [
            {'relative_path': attachment['path'], 'key': PDF} for attachment in entry['attachments']]}])
        self.assert_record_rejected_without_writes(receipt)

    def test_download_rejects_nonpublic_source(self):
        destination = self.root / 'resources/private.pdf'
        with self.assertRaises(ValueError):
            sync.download_pdf('http://127.0.0.1/private.pdf', destination)
        self.assertFalse(destination.exists())

    def test_ambiguous_pdf_and_stale_attachment_binding_fail(self):
        entry = self.paper(attachments=[{'path': 'resources/a.pdf', 'url': 'https://example.org/a.pdf', 'version': 'v1'}])
        self.items.append(item(ITEM, entry['title']))
        self.pdf(self.root / 'zotero/a.pdf')
        self.pdf(self.root / 'zotero/b.pdf', key='PDF00002')
        with self.assertRaises(ValueError):
            sync.prepare()
        self.bind({'a': {'item_key': ITEM, 'attachments': {'resources/a.pdf': {'key': 'ABSENT01', 'status': 'available'}}}})
        with self.assertRaises(ValueError):
            sync.prepare()

    def test_download_failures_do_not_block_other_papers(self):
        self.paper('a', [{'path': 'resources/a.pdf', 'url': 'https://example.org/a.pdf'}])
        self.paper('b', [{'path': 'resources/b.pdf', 'url': 'https://example.org/b.pdf',
                          'downloaded_from': 'https://authors.example/b.pdf', 'downloaded_version': 'author version'}])
        def download(url, path):
            if url.endswith('/a.pdf'):
                raise OSError('network unavailable')
            path.parent.mkdir(parents=True)
            path.write_bytes(b'%PDF-1.7')
        with patch.object(sync, 'download_pdf', side_effect=download):
            plan = sync.prepare(download=True)
        self.assertIn('network unavailable', plan['papers'][0]['missing_files'][0]['reason'])
        self.assertEqual(plan['papers'][1]['files'][0]['url'], 'https://authors.example/b.pdf')
        self.assertEqual(plan['papers'][1]['files'][0]['title'], 'author version')

    def test_historical_missing_download_does_not_guess_version(self):
        entry = self.paper(attachments=[{'path': 'resources/old.pdf', 'url': 'https://arxiv.org/pdf/2605.05819',
                                        'source_zotero_attachment_key': 'AUTHOR01'}])
        with patch.object(sync, 'download_pdf') as fetch:
            plan = sync.prepare(download=True)
        fetch.assert_not_called()
        self.assertIn('未注明确切版本', plan['papers'][0]['missing_files'][0]['reason'])
        self.assertEqual(plan['papers'][0]['files'], [])

    def test_download_magic_check_is_atomic_and_never_executes_content(self):
        destination = self.root / 'resources/a.pdf'
        destination.parent.mkdir()
        destination.write_bytes(b'original')
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'<html>Not PDF</html>')
        with patch.object(sync, 'public_url'), patch.object(sync, 'build_opener', return_value=opener):
            with self.assertRaises(ValueError):
                sync.download_pdf('https://example.org/a.pdf', destination)
        self.assertEqual(destination.read_bytes(), b'original')
        self.assertEqual(list(destination.parent.iterdir()), [destination])
        opener.open.return_value = io.BytesIO(b'%PDF-1.7\nbody')
        with patch.object(sync, 'public_url'), patch.object(sync, 'build_opener', return_value=opener):
            sync.download_pdf('https://example.org/a.pdf', destination)
        self.assertEqual(destination.read_bytes(), b'%PDF-1.7\nbody')

    def test_record_metadata_only_and_missing_status_are_local(self):
        entry = self.paper(attachments=[{'path': 'resources/missing.pdf', 'url': 'https://example.org/missing.pdf'}])
        before = (self.root / entry['metadata_path']).read_bytes()
        sync.prepare()
        self.items.append(item(ITEM, entry['title'], tags=entry['tags']))
        sync.record(self.receipt([{'id': 'a', 'key': ITEM, 'files': []}]))
        data = json.loads((self.local / 'bindings.json').read_text(encoding='utf-8'))
        saved = data['libraries'][f'user:{ACCOUNT}']['entries']['a']
        self.assertEqual(saved['attachments']['resources/missing.pdf']['status'], 'missing')
        self.assertNotIn('reading_attachment_key', saved)
        self.assertEqual((self.root / entry['metadata_path']).read_bytes(), before)
        text = (self.local / 'index.md').read_text(encoding='utf-8')
        self.assertIn('缺失', text)
        self.assertNotIn('zotero://open-pdf', text)

    def test_receipt_omissions_duplicates_and_account_mismatch_preserve_bindings(self):
        _, receipt = self.prepare_record_fixture()
        (self.local / 'index.md').write_text('keep existing index', encoding='utf-8')
        original = json.loads(receipt.read_text(encoding='utf-8'))
        cases = []
        omitted = deepcopy(original)
        omitted['items'][0]['files'] = []
        cases.append(omitted)
        duplicate_file = deepcopy(original)
        duplicate_file['items'][0]['files'] *= 2
        cases.append(duplicate_file)
        duplicate_item = deepcopy(original)
        duplicate_item['items'] *= 2
        cases.append(duplicate_item)
        wrong_account = deepcopy(original)
        wrong_account['library']['id'] += 1
        cases.append(wrong_account)
        for value in cases:
            with self.subTest(receipt=value):
                sync.save_json(receipt, value)
                self.assert_record_rejected_without_writes(receipt)
        sync.save_json(receipt, original)
        plan = json.loads((self.local / 'plan.json').read_text(encoding='utf-8'))
        plan['library']['id'] += 1
        sync.save_json(self.local / 'plan.json', plan)
        self.assert_record_rejected_without_writes(receipt)

    def test_record_rejects_missing_file_wrong_parent_type_and_tags(self):
        _, receipt = self.prepare_record_fixture()
        original = deepcopy(self.children[ITEM][0]['data'])
        for changes in ({'parentItem': 'OTHER001'}, {'contentType': 'text/html'}, {'itemType': 'note'}):
            self.children[ITEM][0]['data'] = {**original, **changes}
            self.assert_record_rejected_without_writes(receipt)
        self.children[ITEM][0]['data'] = original
        actual = self.file_paths.pop(PDF)
        self.assert_record_rejected_without_writes(receipt)
        self.file_paths[PDF] = actual
        self.items[0]['data']['tags'] = []
        self.assert_record_rejected_without_writes(receipt)

    def test_record_preserves_legacy_details_other_accounts_and_note_selection(self):
        entry, receipt = self.prepare_record_fixture()
        self.bind({'unrelated': {'item_key': 'AUTHOR01'}}, account=ACCOUNT + 1)
        notes = self.root / 'papers/a/notes.md'
        notes.write_text('[Old](resources/old.pdf)\n[Read](resources/a.pdf)\n', encoding='utf-8')
        entry['notes_path'] = 'papers/a/notes.md'
        sync.record(receipt)
        data = json.loads((self.local / 'bindings.json').read_text(encoding='utf-8'))
        self.assertEqual(data['libraries'][f'user:{ACCOUNT + 1}']['entries']['unrelated']['item_key'], 'AUTHOR01')
        saved = data['libraries'][f'user:{ACCOUNT}']['entries']['a']
        self.assertEqual(saved['legacy_metadata'], {'library_id': 0})
        self.assertEqual(saved['attachments']['resources/a.pdf']['downloaded_on'], '2026-01-01')
        self.assertEqual(saved['attachments']['resources/a.pdf']['legacy_metadata'], {'source': 'old'})
        self.assertEqual(saved['reading_attachment_key'], PDF)
        text = (self.local / 'index.md').read_text(encoding='utf-8')
        self.assertIn('../../papers/a/notes.md', text)
        self.assertIn(f'zotero://open-pdf/library/items/{PDF}', text)
        self.assertNotIn('AUTHOR01', text)

    def test_index_prefers_first_note_pdf_then_available_fallback_and_rechecks(self):
        entry = self.paper(attachments=[{'path': 'resources/a.pdf', 'url': 'https://example.org/a.pdf'},
                                        {'path': 'resources/b.pdf', 'url': 'https://example.org/b.pdf'}])
        entry['title'] = 'Paper | [unsafe] <x>'
        self.items.append(item(ITEM, entry['title']))
        self.pdf(self.root / 'zotero/a.pdf')
        self.pdf(self.root / 'zotero/b.pdf', key='PDF00002')
        notes = self.root / 'papers/a/notes.md'
        notes.write_text('[B](resources/b.pdf)\n[A](resources/a.pdf)', encoding='utf-8')
        entry['notes_path'] = 'papers/a/notes.md'
        self.bind({'a': {'item_key': ITEM, 'reading_attachment_key': PDF,
                         'attachments': {'resources/a.pdf': {'key': PDF, 'status': 'available'},
                                         'resources/b.pdf': {'key': 'PDF00002', 'status': 'available'}}}})
        sync.index()
        text = (self.local / 'index.md').read_text(encoding='utf-8')
        self.assertIn('zotero://open-pdf/library/items/PDF00002', text)
        self.assertIn('&#124;', text)
        self.assertIn('&lt;x&gt;', text)
        self.file_paths['PDF00002'].unlink()
        sync.index()
        text = (self.local / 'index.md').read_text(encoding='utf-8')
        self.assertIn(f'zotero://open-pdf/library/items/{PDF}', text)
        self.assertNotIn('zotero://open-pdf/library/items/PDF00002', text)
        self.assertIn('resources/b.pdf：缺失', text)
        self.bind({'a': {'item_key': ITEM, 'attachments': {
            'resources/a.pdf': {'key': PDF, 'status': 'available'},
            'resources/b.pdf': {'key': 'PDF00002', 'status': 'missing'},
        }}})
        self.file_paths['PDF00002'].write_bytes(b'%PDF-1.7 restored after Zotero sync')
        sync.index()
        text = (self.local / 'index.md').read_text(encoding='utf-8')
        self.assertIn('zotero://open-pdf/library/items/PDF00002', text)


@unittest.skipUnless(shutil.which('node'), 'Node.js unavailable; importer behavior tests require Node')
class ImporterBehaviorTests(unittest.TestCase):
    """Run our local template against an in-memory API, never a Zotero process."""

    def execute(self, *, account=ACCOUNT, files=(), available=True):
        source = Path(sync.__file__).with_name('zotero_import.js').read_text(encoding='utf-8')
        plan = {'library': {'type': 'user', 'id': ACCOUNT}, 'collection': 'awesome-LLM-infra',
                'papers': [{'id': 'a', 'key': ITEM, 'title': 'A', 'tags': ['topic/inference'],
                            'authors': [], 'files': list(files)}]}
        harness = r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const counters = {saves: 0, imports: 0, snapshots: 0, created: 0};
const attachment = {
  id: 2, key: 'PDF00001', parentID: 1, deleted: false,
  attachmentContentType: 'application/pdf', isAttachment: () => true,
  getFilePathAsync: async () => '/mock/attachment.pdf',
  getField: () => '', setField: () => {}, saveTx: async () => {counters.saves++;}
};
const item = {
  id: 1, key: 'ITEM0001', deleted: false, isRegularItem: () => true,
  addTag: () => {}, addToCollection: () => {}, getAttachments: () => [],
  saveTx: async () => {counters.saves++;}
};
const Zotero = {
  Users: {getCurrentUserID: () => input.account},
  Libraries: {userLibraryID: 1},
  Collections: {getByLibrary: async () => [{id: 3, name: input.plan.collection}]},
  Items: {getByLibraryAndKeyAsync: async (library, key) => key === 'ITEM0001' ? item : attachment,
          getAsync: async () => []},
  File: {pathToFile: () => ({exists: () => input.available})},
  Attachments: {importFromFile: async args => {
    counters.imports++; counters.copy = args; return attachment;
  }}
};
const env = {snapshot: () => {counters.snapshots++;},
             recordCreatedItem: () => {counters.created++;}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  try {
    const receipt = await new AsyncFunction('Zotero', 'env', 'plan', input.source)(Zotero, env, input.plan);
    console.log(JSON.stringify({receipt: JSON.parse(receipt), counters}));
  } catch (error) {
    console.log(JSON.stringify({error: error.message, counters}));
  }
})();
"""
        result = subprocess.run([shutil.which('node'), '-e', harness],
                                input=json.dumps({'source': source, 'plan': plan, 'account': account,
                                                  'available': available}),
                                text=True, capture_output=True, check=True, timeout=15)
        return json.loads(result.stdout)

    def test_account_guard_runs_before_any_write(self):
        result = self.execute(account=ACCOUNT + 1)
        self.assertIn('does not match', result['error'])
        self.assertEqual(result['counters']['saves'], 0)
        self.assertEqual(result['counters']['imports'], 0)
        self.assertEqual(result['counters']['snapshots'], 0)


    def test_attachment_path_without_actual_file_is_rejected(self):
        result = self.execute(available=False, files=[
            {'relative_path': 'resources/a.pdf', 'key': PDF, 'file': None, 'url': 'https://example.org/a.pdf', 'title': 'v1'}])
        self.assertIn('unavailable PDF', result['error'])
        self.assertEqual(result['counters']['imports'], 0)



if __name__ == '__main__':
    unittest.main()
