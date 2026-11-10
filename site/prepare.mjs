import { spawnSync } from 'node:child_process';
import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const site = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(site);
const base = process.env.SITE_BASE || '/';
const result = spawnSync('uv', ['run', '--locked', 'python', 'scripts/site_export.py', '--base', base], {
  cwd: root, stdio: 'inherit', shell: false,
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status || 1);
const generated = path.join(site, '.generated');
writeFileSync(path.join(generated, 'build-config.json'), JSON.stringify({ base }));
const vendor = path.join(generated, 'public', 'vendor');
mkdirSync(vendor, { recursive: true });
for (const name of ['mermaid.min.js', 'mermaid.LICENSE']) {
  copyFileSync(path.join(root, 'web', 'vendor', name), path.join(vendor, name));
}
