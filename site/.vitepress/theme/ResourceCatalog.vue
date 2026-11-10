<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import catalog from '../../.generated/catalog.json'

type Entry = {
  id: string; title: string; type: string; url: string; status: string
  summary: string; tags: string[]; year?: number; authors?: string[]
  resource_path: string; notes_path: string; search_text: string
}
type Tag = { label: string; aliases: string[] }
const entries = catalog.entries as Entry[]
const tags = catalog.tags as Record<string, Tag>
const keyword = ref('')
const resourceType = ref('')
const notesOnly = ref(false)
const selectedTags = ref<string[]>([])
const tagSearch = ref('')
const mounted = ref(false)
const typeLabels: Record<string, string> = { paper: '论文', blog: '博客', doc: '文档' }
const statusLabels: Record<string, string> = { unread: '未读', reading: '在读', read: '已读' }
const dimensions: Record<string, string> = {
  topic: '主题', architecture: '模型架构', transform: '变换', availability: '开放情况',
  target: '优化对象', scheme: '量化方案', granularity: '量化粒度', precision: '精度',
  method: '方法', system: '系统优化', workload: '工作负载', stage: '推理阶段',
  memory: '内存架构', hardware: '硬件', backend: '后端', evidence: '实现证据',
}
const normalize = (value: string) => value.trim().toLocaleLowerCase()
const tagLabel = (key: string) => tags[key]?.label || key
const tagLookup = new Map<string, string>()
for (const [key, tag] of Object.entries(tags)) {
  for (const name of [key, tag.label, ...tag.aliases]) tagLookup.set(normalize(name), key)
}
const searchable = entries.map(entry => ({
  entry,
  text: normalize([entry.search_text, ...entry.tags.flatMap(key => {
    const tag = tags[key]
    return tag ? [key, tag.label, ...tag.aliases] : [key]
  })].join(' ')),
}))
const filteredEntries = computed(() => {
  const words = normalize(keyword.value).split(/\s+/).filter(Boolean)
  return searchable.filter(({ entry, text }) =>
    (!resourceType.value || entry.type === resourceType.value) &&
    (!notesOnly.value || Boolean(entry.notes_path)) &&
    selectedTags.value.every(key => entry.tags.includes(key)) &&
    words.every(word => text.includes(word)),
  ).map(({ entry }) => entry)
})
const tagGroups = computed(() => {
  const query = normalize(tagSearch.value)
  const groups = new Map<string, { key: string; label: string }[]>()
  for (const [key, tag] of Object.entries(tags)) {
    if (query && ![key, tag.label, ...tag.aliases].some(text => normalize(text).includes(query))) continue
    const dimension = key.split('/')[0]
    if (!groups.has(dimension)) groups.set(dimension, [])
    groups.get(dimension)!.push({ key, label: tag.label })
  }
  return [...groups].map(([key, items]) => ({ key, label: dimensions[key] || key, items }))
})
function toggleTag(key: string) {
  selectedTags.value = selectedTags.value.includes(key)
    ? selectedTags.value.filter(tag => tag !== key)
    : [...selectedTags.value, key]
}
function reset() {
  keyword.value = ''
  resourceType.value = ''
  notesOnly.value = false
  selectedTags.value = []
  tagSearch.value = ''
}
function readQuery() {
  const params = new URLSearchParams(window.location.search)
  keyword.value = params.get('q') || ''
  const type = params.get('type') || ''
  resourceType.value = type in typeLabels ? type : ''
  notesOnly.value = params.get('notes') === '1'
  selectedTags.value = [...new Set(params.getAll('tag')
    .map(value => tagLookup.get(normalize(value))).filter((value): value is string => Boolean(value)))]
}
function writeQuery() {
  if (!mounted.value) return
  const url = new URL(window.location.href)
  for (const key of ['q', 'type', 'notes', 'tag']) url.searchParams.delete(key)
  if (keyword.value.trim()) url.searchParams.set('q', keyword.value.trim())
  if (resourceType.value) url.searchParams.set('type', resourceType.value)
  if (notesOnly.value) url.searchParams.set('notes', '1')
  for (const tag of selectedTags.value) url.searchParams.append('tag', tag)
  if (url.href !== window.location.href) window.history.replaceState(window.history.state, '', url)
}
watch([keyword, resourceType, notesOnly, selectedTags], writeQuery)
onMounted(() => {
  readQuery()
  mounted.value = true
  window.addEventListener('popstate', readQuery)
  window.addEventListener('catalog:navigate', readQuery)
})
onUnmounted(() => {
  window.removeEventListener('popstate', readQuery)
  window.removeEventListener('catalog:navigate', readQuery)
})
</script>

<template>
  <main class="resource-catalog" aria-labelledby="catalog-title">
    <header class="catalog-intro">
      <p class="catalog-eyebrow">端侧推理 · 量化 · 系统优化</p>
      <h1 id="catalog-title">端侧 LLM 资源库</h1>
      <p>检索论文、博客和技术文档，沿着标签找到相关资源与阅读笔记。</p>
      <p class="catalog-hint">本页搜索资源元数据；查找笔记正文请使用导航栏的全站搜索。本站仅供阅读。</p>
    </header>

    <section class="catalog-filters" aria-label="资源筛选">
      <div class="catalog-scope" role="group" aria-label="资源范围">
        <button type="button" data-testid="scope-all" :aria-pressed="!notesOnly" @click="notesOnly = false">全部资源</button>
        <button type="button" data-testid="scope-notes" :aria-pressed="notesOnly" @click="notesOnly = true">仅有笔记</button>
      </div>
      <div class="catalog-search-row">
        <label class="catalog-keyword" for="catalog-keyword">关键词（资源元数据）
          <input id="catalog-keyword" v-model="keyword" type="search" placeholder="标题、摘要、作者、标签…" autocomplete="off" />
        </label>
        <label for="catalog-type">资源类型
          <select id="catalog-type" v-model="resourceType">
            <option value="">全部类型</option>
            <option v-for="(label, key) in typeLabels" :key="key" :value="key">{{ label }}</option>
          </select>
        </label>
      </div>
      <details class="catalog-tag-panel">
        <summary>按标签筛选 <span>（多选取交集 · 已选 {{ selectedTags.length }} 项）</span></summary>
        <label for="catalog-tag-search">查找标签（支持中文名、规范键与别名）
          <input id="catalog-tag-search" v-model="tagSearch" type="search" placeholder="例如：量化、orin、kv-cache" autocomplete="off" />
        </label>
        <div v-if="!tagGroups.length" class="catalog-hint" role="status">没有匹配的标签。</div>
        <div v-for="group in tagGroups" :key="group.key" class="catalog-tag-group">
          <h2>{{ group.label }}</h2>
          <div class="catalog-tags">
            <button v-for="tag in group.items" :key="tag.key" type="button" :data-tag="tag.key"
              :aria-pressed="selectedTags.includes(tag.key)" :title="tag.key" @click="toggleTag(tag.key)">{{ tag.label }}</button>
          </div>
        </div>
      </details>
      <div v-if="selectedTags.length" class="catalog-selected" aria-label="已选标签">
        <button v-for="tag in selectedTags" :key="tag" type="button" :aria-label="`移除标签：${tagLabel(tag)}`"
          :data-selected-tag="tag" @click="toggleTag(tag)">{{ tagLabel(tag) }} <span aria-hidden="true">×</span></button>
      </div>
      <div class="catalog-results-bar">
        <p role="status" aria-live="polite" aria-atomic="true">找到 <strong data-testid="result-count">{{ filteredEntries.length }}</strong> / {{ entries.length }} 条资源</p>
        <button class="catalog-reset" type="button" @click="reset">重置筛选</button>
      </div>
    </section>

    <div v-if="!filteredEntries.length" class="catalog-empty" role="status">
      <h2>没有匹配的资源</h2>
      <p>试试减少标签、切换资源类型，或使用更短的关键词。</p>
      <button type="button" @click="reset">查看全部资源</button>
    </div>
    <div v-else class="catalog-grid" data-testid="resource-list">
      <article v-for="entry in filteredEntries" :key="entry.id" class="resource-card" :data-resource-id="entry.id">
        <div class="resource-card-meta">
          <span>{{ typeLabels[entry.type] || entry.type }}</span>
          <span v-if="entry.year">{{ entry.year }}</span>
          <span>{{ statusLabels[entry.status] || entry.status }}</span>
        </div>
        <h2><a :href="withBase('/' + entry.resource_path)">{{ entry.title }}</a></h2>
        <p class="resource-summary">{{ entry.summary }}</p>
        <div class="catalog-tags resource-card-tags" aria-label="资源标签">
          <button v-for="tag in entry.tags" :key="tag" type="button" :data-tag="tag"
            :aria-pressed="selectedTags.includes(tag)" :title="tag" @click="toggleTag(tag)">{{ tagLabel(tag) }}</button>
        </div>
        <div class="resource-card-links">
          <a :href="withBase('/' + entry.resource_path)">查看资源 <span aria-hidden="true">→</span></a>
          <a v-if="entry.notes_path" class="resource-notes-link" :href="withBase('/' + entry.notes_path)">阅读笔记 <span aria-hidden="true">→</span></a>
          <span v-else class="resource-no-notes">暂无笔记</span>
        </div>
      </article>
    </div>
  </main>
</template>
