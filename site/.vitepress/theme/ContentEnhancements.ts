import { defineComponent, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'
import katex from 'katex'

type Mermaid = {
  initialize: (options: Record<string, unknown>) => void
  render: (id: string, source: string, container?: HTMLElement) => Promise<{ svg: string }>
}
type MermaidWindow = Window & { mermaid?: Mermaid }
let mermaidLoader: Promise<Mermaid> | undefined
let diagramSequence = 0

function loadMermaid(): Promise<Mermaid> {
  if (mermaidLoader) return mermaidLoader
  const { promise, resolve, reject } = Promise.withResolvers<Mermaid>()
  mermaidLoader = promise
  const existing = (window as MermaidWindow).mermaid
  if (existing) {
    resolve(existing)
  } else {
    const script = document.createElement('script')
    script.src = withBase('/vendor/mermaid.min.js')
    script.async = true
    script.onload = () => {
      const mermaid = (window as MermaidWindow).mermaid
      if (mermaid) resolve(mermaid)
      else { script.remove(); reject(new Error('本地图表组件未能初始化')) }
    }
    script.onerror = () => { script.remove(); reject(new Error('本地图表组件加载失败')) }
    document.head.append(script)
  }
  mermaidLoader.catch(() => { mermaidLoader = undefined })
  return mermaidLoader
}

function showWarning(parent: HTMLElement, message: string) {
  if (parent.querySelector('.content-render-warning')) return
  const warning = document.createElement('p')
  warning.className = 'content-render-warning'
  warning.setAttribute('role', 'status')
  warning.textContent = message
  parent.append(warning)
}

function renderMath(root: HTMLElement) {
  for (const formula of root.querySelectorAll<HTMLElement>('.math-source:not([data-math-rendered])')) {
    const source = formula.textContent || ''
    try {
      katex.render(source, formula, {
        displayMode: formula.dataset.mathDisplay === 'block',
        throwOnError: true,
        trust: false,
        macros: {},
        maxExpand: 1000,
        maxSize: 20,
        output: 'htmlAndMathml',
      })
    } catch {
      formula.textContent = source
      formula.classList.add('math-render-error')
      formula.title = '公式无法解析，已保留原始 TeX。'
      showWarning(root, '部分公式无法解析，已保留原始 TeX。')
    }
    formula.dataset.mathRendered = 'true'
  }
}

export default defineComponent({
  name: 'ContentEnhancements',
  setup() {
    const route = useRoute()
    const { isDark } = useData()
    let active = false
    let running = false
    let pending = false

    async function enhance() {
      pending = true
      if (running || !active) return
      running = true
      try {
        while (pending && active) {
          pending = false
          await nextTick()
          const roots = [...document.querySelectorAll<HTMLElement>('.resource-content')]
          roots.forEach(renderMath)
          const blocks = roots.flatMap(root => [...root.querySelectorAll<HTMLElement>('code.language-mermaid')])
          if (!blocks.length) continue
          let mermaid: Mermaid
          try {
            mermaid = await loadMermaid()
          } catch {
            for (const root of roots) {
              if (root.isConnected && root.querySelector('code.language-mermaid')) {
                showWarning(root, '本地 Mermaid 组件不可用，图表保留为源代码。')
              }
            }
            continue
          }
          if (!active) break
          const theme = isDark.value ? 'dark' : 'default'
          mermaid.initialize({
            startOnLoad: false,
            securityLevel: 'strict',
            theme,
            htmlLabels: false,
            flowchart: { htmlLabels: false },
          })
          for (const code of blocks) {
            if (!active || !code.isConnected || code.dataset.mermaidTheme === theme) continue
            const pre = code.closest('pre')
            if (!pre) continue
            const previous = pre.nextElementSibling
            const oldFigure = previous?.classList.contains('mermaid-diagram') ? previous : null
            const container = document.createElement('div')
            container.className = 'mermaid-render-container'
            container.setAttribute('aria-hidden', 'true')
            document.body.append(container)
            try {
              const result = await mermaid.render(`resource-diagram-${++diagramSequence}`, code.textContent || '', container)
              if (!active || !code.isConnected) continue
              const figure = document.createElement('figure')
              figure.className = 'mermaid-diagram'
              figure.setAttribute('role', 'img')
              figure.setAttribute('aria-label', 'Mermaid 图表')
              figure.innerHTML = result.svg
              oldFigure?.remove()
              pre.after(figure)
              pre.hidden = true
              code.dataset.mermaidTheme = theme
            } catch {
              if (code.isConnected) {
                oldFigure?.remove()
                pre.hidden = false
                code.dataset.mermaidTheme = theme
                showWarning(pre.parentElement!, 'Mermaid 图表无法解析，已保留原始代码。')
              }
            } finally {
              container.remove()
            }
          }
        }
      } finally {
        running = false
      }
    }
    onMounted(() => {
      active = true
      void enhance()
    })
    watch([() => route.path, isDark], () => { if (active) void enhance() }, { flush: 'post' })
    onUnmounted(() => { active = false })
    return () => null
  },
})
