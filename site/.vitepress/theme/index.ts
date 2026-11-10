import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import ResourceCatalog from './ResourceCatalog.vue'
import ContentEnhancements from './ContentEnhancements'
import 'katex/dist/katex.min.css'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => h(DefaultTheme.Layout, null, {
    'layout-bottom': () => h(ContentEnhancements),
  }),
  enhanceApp({ app, router }) {
    app.component('ResourceCatalog', ResourceCatalog)
    router.onAfterRouteChange = () => {
      if (typeof window !== 'undefined') window.dispatchEvent(new Event('catalog:navigate'))
    }
  },
} satisfies Theme
