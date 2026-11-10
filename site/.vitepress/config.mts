import { defineConfig } from 'vitepress';
import { readFileSync } from 'node:fs';

const { base } = JSON.parse(readFileSync(new URL('../.generated/build-config.json', import.meta.url), 'utf8'));

export default defineConfig({
  lang: 'zh-CN',
  title: '端侧 LLM 资源库',
  description: '端侧大模型推理、量化与软硬件协同的论文、技术资源和阅读笔记。',
  base,
  srcDir: '.generated',
  cleanUrls: false,
  // Exported HTML already points at base-prefixed files in the public directory.
  vue: { template: { transformAssetUrls: { includeAbsolute: false } } },
  lastUpdated: false,
  appearance: true,
  themeConfig: {
    nav: [
      { text: '资源检索', link: '/' },
      { text: '阅读笔记', link: '/?notes=1' },
      { text: 'GitHub', link: 'https://github.com/iicesprite/awesome-on-decive-LLM' },
    ],
    outline: { label: '本页内容', level: [2, 3] },
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
    darkModeSwitchLabel: '切换外观',
    docFooter: { prev: '上一篇', next: '下一篇' },
    footer: { message: '从仓库元数据与阅读笔记生成 · 只读展示，不提供 PDF 下载' },
    search: {
      provider: 'local',
      options: {
        miniSearch: {
          options: {
            tokenize: (text: string) => Array.from(
              new Intl.Segmenter('zh', { granularity: 'word' }).segment(text),
            ).filter(part => part.isWordLike).map(part => part.segment),
          },
        },
        translations: {
          button: { buttonText: '全文搜索', buttonAriaLabel: '搜索资源与笔记正文' },
          modal: {
            noResultsText: '没有找到结果', resetButtonTitle: '清空搜索',
            displayDetails: '显示详细结果',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
  },
});
