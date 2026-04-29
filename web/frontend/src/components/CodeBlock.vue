<template>
  <div class="code-block" ref="codeBlockRef">
    <button
      class="copy-btn"
      :class="{ copied: justCopied }"
      @click="copyCode"
      title="复制代码"
    >
      {{ justCopied ? '已复制' : '复制' }}
    </button>
    <div class="code-wrapper">
      <div class="line-numbers">
        <span v-for="line in lineCount" :key="line">{{ line }}</span>
      </div>
      <pre class="code-content"><code ref="codeRef"></code></pre>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import hljs from 'highlight.js/lib/core'
// Import common languages
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import java from 'highlight.js/lib/languages/java'
import typescript from 'highlight.js/lib/languages/typescript'
import go from 'highlight.js/lib/languages/go'
import cpp from 'highlight.js/lib/languages/cpp'
import c from 'highlight.js/lib/languages/c'
import rust from 'highlight.js/lib/languages/rust'
import ruby from 'highlight.js/lib/languages/ruby'
import php from 'highlight.js/lib/languages/php'
import bash from 'highlight.js/lib/languages/bash'
import sql from 'highlight.js/lib/languages/sql'
import json from 'highlight.js/lib/languages/json'
import yaml from 'highlight.js/lib/languages/yaml'
import xml from 'highlight.js/lib/languages/xml'
import css from 'highlight.js/lib/languages/css'
import markdown from 'highlight.js/lib/languages/markdown'
import kotlin from 'highlight.js/lib/languages/kotlin'

// Register languages
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('python', python)
hljs.registerLanguage('java', java)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('go', go)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('c', c)
hljs.registerLanguage('rust', rust)
hljs.registerLanguage('ruby', ruby)
hljs.registerLanguage('php', php)
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('sql', sql)
hljs.registerLanguage('json', json)
hljs.registerLanguage('yaml', yaml)
hljs.registerLanguage('xml', xml)
hljs.registerLanguage('html', xml)
hljs.registerLanguage('css', css)
hljs.registerLanguage('markdown', markdown)
hljs.registerLanguage('kotlin', kotlin)

const props = defineProps({
  code: {
    type: String,
    required: true
  },
  language: {
    type: String,
    default: ''
  }
})

const codeRef = ref(null)
const codeBlockRef = ref(null)
const justCopied = ref(false)

const lineCount = computed(() => {
  if (!props.code) return 0
  return props.code.split('\n').length
})

function highlightCode() {
  if (!codeRef.value) return
  codeRef.value.textContent = props.code
  if (props.language && hljs.getLanguage(props.language)) {
    hljs.highlightElement(codeRef.value, { language: props.language })
  } else {
    hljs.highlightElement(codeRef.value)
  }
}

async function copyCode() {
  try {
    await navigator.clipboard.writeText(props.code)
    justCopied.value = true
    setTimeout(() => {
      justCopied.value = false
    }, 2000)
  } catch (err) {
    console.error('Copy failed:', err)
  }
}

onMounted(highlightCode)

watch(() => props.code, highlightCode)
</script>

<style scoped>
.code-block {
  background: #1e1e1e;
  border-radius: 8px;
  position: relative;
  overflow: hidden;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
}

.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 12px;
  background: #333;
  color: #ccc;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  opacity: 0;
  transition: opacity 0.2s, background 0.2s;
}

.code-block:hover .copy-btn {
  opacity: 1;
}

.copy-btn:hover {
  background: #444;
}

.copy-btn.copied {
  background: #22c55e;
  color: white;
  opacity: 1;
}

.code-wrapper {
  display: flex;
  overflow-x: auto;
}

.line-numbers {
  background: #252526;
  color: #858585;
  padding: 12px 12px 12px 8px;
  text-align: right;
  user-select: none;
  border-right: 1px solid #333;
  display: flex;
  flex-direction: column;
  min-width: 40px;
}

.line-numbers span {
  line-height: 1.5;
  font-size: 14px;
}

.code-content {
  margin: 0;
  padding: 12px;
  background: transparent;
  overflow-x: auto;
  flex: 1;
}

.code-content code {
  background: transparent;
  font-size: 14px;
  line-height: 1.5;
  color: #d4d4d4;
}
</style>