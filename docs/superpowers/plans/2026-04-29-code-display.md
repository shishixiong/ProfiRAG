# Code Display in SearchView Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add code format display with syntax highlighting, line numbers, and copy button for code files in SearchView retrieval results.

**Architecture:** Create a reusable CodeBlock Vue component that handles syntax highlighting via highlight.js, line number rendering, and copy-to-clipboard functionality. SearchView detects code files by extension and renders with CodeBlock instead of plain text.

**Tech Stack:** Vue 3, highlight.js, marked (existing)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `web/frontend/src/components/CodeBlock.vue` | New component: syntax highlighting, line numbers, copy button |
| `web/frontend/src/views/SearchView.vue` | Import CodeBlock, add file type detection, conditional rendering |
| `web/frontend/package.json` | Add highlight.js dependency |

---

### Task 1: Add highlight.js Dependency

**Files:**
- Modify: `web/frontend/package.json`

- [ ] **Step 1: Add highlight.js to package.json**

Open `web/frontend/package.json` and add highlight.js to dependencies:

```json
{
  "dependencies": {
    "axios": "^1.6.0",
    "highlight.js": "^11.9.0",
    "marked": "^18.0.2",
    "vue": "^3.4.0",
    "vue-router": "^4.2.0"
  }
}
```

- [ ] **Step 2: Install dependency**

Run: `cd web/frontend && npm install`
Expected: highlight.js installed successfully

- [ ] **Step 3: Commit**

```bash
git add web/frontend/package.json web/frontend/package-lock.json
git commit -m "chore: add highlight.js for syntax highlighting"
```

---

### Task 2: Create CodeBlock Component

**Files:**
- Create: `web/frontend/src/components/CodeBlock.vue`

- [ ] **Step 1: Create CodeBlock.vue with basic structure**

Create file `web/frontend/src/components/CodeBlock.vue`:

```vue
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
import hljs from 'highlight.js'

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
```

- [ ] **Step 2: Import highlight.js styles**

Add highlight.js base theme import at the top of the script section:

```vue
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
import html from 'highlight.js/lib/languages/xml'
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
hljs.registerLanguage('html', html)
hljs.registerLanguage('css', css)
hljs.registerLanguage('markdown', markdown)
hljs.registerLanguage('kotlin', kotlin)

// ... rest of the component
```

- [ ] **Step 3: Update CodeBlock.vue with full code**

Replace the entire file with the complete component:

```vue
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
```

- [ ] **Step 4: Commit**

```bash
git add web/frontend/src/components/CodeBlock.vue
git commit -m "feat: add CodeBlock component with syntax highlighting"
```

---

### Task 3: Modify SearchView to Use CodeBlock

**Files:**
- Modify: `web/frontend/src/views/SearchView.vue`

- [ ] **Step 1: Add language detection helper functions**

Add these helper functions to SearchView.vue script section after the existing imports:

```javascript
// File extension to language mapping
const LANGUAGE_MAP = {
  '.py': 'python',
  '.java': 'java',
  '.js': 'javascript',
  '.jsx': 'javascript',
  '.ts': 'typescript',
  '.tsx': 'typescript',
  '.go': 'go',
  '.cpp': 'cpp',
  '.cc': 'cpp',
  '.cxx': 'cpp',
  '.c': 'c',
  '.h': 'c',
  '.hpp': 'cpp',
  '.rs': 'rust',
  '.rb': 'ruby',
  '.php': 'php',
  '.sh': 'bash',
  '.bash': 'bash',
  '.sql': 'sql',
  '.json': 'json',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.xml': 'xml',
  '.html': 'html',
  '.htm': 'html',
  '.css': 'css',
  '.scss': 'css',
  '.sass': 'css',
  '.md': 'markdown',
  '.markdown': 'markdown',
  '.vue': 'javascript',
  '.kt': 'kotlin',
  '.kts': 'kotlin',
}

// Check if file is a code file
function isCodeFile(filename) {
  if (!filename) return false
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  return ext in LANGUAGE_MAP
}

// Get language from filename
function getLanguage(filename) {
  if (!filename) return ''
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  return LANGUAGE_MAP[ext] || ''
}
```

- [ ] **Step 2: Import CodeBlock component**

Add import at the top of script section:

```javascript
import { ref, computed } from 'vue'
import { searchApi } from '../api'
import CodeBlock from '../components/CodeBlock.vue'
```

- [ ] **Step 3: Modify chunk-full template section**

Replace the `chunk-text` div in the expanded chunk section with conditional rendering:

```vue
<div class="chunk-full" v-else>
  <div class="chunk-meta">
    <span>文件: {{ chunk.source_file }}</span>
    <span v-if="chunk.header_path">路径: {{ chunk.header_path }}</span>
  </div>
  <!-- Code file: use CodeBlock -->
  <CodeBlock
    v-if="isCodeFile(chunk.source_file)"
    :code="chunk.full_text"
    :language="getLanguage(chunk.source_file)"
  />
  <!-- Non-code file: plain text -->
  <div class="chunk-text" v-else>{{ chunk.full_text }}</div>
  <button class="btn btn-secondary collapse-btn" @click.stop="expandedChunk = null">
    收起
  </button>
</div>
```

- [ ] **Step 4: Full SearchView.vue modification**

The complete modified script section:

```vue
<script setup>
import { ref, computed } from 'vue'
import { searchApi } from '../api'
import CodeBlock from '../components/CodeBlock.vue'

const query = ref('')
const topK = ref(20)
const rerank = ref(true)
const loading = ref(false)
const results = ref(null)
const selectedFile = ref(null)
const expandedChunk = ref(null)

// File extension to language mapping
const LANGUAGE_MAP = {
  '.py': 'python',
  '.java': 'java',
  '.js': 'javascript',
  '.jsx': 'javascript',
  '.ts': 'typescript',
  '.tsx': 'typescript',
  '.go': 'go',
  '.cpp': 'cpp',
  '.cc': 'cpp',
  '.cxx': 'cpp',
  '.c': 'c',
  '.h': 'c',
  '.hpp': 'cpp',
  '.rs': 'rust',
  '.rb': 'ruby',
  '.php': 'php',
  '.sh': 'bash',
  '.bash': 'bash',
  '.sql': 'sql',
  '.json': 'json',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.xml': 'xml',
  '.html': 'html',
  '.htm': 'html',
  '.css': 'css',
  '.scss': 'css',
  '.sass': 'css',
  '.md': 'markdown',
  '.markdown': 'markdown',
  '.vue': 'javascript',
  '.kt': 'kotlin',
  '.kts': 'kotlin',
}

function isCodeFile(filename) {
  if (!filename) return false
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  return ext in LANGUAGE_MAP
}

function getLanguage(filename) {
  if (!filename) return ''
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  return LANGUAGE_MAP[ext] || ''
}

const filteredChunks = computed(() => {
  if (!results.value) return []
  if (selectedFile.value === null) return results.value.chunks
  return results.value.chunks.filter(c => c.source_file === selectedFile.value)
})

async function executeSearch() {
  if (!query.value.trim() || loading.value) return

  const searchQuery = query.value.trim()
  loading.value = true
  expandedChunk.value = null
  selectedFile.value = null

  try {
    const res = await searchApi.query(searchQuery, topK.value, rerank.value)
    results.value = res.data
  } catch (err) {
    results.value = {
      query: searchQuery,
      total_results: 0,
      files: [],
      chunks: [],
      metadata: { error: err.message }
    }
  } finally {
    loading.value = false
  }
}

function toggleExpand(chunkId) {
  expandedChunk.value = expandedChunk.value === chunkId ? null : chunkId
}
</script>
```

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/views/SearchView.vue
git commit -m "feat: integrate CodeBlock in SearchView for code file display"
```

---

### Task 4: Manual Testing and Verification

- [ ] **Step 1: Start frontend development server**

Run: `cd web/frontend && npm run dev`
Expected: Server starts on port 3000

- [ ] **Step 2: Start backend API server**

Run: `cd web/api && python main.py` (or appropriate command)
Expected: Backend API running on port 8000

- [ ] **Step 3: Test in browser**

1. Open `http://localhost:3000/search`
2. Search for code-related documents (e.g., search "function" or "class")
3. Click on a chunk from a `.py` or `.js` file
4. Verify:
   - Code displays with syntax highlighting
   - Line numbers appear on left side
   - Copy button appears on hover
   - Clicking copy shows "已复制" briefly
   - Non-code files still show plain text

- [ ] **Step 4: Final commit if adjustments needed**

```bash
git add -A
git commit -m "fix: adjust CodeBlock styling based on testing"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add highlight.js dependency |
| 2 | Create CodeBlock.vue component |
| 3 | Modify SearchView to detect code files and use CodeBlock |
| 4 | Manual testing and verification |