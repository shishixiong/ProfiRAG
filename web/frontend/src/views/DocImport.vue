<template>
  <div class="workspace">
    <h2 class="workspace-title">导入文档</h2>

    <!-- Mode Switcher -->
    <div class="mode-switcher">
      <button
        class="mode-btn"
        :class="{ active: importMode === 'file' }"
        @click="importMode = 'file'"
      >
        文件上传
      </button>
      <button
        class="mode-btn"
        :class="{ active: importMode === 'wiki' }"
        @click="importMode = 'wiki'"
      >
        Wiki 导入
      </button>
    </div>

    <div class="split-view">
      <!-- Config Panel -->
      <div class="card" style="flex: 0.6;">
        <div class="card-header">配置</div>

        <div class="form-group">
          <label>Splitter 类型</label>
          <select v-model="config.splitter_type">
            <option value="sentence">Sentence</option>
            <option value="token">Token</option>
            <option value="chinese">Chinese</option>
            <option value="markdown">Markdown</option>
            <option value="ast">AST</option>
          </select>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Chunk Size</label>
            <input type="number" v-model.number="config.chunk_size" min="50" max="4000" />
          </div>
          <div class="form-group">
            <label>Overlap</label>
            <input type="number" v-model.number="config.chunk_overlap" min="0" max="500" />
          </div>
        </div>

        <div v-if="config.splitter_type === 'ast'" class="form-group">
          <label>AST 语言</label>
          <select v-model="config.ast_language">
            <option value="python">Python</option>
            <option value="java">Java</option>
            <option value="cpp">C/C++</option>
            <option value="go">Go</option>
          </select>
        </div>

        <div class="form-group">
          <label>索引模式</label>
          <select v-model="config.index_mode">
            <option value="hybrid">Hybrid (向量 + BM25)</option>
            <option value="vector">Vector (仅向量)</option>
          </select>
        </div>

        <div class="form-group">
          <label>.env 配置文件路径</label>
          <input v-model="config.env_file" placeholder=".env" />
        </div>

        <div class="form-group">
          <label>
            自定义元数据
            <button class="btn-add-meta" @click="addMetadataItem" type="button">+</button>
          </label>
          <div v-if="metadataItems.length > 0" class="metadata-list">
            <div v-for="(item, index) in metadataItems" :key="index" class="metadata-item">
              <input v-model="item.key" placeholder="键" class="meta-key" />
              <input v-model="item.value" placeholder="值" class="meta-value" />
              <button class="btn-remove-meta" @click="removeMetadataItem(index)" type="button">×</button>
            </div>
          </div>
          <p v-if="metadataItems.length === 0" style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
            点击 + 添加元数据条目（可选）
          </p>
        </div>
      </div>

      <!-- Upload / Wiki URL -->
      <div class="card" style="flex: 0.4;">
        <div class="card-header">{{ importMode === 'file' ? '上传文档' : 'Wiki URL' }}</div>

        <!-- File Upload Mode -->
        <div v-if="importMode === 'file'">
          <div
            class="upload-area"
            @click="triggerUpload"
            @dragover.prevent="dragover = true"
            @dragleave="dragover = false"
            @drop.prevent="handleDrop"
            :class="{ dragover }"
          >
            <p>点击或拖拽文件</p>
            <p style="font-size: 12px; color: var(--text-secondary);">支持多文件上传</p>
            <input
              ref="fileInput"
              type="file"
              multiple
              accept=".pdf,.md,.txt,.py,.java,.cpp,.go"
              @change="handleFileSelect"
              hidden
            />
          </div>

          <div v-if="uploadedFiles.length > 0" class="files-list" style="margin-top: 16px;">
            <p style="font-size: 13px; color: var(--text-secondary);">
              已选择 {{ uploadedFiles.length }} 个文件
            </p>
            <ul>
              <li v-for="file in uploadedFiles" :key="file.file_id">
                {{ file.filename }}
              </li>
            </ul>
          </div>
        </div>

        <!-- Wiki URL Mode -->
        <div v-else>
          <div class="form-group">
            <label>Wiki URLs（每行一个）</label>
            <textarea
              v-model="wikiUrlsText"
              placeholder="https://wiki.example.com/pages/123&#10;https://wiki.example.com/pages/456&#10;https://wiki.example.com/pages/789"
              rows="8"
              style="width: 100%; resize: vertical; font-family: monospace; font-size: 13px;"
            />
          </div>
          <div class="form-group" style="margin-top: 12px;">
            <label>Cookie（可选）</label>
            <textarea
              v-model="wikiCookie"
              placeholder="name=value; name2=value2&#10;留空则使用自动刷新 cookie"
              rows="3"
              style="width: 100%; resize: vertical; font-family: monospace; font-size: 12px;"
            />
          </div>
          <p style="font-size: 12px; color: var(--text-secondary); margin-top: 8px;">
            每行输入一个 Wiki URL，将按顺序逐个导入。可配置 Cookie 用于认证，留空则使用默认认证方式。
          </p>

          <!-- URL List with Status -->
          <div v-if="parsedWikiUrls.length > 0" class="wiki-url-list" style="margin-top: 12px;">
            <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;">
              共 {{ parsedWikiUrls.length }} 个 URL
            </p>
            <div v-for="(urlItem, index) in parsedWikiUrls" :key="index" class="wiki-url-item">
              <span class="wiki-url-index">{{ index + 1 }}</span>
              <span class="wiki-url-text" :title="urlItem.url">{{ urlItem.url }}</span>
              <span class="wiki-url-status" :class="urlItem.status">
                {{ getStatusText(urlItem.status) }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Progress & Stats -->
      <div class="card" style="flex: 1;">
        <div class="card-header">进度</div>

        <!-- File mode: no URLs -->
        <div v-if="importMode === 'file' && !importing && !importResult" style="color: var(--text-secondary);">
          <p>请先上传文档并点击开始导入</p>
        </div>

        <!-- Wiki mode: no URLs -->
        <div v-else-if="importMode === 'wiki' && parsedWikiUrls.length === 0 && !importing && !batchImportStatus.started" style="color: var(--text-secondary);">
          <p>请输入 Wiki URL 并点击开始导入</p>
        </div>

        <!-- Batch import summary for wiki mode -->
        <div v-if="importMode === 'wiki' && batchImportStatus.started">
          <div style="font-size: 14px; margin-bottom: 16px;">
            <p><strong>批量导入进度</strong></p>
            <p>已完成: {{ batchImportStatus.completed }} / {{ batchImportStatus.total }} 个 URL</p>
            <p>成功: {{ batchImportStatus.successCount }} | 失败: {{ batchImportStatus.failCount }}</p>
          </div>

          <div v-if="importing || isPaused" style="margin-top: 12px;">
            <div class="progress-bar">
              <div class="progress" :style="{ width: batchProgressPercent + '%' }"></div>
            </div>
            <div style="margin-top: 12px; font-size: 13px;">
              <p v-if="!isPaused">当前 URL: {{ currentImportingUrl }}</p>
              <p v-else style="color: var(--warning);">已暂停: {{ currentImportingUrl }}</p>
              <p>状态: <span class="status-badge" :class="isPaused ? 'paused' : 'running'">{{ isPaused ? 'paused' : progress.status }}</span></p>
              <p>Chunks: {{ progress.chunks_created }}</p>
              <p>耗时: {{ progress.elapsed_seconds.toFixed(1) }} 秒</p>
            </div>

            <!-- Pause/Cancel/Resume buttons -->
            <div class="import-controls" style="margin-top: 12px; display: flex; gap: 8px;">
              <button
                v-if="!isPaused"
                class="btn btn-warning"
                @click="pauseImport"
                type="button"
              >
                暂停
              </button>
              <button
                v-else
                class="btn btn-success"
                @click="resumeImport"
                type="button"
              >
                继续
              </button>
              <button
                class="btn btn-danger"
                @click="cancelImport"
                type="button"
              >
                取消
              </button>
            </div>
          </div>
        </div>

        <!-- Single import progress (file mode or single wiki URL) -->
        <div v-else-if="importing || isPaused">
          <div class="progress-bar">
            <div class="progress" :style="{ width: progressPercent + '%' }"></div>
          </div>
          <div style="margin-top: 12px; font-size: 13px;">
            <p v-if="!isPaused">已处理: {{ progress.documents_processed }} / {{ progress.documents_total }} 文档</p>
            <p v-else style="color: var(--warning);">已暂停: {{ progress.documents_processed }} / {{ progress.documents_total }} 文档</p>
            <p>Chunks: {{ progress.chunks_created }}</p>
            <p>耗时: {{ progress.elapsed_seconds.toFixed(1) }} 秒</p>
            <p>
              状态:
              <span class="status-badge" :class="isPaused ? 'paused' : 'running'">{{ isPaused ? 'paused' : progress.status }}</span>
            </p>
          </div>

          <!-- Pause/Cancel/Resume buttons -->
          <div class="import-controls" style="margin-top: 12px; display: flex; gap: 8px;">
            <button
              v-if="!isPaused"
              class="btn btn-warning"
              @click="pauseImport"
              type="button"
            >
              暂停
            </button>
            <button
              v-else
              class="btn btn-success"
              @click="resumeImport"
              type="button"
            >
              继续
            </button>
            <button
              class="btn btn-danger"
              @click="cancelImport"
              type="button"
            >
              取消
            </button>
          </div>
        </div>

        <!-- Last import result -->
        <div v-if="importResult">
          <div style="font-size: 13px; margin-top: 16px;">
            <p><strong>最近一次导入结果</strong></p>
            <p>
              状态:
              <span class="status-badge completed">completed</span>
            </p>
            <p>文档加载: {{ importResult.documents_loaded }}</p>
            <p>文档导入: {{ importResult.documents_ingested }}</p>
            <p>Chunks 创建: {{ importResult.chunks_created }}</p>
            <p>向量存储: {{ importResult.vector_store_count }} 条</p>
            <p>耗时: {{ importResult.elapsed_seconds.toFixed(2) }} 秒</p>
          </div>
        </div>

        <button
          v-if="((importMode === 'file' && uploadedFiles.length > 0) || (importMode === 'wiki' && parsedWikiUrls.length > 0)) && !importing && !isPaused"
          class="btn btn-success"
          @click="startImport"
          :disabled="importing"
        >
          开始导入
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { importApi } from '../api'

const fileInput = ref(null)
const dragover = ref(false)
const uploadedFiles = ref([])
const importing = ref(false)
const progress = ref({})
const jobId = ref(null)
const importResult = ref(null)
const importMode = ref('file')
const wikiUrlsText = ref('')
const wikiCookie = ref('')

// Batch import tracking for wiki mode
const batchImportStatus = ref({
  started: false,
  completed: 0,
  total: 0,
  successCount: 0,
  failCount: 0,
})
const currentImportingUrl = ref('')

const config = ref({
  splitter_type: 'chinese',
  chunk_size: 512,
  chunk_overlap: 50,
  ast_language: 'python',
  index_mode: 'hybrid',
  env_file: '.env',
})

// Track if import was paused
const isPaused = ref(false)

const metadataItems = ref([])

// Watch for mode changes to update default splitter type
watch(importMode, (newMode) => {
  if (newMode === 'wiki') {
    config.value.splitter_type = 'chinese'
  } else {
    config.value.splitter_type = 'markdown'
  }
})

// Warn user before page refresh if import is in progress
function handleBeforeUnload(e) {
  if (importing.value || batchImportStatus.value.started) {
    e.preventDefault()
    e.returnValue = ''
    return ''
  }
}

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload)
})

onUnmounted(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload)
})

function addMetadataItem() {
  metadataItems.value.push({ key: '', value: '' })
}

function removeMetadataItem(index) {
  metadataItems.value.splice(index, 1)
}

function buildMetadata() {
  const metadata = {}
  for (const item of metadataItems.value) {
    if (item.key && item.key.trim()) {
      metadata[item.key.trim()] = item.value || ''
    }
  }
  return metadata
}

// Parse wiki URLs from textarea (one per line, skip empty lines and comments)
const parsedWikiUrls = computed(() => {
  if (!wikiUrlsText.value) return []
  return wikiUrlsText.value
    .split('\n')
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#'))
    .map(url => ({ url, status: 'pending' }))
})

const batchProgressPercent = computed(() => {
  if (batchImportStatus.value.total === 0) return 0
  return (batchImportStatus.value.completed / batchImportStatus.value.total) * 100
})

const progressPercent = computed(() => {
  if (!progress.value.documents_total) return 0
  return (progress.value.documents_processed / progress.value.documents_total) * 100
})

function getStatusText(status) {
  const map = {
    pending: '等待中',
    importing: '导入中',
    success: '成功',
    failed: '失败',
  }
  return map[status] || status
}

function triggerUpload() {
  fileInput.value.click()
}

async function handleFileSelect(e) {
  const files = Array.from(e.target.files)
  await uploadFiles(files)
}

async function handleDrop(e) {
  dragover.value = false
  const files = Array.from(e.dataTransfer.files)
  await uploadFiles(files)
}

async function uploadFiles(files) {
  if (files.length === 0) return

  try {
    const res = await importApi.upload(files)
    uploadedFiles.value = res.data
    importResult.value = null
  } catch (err) {
    alert('上传失败: ' + err.message)
  }
}

async function startImport() {
  if (importMode.value === 'file' && uploadedFiles.value.length === 0) return
  if (importMode.value === 'wiki' && parsedWikiUrls.value.length === 0) return

  // Reset batch status for wiki mode
  if (importMode.value === 'wiki') {
    batchImportStatus.value = {
      started: true,
      completed: 0,
      total: parsedWikiUrls.value.length,
      successCount: 0,
      failCount: 0,
    }
    // Reset all URL statuses to pending
    for (const urlItem of parsedWikiUrls.value) {
      urlItem.status = 'pending'
    }
  }

  importResult.value = null

  try {
    if (importMode.value === 'wiki') {
      // Batch import URLs sequentially
      await batchImportWikis()
    } else {
      // Single file import
      importing.value = true
      const metadata = buildMetadata()
      const configPayload = { ...config.value, metadata }
      const fileIds = uploadedFiles.value.map(f => f.file_id)
      const res = await importApi.start(fileIds, configPayload)

      jobId.value = res.data.job_id
      progress.value = res.data

      if (res.data.status !== 'completed') {
        await pollProgress()
      }
    }
  } catch (err) {
    alert('导入失败: ' + err.message)
    importing.value = false
  }
}

async function batchImportWikis() {
  const urls = parsedWikiUrls.value

  for (let i = 0; i < urls.length; i++) {
    // Check if import was cancelled
    if (!batchImportStatus.value.started && i > 0) {
      break
    }

    const urlItem = urls[i]

    // Skip already completed URLs
    if (urlItem.status === 'success' || urlItem.status === 'failed') {
      continue
    }

    urlItem.status = 'importing'
    currentImportingUrl.value = urlItem.url
    importing.value = true
    isPaused.value = false

    try {
      const metadata = buildMetadata()
      const configPayload = { ...config.value, metadata }

      const wikiConfig = {
        wiki_url: urlItem.url,
        ...configPayload,
      }

      // Add cookie if configured
      if (wikiCookie.value.trim()) {
        wikiConfig.cookie = wikiCookie.value.trim()
      }

      const res = await importApi.importWiki(wikiConfig)

      jobId.value = res.data.job_id
      progress.value = res.data

      // Poll for this URL's progress (respects pause)
      if (res.data.status !== 'completed') {
        await pollProgress()
      }

      // Only mark as success if not paused/cancelled
      if (!isPaused.value && batchImportStatus.value.started) {
        urlItem.status = 'success'
        batchImportStatus.value.successCount++
      }
    } catch (err) {
      if (!isPaused.value) {
        urlItem.status = 'failed'
        batchImportStatus.value.failCount++
        console.error(`Failed to import ${urlItem.url}:`, err)
      }
    } finally {
      if (!isPaused.value) {
        batchImportStatus.value.completed++
        importing.value = false
      }
    }
  }

  // All URLs processed
  if (!isPaused.value) {
    currentImportingUrl.value = ''
    batchImportStatus.value.started = false
  }
}

// Pause the current import (stop polling, keep job running on server)
function pauseImport() {
  isPaused.value = true
  importing.value = false
}

// Resume a paused import
function resumeImport() {
  isPaused.value = false
  importing.value = true
  if (jobId.value) {
    pollProgress()
  }
}

// Cancel the current import batch
function cancelImport() {
  if (confirm('确定要取消当前导入任务吗？')) {
    isPaused.value = false
    importing.value = false
    batchImportStatus.value.started = false
    jobId.value = null
    progress.value = {}
    currentImportingUrl.value = ''

    // Reset all pending wiki URLs
    if (importMode.value === 'wiki') {
      for (const urlItem of parsedWikiUrls.value) {
        if (urlItem.status === 'importing' || urlItem.status === 'pending') {
          urlItem.status = 'pending'
        }
      }
    }
  }
}

async function pollProgress() {
  while (importing.value && jobId.value && !isPaused.value) {
    await new Promise(resolve => setTimeout(resolve, 2000))

    try {
      const res = await importApi.progress(jobId.value)
      progress.value = res.data

      if (res.data.status === 'completed') {
        importing.value = false
        const statsRes = await importApi.stats(jobId.value)
        importResult.value = statsRes.data
        break
      } else if (res.data.status === 'failed') {
        importing.value = false
        alert('导入失败: ' + res.data.error)
        break
      }
    } catch (err) {
      console.error('Poll error:', err)
    }
  }
}
</script>

<style scoped>
.workspace-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 24px;
}

.mode-switcher {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.mode-btn {
  padding: 8px 16px;
  border: 1px solid var(--border);
  background: var(--bg-secondary);
  color: var(--text-secondary);
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.mode-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}

.mode-btn.active {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

.files-list ul {
  list-style: none;
  font-size: 13px;
}

.files-list li {
  padding: 4px 0;
  color: var(--text-secondary);
}

.btn-add-meta {
  background: var(--primary);
  color: white;
  border: none;
  border-radius: 4px;
  width: 24px;
  height: 24px;
  font-size: 16px;
  cursor: pointer;
  margin-left: 8px;
  line-height: 1;
}

.btn-add-meta:hover {
  opacity: 0.9;
}

.metadata-list {
  margin-top: 8px;
}

.metadata-item {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  align-items: center;
}

.meta-key {
  flex: 1;
  min-width: 80px;
}

.meta-value {
  flex: 2;
  min-width: 120px;
}

.btn-remove-meta {
  background: var(--bg-secondary);
  color: var(--text-secondary);
  border: 1px solid var(--border);
  border-radius: 4px;
  width: 24px;
  height: 24px;
  font-size: 14px;
  cursor: pointer;
  line-height: 1;
}

.btn-remove-meta:hover {
  background: #fee;
  color: #c00;
}

.wiki-url-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px;
  background: var(--bg-secondary);
}

.wiki-url-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  font-size: 12px;
  border-bottom: 1px solid var(--border);
}

.wiki-url-item:last-child {
  border-bottom: none;
}

.wiki-url-index {
  color: var(--text-secondary);
  min-width: 20px;
  text-align: right;
}

.wiki-url-text {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-primary);
}

.wiki-url-status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  min-width: 50px;
  text-align: center;
}

.wiki-url-status.pending {
  background: var(--bg-secondary);
  color: var(--text-secondary);
}

.wiki-url-status.importing {
  background: #fff3cd;
  color: #856404;
}

.wiki-url-status.success {
  background: #d4edda;
  color: #155724;
}

.wiki-url-status.failed {
  background: #f8d7da;
  color: #721c24;
}

.status-badge.paused {
  background: #fff3cd;
  color: #856404;
}

.btn-warning {
  background: #ffc107;
  color: #212529;
}

.btn-warning:hover {
  background: #e0a800;
}

.btn-danger {
  background: #dc3545;
  color: white;
}

.btn-danger:hover {
  background: #c82333;
}
</style>