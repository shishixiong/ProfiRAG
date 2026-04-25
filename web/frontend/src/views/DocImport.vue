<template>
  <div class="workspace">
    <h2 class="workspace-title">导入文档</h2>

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

      <!-- Upload -->
      <div class="card" style="flex: 0.4;">
        <div class="card-header">上传文档</div>

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

      <!-- Progress & Stats -->
      <div class="card" style="flex: 1;">
        <div class="card-header">进度</div>

        <div v-if="!importing && !importResult" style="color: var(--text-secondary);">
          <p>请先上传文档并点击开始导入</p>
        </div>

        <div v-if="importing">
          <div class="progress-bar">
            <div class="progress" :style="{ width: progressPercent + '%' }"></div>
          </div>
          <div style="margin-top: 12px; font-size: 13px;">
            <p>已处理: {{ progress.documents_processed }} / {{ progress.documents_total }} 文档</p>
            <p>Chunks: {{ progress.chunks_created }}</p>
            <p>耗时: {{ progress.elapsed_seconds.toFixed(1) }} 秒</p>
            <p>
              状态:
              <span class="status-badge running">{{ progress.status }}</span>
            </p>
          </div>
        </div>

        <div v-if="importResult">
          <div style="font-size: 13px;">
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
          v-if="uploadedFiles.length > 0 && !importing"
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
import { ref, computed } from 'vue'
import { importApi } from '../api'

const fileInput = ref(null)
const dragover = ref(false)
const uploadedFiles = ref([])
const importing = ref(false)
const progress = ref({})
const jobId = ref(null)
const importResult = ref(null)

const config = ref({
  splitter_type: 'markdown',
  chunk_size: 1024,
  chunk_overlap: 100,
  ast_language: 'python',
  index_mode: 'hybrid',
  env_file: '.env',
})

const metadataItems = ref([])

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

const progressPercent = computed(() => {
  if (!progress.value.documents_total) return 0
  return (progress.value.documents_processed / progress.value.documents_total) * 100
})

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
  if (uploadedFiles.value.length === 0) return

  importing.value = true
  importResult.value = null

  try {
    const fileIds = uploadedFiles.value.map(f => f.file_id)

    // Build metadata from KV items
    const metadata = buildMetadata()

    const configPayload = {
      ...config.value,
      metadata,
    }

    const res = await importApi.start(fileIds, configPayload)

    jobId.value = res.data.job_id
    progress.value = res.data

    // Poll for progress
    if (res.data.status !== 'completed') {
      await pollProgress()
    }
  } catch (err) {
    alert('导入失败: ' + err.message)
    importing.value = false
  }
}

async function pollProgress() {
  while (importing.value && jobId.value) {
    await new Promise(resolve => setTimeout(resolve, 1000))

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
</style>