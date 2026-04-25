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
          <label>自定义元数据 (JSON格式)</label>
          <textarea
            v-model="config.metadata"
            placeholder='{"category": "docs", "version": "1.0"}'
            rows="3"
            style="width: 100%; resize: vertical;"
          ></textarea>
          <p style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
            可选，导入的文档将携带这些元数据
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
  metadata: '',
})

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

    // Parse metadata if provided
    let metadata = {}
    if (config.value.metadata && config.value.metadata.trim()) {
      try {
        metadata = JSON.parse(config.value.metadata)
      } catch (e) {
        alert('元数据格式错误，请输入有效的 JSON 格式')
        importing.value = false
        return
      }
    }

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
</style>