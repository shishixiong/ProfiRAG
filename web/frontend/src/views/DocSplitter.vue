<template>
  <div class="workspace">
    <h2 class="workspace-title">文档 Splitter</h2>

    <div class="split-view">
      <!-- Config Panel -->
      <div class="card">
        <div class="card-header">配置</div>

        <div
          class="upload-area"
          @click="triggerUpload"
          @dragover.prevent="dragover = true"
          @dragleave="dragover = false"
          @drop.prevent="handleDrop"
          :class="{ dragover }"
        >
          <p v-if="!uploadedFile">点击或拖拽文档到此处</p>
          <p v-else>✅ {{ uploadedFile.name }}</p>
          <input
            ref="fileInput"
            type="file"
            accept=".pdf,.md,.txt,.py,.java,.cpp,.go"
            @change="handleFileSelect"
            hidden
          />
        </div>

        <div class="form-group" style="margin-top: 16px;">
          <label>Splitter 类型</label>
          <select v-model="config.splitter_type">
            <option value="sentence">Sentence</option>
            <option value="token">Token</option>
            <option value="semantic">Semantic</option>
            <option value="chinese">Chinese</option>
            <option value="markdown">Markdown</option>
            <option value="ast">AST (代码)</option>
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
          <select v-model="config.language">
            <option value="python">Python</option>
            <option value="java">Java</option>
            <option value="cpp">C/C++</option>
            <option value="go">Go</option>
          </select>
        </div>

        <button
          class="btn btn-primary"
          @click="preview"
          :disabled="!uploadedFile || loading"
        >
          {{ loading ? '处理中...' : '预览分割' }}
        </button>
      </div>

      <!-- Preview + Metadata -->
      <div class="card" style="flex: 1.5;">
        <div class="card-header">
          分割结果
          <span v-if="totalChunks > 0" style="color: var(--text-secondary);">
            (共 {{ totalChunks }} 个 chunks)
          </span>
        </div>

        <!-- Chunk Tabs -->
        <div v-if="chunks.length > 0" class="chunk-tabs">
          <div
            v-for="chunk in chunks.slice(0, 20)"
            :key="chunk.chunk_index"
            class="chunk-tab"
            :class="{ active: selectedChunk === chunk.chunk_index }"
            @click="selectedChunk = chunk.chunk_index"
          >
            #{{ chunk.chunk_index }}
          </div>
        </div>

        <!-- Chunk Content -->
        <div v-if="selectedChunkData" class="preview-panel">
          <pre>{{ selectedChunkData.text_preview }}</pre>
        </div>

        <!-- Metadata -->
        <div v-if="selectedChunkData" class="card" style="margin-top: 16px; background: var(--bg-secondary);">
          <div class="card-header">Metadata</div>
          <table class="metadata-table">
            <tr>
              <th>属性</th>
              <th>值</th>
            </tr>
            <tr>
              <td>chunk_index</td>
              <td>{{ selectedChunkData.metadata.chunk_index }}</td>
            </tr>
            <tr>
              <td>source_file</td>
              <td>{{ selectedChunkData.metadata.source_file }}</td>
            </tr>
            <tr>
              <td>char_count</td>
              <td>{{ selectedChunkData.metadata.char_count }}</td>
            </tr>
            <tr v-if="selectedChunkData.metadata.header_path">
              <td>header_path</td>
              <td>{{ selectedChunkData.metadata.header_path }}</td>
            </tr>
            <tr v-if="selectedChunkData.metadata.current_heading">
              <td>current_heading</td>
              <td>{{ selectedChunkData.metadata.current_heading }}</td>
            </tr>
            <tr>
              <td>has_code_block</td>
              <td>{{ selectedChunkData.metadata.has_code_block ? '是' : '否' }}</td>
            </tr>
            <tr>
              <td>has_table</td>
              <td>{{ selectedChunkData.metadata.has_table ? '是' : '否' }}</td>
            </tr>
          </table>
        </div>

        <!-- Summary -->
        <div v-if="summary" class="card" style="margin-top: 16px; background: var(--bg-secondary);">
          <div class="card-header">统计</div>
          <div style="font-size: 13px;">
            <p>文档数: {{ summary.documents_loaded }}</p>
            <p>平均 chunks/文档: {{ summary.avg_chunks_per_doc?.toFixed(1) }}</p>
            <p>总字符数: {{ summary.total_chars }}</p>
          </div>
        </div>

        <!-- Download -->
        <button
          v-if="chunks.length > 0"
          class="btn btn-success"
          @click="download"
          style="margin-top: 16px;"
        >
          下载分割结果
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { splitApi } from '../api'

const fileInput = ref(null)
const dragover = ref(false)
const uploadedFile = ref(null)
const fileId = ref(null)
const loading = ref(false)
const chunks = ref([])
const totalChunks = ref(0)
const summary = ref(null)
const selectedChunk = ref(0)

const config = ref({
  splitter_type: 'sentence',
  chunk_size: 512,
  chunk_overlap: 50,
  language: 'python',
})

const selectedChunkData = computed(() => {
  return chunks.value.find(c => c.chunk_index === selectedChunk.value)
})

function triggerUpload() {
  fileInput.value.click()
}

async function handleFileSelect(e) {
  const file = e.target.files[0]
  if (file) {
    await uploadFile(file)
  }
}

async function handleDrop(e) {
  dragover.value = false
  const file = e.dataTransfer.files[0]
  if (file) {
    await uploadFile(file)
  }
}

async function uploadFile(file) {
  uploadedFile.value = file
  try {
    const res = await splitApi.upload(file)
    fileId.value = res.data.file_id
    chunks.value = []
    totalChunks.value = 0
    summary.value = null
  } catch (err) {
    alert('上传失败: ' + err.message)
    uploadedFile.value = null
  }
}

async function preview() {
  if (!fileId.value) return

  loading.value = true
  try {
    const res = await splitApi.preview({
      file_id: fileId.value,
      splitter_type: config.value.splitter_type,
      chunk_size: config.value.chunk_size,
      chunk_overlap: config.value.chunk_overlap,
      language: config.value.language,
    })

    chunks.value = res.data.chunks
    totalChunks.value = res.data.total_chunks
    summary.value = res.data.summary
    selectedChunk.value = chunks.value[0]?.chunk_index || 0
  } catch (err) {
    alert('预览失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

async function download() {
  if (!fileId.value) return

  try {
    const res = await splitApi.download(fileId.value, 'json')
    const url = window.URL.createObjectURL(new Blob([res.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'chunks.json')
    document.body.appendChild(link)
    link.click()
    link.remove()
  } catch (err) {
    alert('下载失败: ' + err.message)
  }
}
</script>

<style scoped>
.workspace-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 24px;
}
</style>