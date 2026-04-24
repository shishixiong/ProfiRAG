<template>
  <div class="workspace">
    <h2 class="workspace-title">PDF 转 Markdown</h2>

    <!-- Upload and Config Panel -->
    <div class="split-view">
      <!-- Upload -->
      <div class="card">
        <div class="card-header">上传与配置</div>

        <div
          class="upload-area"
          @click="triggerUpload"
          @dragover.prevent="dragover = true"
          @dragleave="dragover = false"
          @drop.prevent="handleDrop"
          :class="{ dragover }"
        >
          <p v-if="!uploadedFile">点击或拖拽 PDF 文件到此处</p>
          <p v-else>✅ {{ uploadedFile.name }}</p>
          <input
            ref="fileInput"
            type="file"
            accept=".pdf"
            @change="handleFileSelect"
            hidden
          />
        </div>

        <div class="form-group" style="margin-top: 16px;">
          <label>页码范围 (可选)</label>
          <input v-model="config.pages" placeholder="例如: 1-10, 15, 20-25" />
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="checkbox">
              <input type="checkbox" v-model="config.extract_tables" />
              提取表格
            </label>
          </div>
          <div class="form-group">
            <label class="checkbox">
              <input type="checkbox" v-model="config.write_images" />
              提取图片
            </label>
          </div>
        </div>

        <div class="form-group">
          <label class="checkbox">
            <input type="checkbox" v-model="config.exclude_header_footer" />
            过滤页眉页脚
          </label>
        </div>

        <button
          class="btn btn-primary"
          @click="convert"
          :disabled="!uploadedFile || converting"
        >
          {{ converting ? '转换中...' : '开始转换' }}
        </button>
      </div>

      <!-- Preview -->
      <div class="card">
        <div class="card-header">预览</div>

        <div v-if="!converted" class="preview-panel" style="color: var(--text-secondary);">
          <p>请先上传 PDF 并点击转换</p>
        </div>

        <div v-else class="preview-panel">
          <pre>{{ previewContent }}</pre>
        </div>

        <button
          v-if="converted"
          class="btn btn-success"
          @click="download"
          style="margin-top: 16px;"
        >
          下载 Markdown 文件
        </button>
      </div>
    </div>

    <!-- Tables Section -->
    <div v-if="tables.length > 0" class="card">
      <div class="card-header">提取的表格</div>
      <div class="tables-list">
        <div v-for="table in tables" :key="table" class="table-item">
          {{ table }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { pdfApi } from '../api'

const fileInput = ref(null)
const dragover = ref(false)
const uploadedFile = ref(null)
const fileId = ref(null)
const converting = ref(false)
const converted = ref(false)
const previewContent = ref('')
const tables = ref([])

const config = ref({
  pages: '',
  extract_tables: false,
  write_images: false,
  exclude_header_footer: false,
  header_footer_min_occurrences: 3,
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
  if (file && file.name.endsWith('.pdf')) {
    await uploadFile(file)
  }
}

async function uploadFile(file) {
  uploadedFile.value = file
  try {
    const res = await pdfApi.upload(file)
    fileId.value = res.data.file_id
    converted.value = false
    previewContent.value = ''
    tables.value = []
  } catch (err) {
    alert('上传失败: ' + err.message)
    uploadedFile.value = null
  }
}

async function convert() {
  if (!fileId.value) return

  converting.value = true
  try {
    const res = await pdfApi.convert(fileId.value, {
      pages: config.value.pages || null,
      extract_tables: config.value.extract_tables,
      write_images: config.value.write_images,
      exclude_header_footer: config.value.exclude_header_footer,
      header_footer_min_occurrences: config.value.header_footer_min_occurrences,
    })

    previewContent.value = res.data.markdown_content.slice(0, 5000)
    tables.value = res.data.table_files
    converted.value = true
  } catch (err) {
    alert('转换失败: ' + err.message)
  } finally {
    converting.value = false
  }
}

async function download() {
  if (!fileId.value) return

  try {
    const res = await pdfApi.download(fileId.value)
    const url = window.URL.createObjectURL(new Blob([res.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', uploadedFile.value.name.replace('.pdf', '.md'))
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

.tables-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.table-item {
  background: var(--bg-secondary);
  padding: 8px 16px;
  border-radius: var(--border-radius);
  font-size: 13px;
}
</style>