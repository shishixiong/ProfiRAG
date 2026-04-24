<template>
  <div class="workspace">
    <h2 class="workspace-title">知识问答</h2>

    <div class="chat-container">
      <!-- Messages -->
      <div class="messages" ref="messagesContainer">
        <div v-if="messages.length === 0" class="empty-state">
          <p>请输入问题，开始专业知识问答</p>
          <p style="color: var(--text-secondary); font-size: 13px;">
            系统会基于已导入的文档进行检索和回答，支持表格和图片显示
          </p>
        </div>

        <div v-for="(msg, index) in messages" :key="index" class="message" :class="msg.role">
          <div class="message-header">
            <span class="role-label">{{ msg.role === 'user' ? '👤 问题' : '🤖 回答' }}</span>
          </div>
          <div class="message-content">
            <p v-if="msg.role === 'user'">{{ msg.content }}</p>
            <div v-else class="response-content">
              <!-- Markdown rendered content (supports tables) -->
              <div class="markdown-body" v-html="renderMarkdown(msg.content)"></div>

              <!-- Images -->
              <div v-if="msg.images && msg.images.length > 0" class="images-section">
                <p class="images-label">🖼️ 相关图片:</p>
                <div class="images-list">
                  <div v-for="(img, iIdx) in msg.images" :key="iIdx" class="image-item">
                    <img v-if="img.path" :src="getImageUrl(img.path)" :alt="img.description" class="chat-image" />
                    <p class="image-desc">{{ img.description }}</p>
                  </div>
                </div>
              </div>

              <!-- Sources -->
              <div v-if="msg.sources && msg.sources.length > 0" class="sources-section">
                <p class="sources-label">📚 参考来源:</p>
                <div class="sources-list">
                  <div v-for="(source, sIdx) in msg.sources" :key="sIdx" class="source-item">
                    <span class="source-file">{{ source.source_file || '未知来源' }}</span>
                    <span v-if="source.header_path" class="source-path">{{ source.header_path }}</span>
                    <span class="source-score">相关性: {{ source.score?.toFixed(2) || 'N/A' }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div v-if="loading" class="message assistant loading">
          <div class="message-content">
            <p>正在检索和生成回答...</p>
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="input-area">
        <input
          v-model="query"
          type="text"
          placeholder="输入您的问题..."
          @keyup.enter="sendQuery"
          :disabled="loading"
        />
        <button class="btn btn-primary" @click="sendQuery" :disabled="loading || !query">
          {{ loading ? '查询中...' : '发送' }}
        </button>
        <button class="btn btn-secondary" @click="clearHistory" :disabled="messages.length === 0">
          清空历史
        </button>
      </div>

      <!-- Config -->
      <div class="config-area">
        <label style="font-size: 12px;">
          返回结果数:
          <input type="number" v-model.number="topK" min="1" max="50" style="width: 60px;" />
        </label>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { marked } from 'marked'
import { chatApi } from '../api'

const messages = ref([])
const query = ref('')
const loading = ref(false)
const topK = ref(10)
const messagesContainer = ref(null)

// Configure marked for better rendering
marked.setOptions({
  breaks: true,
  gfm: true,
})

function renderMarkdown(content) {
  if (!content) return ''
  return marked.parse(content)
}

function getImageUrl(path) {
  // Convert local file path to URL if needed
  if (path.startsWith('/')) {
    return path
  }
  return path
}

async function sendQuery() {
  if (!query.value.trim() || loading.value) return

  const userQuery = query.value.trim()

  // Add user message
  messages.value.push({
    role: 'user',
    content: userQuery,
  })

  query.value = ''
  loading.value = true

  // Scroll to bottom
  await nextTick()
  scrollToBottom()

  try {
    const res = await chatApi.query(userQuery, topK.value)

    // Add assistant message with sources and images
    messages.value.push({
      role: 'assistant',
      content: res.data.response,
      sources: res.data.source_nodes,
      images: res.data.images,
    })

    await nextTick()
    scrollToBottom()
  } catch (err) {
    messages.value.push({
      role: 'assistant',
      content: '查询失败: ' + err.message,
      sources: [],
      images: [],
    })
  } finally {
    loading.value = false
  }
}

function clearHistory() {
  messages.value = []
}

function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}
</script>

<style scoped>
.workspace-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 24px;
}

.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 100px);
  max-height: 800px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  margin-bottom: 16px;
}

.empty-state {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
}

.message {
  margin-bottom: 16px;
  padding: 12px 16px;
  border-radius: var(--border-radius);
}

.message.user {
  background: var(--primary);
  color: white;
  margin-left: 20%;
}

.message.assistant {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  margin-right: 20%;
}

.message.loading {
  opacity: 0.7;
}

.message-header {
  margin-bottom: 8px;
}

.role-label {
  font-size: 12px;
  font-weight: 500;
}

.message-content p {
  margin: 0;
  line-height: 1.6;
}

.response-content {
  white-space: normal;
}

/* Markdown styling for tables */
.markdown-body {
  line-height: 1.6;
}

.markdown-body table {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0;
}

.markdown-body th,
.markdown-body td {
  border: 1px solid var(--border);
  padding: 8px 12px;
  text-align: left;
}

.markdown-body th {
  background: var(--bg-secondary);
  font-weight: 600;
}

.markdown-body tr:nth-child(even) {
  background: var(--bg-secondary);
}

.markdown-body code {
  background: var(--bg-secondary);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.markdown-body pre {
  background: var(--bg-secondary);
  padding: 12px;
  border-radius: var(--border-radius);
  overflow-x: auto;
}

.markdown-body pre code {
  background: none;
  padding: 0;
}

/* Images section */
.images-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}

.images-label {
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 8px;
}

.images-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.image-item {
  max-width: 300px;
}

.chat-image {
  max-width: 100%;
  border-radius: var(--border-radius);
  border: 1px solid var(--border);
}

.image-desc {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.sources-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}

.sources-label {
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 8px;
}

.sources-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.source-item {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 12px;
  padding: 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
}

.source-file {
  font-weight: 500;
  color: var(--text-primary);
}

.source-path {
  color: var(--text-secondary);
}

.source-score {
  color: var(--success);
}

.input-area {
  display: flex;
  gap: 8px;
  padding: 16px;
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border);
}

.input-area input {
  flex: 1;
}

.config-area {
  margin-top: 8px;
  text-align: right;
  color: var(--text-secondary);
}
</style>