<template>
  <div class="search-container">
    <!-- Header -->
    <h2 class="search-title">文档检索</h2>

    <!-- Query Input Area -->
    <div class="search-input-area">
      <input
        v-model="query"
        type="text"
        placeholder="输入自然语言查询..."
        @keyup.enter="executeSearch"
        :disabled="loading"
      />
      <button class="btn btn-primary" @click="executeSearch" :disabled="loading || !query">
        {{ loading ? '检索中...' : '搜索' }}
      </button>
      <div class="search-config">
        <label style="font-size: 12px;">
          结果数:
          <input type="number" v-model.number="topK" min="1" max="100" style="width: 60px;" />
        </label>
        <label style="font-size: 12px;">
          <input type="checkbox" v-model="rerank" />
          重排序
        </label>
      </div>
    </div>

    <!-- Results Split View -->
    <div class="search-results" v-if="results">
      <!-- Left: File Tree -->
      <aside class="file-tree">
        <div class="file-header">文件列表</div>
        <div
          class="file-item"
          :class="{ active: selectedFile === null }"
          @click="selectedFile = null"
        >
          📁 全部文件 ({{ results.total_results }})
        </div>
        <div
          class="file-item"
          v-for="file in results.files"
          :key="file.filename"
          :class="{ active: selectedFile === file.filename }"
          @click="selectedFile = file.filename"
        >
          📄 {{ file.filename }} ({{ file.chunk_count }})
        </div>
      </aside>

      <!-- Right: Chunk Cards -->
      <div class="chunk-list">
        <div v-if="filteredChunks.length === 0" class="empty-state">
          <p>未找到相关内容</p>
        </div>
        <div
          class="chunk-card"
          v-for="chunk in filteredChunks"
          :key="chunk.chunk_id"
          :class="{ expanded: expandedChunk === chunk.chunk_id }"
          @click="toggleExpand(chunk.chunk_id)"
        >
          <div class="chunk-header">
            <span class="heading">{{ chunk.heading || '无标题' }}</span>
            <span class="score-badge">{{ chunk.score.toFixed(2) }}</span>
          </div>
          <div class="chunk-preview" v-if="expandedChunk !== chunk.chunk_id">
            {{ chunk.text_preview }}
          </div>
          <div class="chunk-full" v-else>
            <div class="chunk-meta">
              <span>文件: {{ chunk.source_file }}</span>
              <span v-if="chunk.header_path">路径: {{ chunk.header_path }}</span>
            </div>
            <div class="chunk-text">{{ chunk.full_text }}</div>
            <button class="btn btn-secondary collapse-btn" @click.stop="expandedChunk = null">
              收起
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Empty State (before search) -->
    <div class="empty-state" v-if="!results && !loading">
      <p>输入查询开始文档检索</p>
      <p style="color: var(--text-secondary); font-size: 13px;">
        仅返回检索结果，不生成回答
      </p>
    </div>

    <!-- Loading State -->
    <div class="loading-state" v-if="loading">
      <p>正在检索相关文档...</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { searchApi } from '../api'

const query = ref('')
const topK = ref(20)
const rerank = ref(true)
const loading = ref(false)
const results = ref(null)
const selectedFile = ref(null)
const expandedChunk = ref(null)

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

<style scoped>
.search-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 48px);
}

.search-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 16px;
}

.search-input-area {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 16px;
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border);
  margin-bottom: 16px;
}

.search-input-area input[type="text"] {
  flex: 1;
  min-width: 300px;
}

.search-config {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-left: auto;
}

.search-results {
  display: flex;
  gap: 16px;
  flex: 1;
  overflow: hidden;
}

.file-tree {
  width: 220px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  padding: 12px;
  overflow-y: auto;
}

.file-header {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
  text-transform: uppercase;
}

.file-item {
  padding: 10px 12px;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 13px;
  transition: background 0.2s;
  margin-bottom: 4px;
}

.file-item:hover {
  background: var(--border);
}

.file-item.active {
  background: var(--primary);
  color: white;
}

.chunk-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 4px;
}

.chunk-card {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--border-radius);
  padding: 16px;
  cursor: pointer;
  transition: border-color 0.2s;
}

.chunk-card:hover {
  border-color: var(--primary);
}

.chunk-card.expanded {
  border: 2px solid var(--primary);
  background: #eff6ff;
}

.chunk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.heading {
  font-weight: 600;
  color: var(--text-primary);
}

.score-badge {
  background: #dcfce7;
  color: #166534;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}

.chunk-preview {
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.5;
}

.chunk-full {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chunk-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: var(--text-secondary);
}

.chunk-text {
  background: var(--bg-primary);
  padding: 12px;
  border-radius: var(--border-radius);
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
}

.collapse-btn {
  align-self: flex-end;
  font-size: 12px;
  padding: 6px 12px;
}

.empty-state, .loading-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: var(--text-secondary);
}

/* Mobile responsive */
@media (max-width: 768px) {
  .search-results {
    flex-direction: column;
  }

  .file-tree {
    width: 100%;
    max-height: 200px;
  }

  .search-input-area input[type="text"] {
    min-width: 100%;
  }

  .search-config {
    width: 100%;
    margin-left: 0;
  }
}
</style>