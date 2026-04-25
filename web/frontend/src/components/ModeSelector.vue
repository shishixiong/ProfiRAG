<template>
  <div class="mode-selector">
    <div class="mode-tabs">
      <button
        v-for="m in modes"
        :key="m.value"
        :class="['mode-tab', { active: selectedMode === m.value }]"
        @click="selectMode(m.value)"
      >
        <span class="mode-label">{{ m.label }}</span>
        <span class="mode-desc">{{ m.description }}</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const modes = [
  { value: 'pipeline', label: '直接问答', description: '快速检索' },
  { value: 'agent', label: 'Agent', description: '智能工具选择' },
  { value: 'plan', label: 'Plan', description: '结构化执行' },
]

const selectedMode = ref('pipeline')
const emit = defineEmits(['change'])

function selectMode(modeValue) {
  selectedMode.value = modeValue
  emit('change', modeValue)
}
</script>

<style scoped>
.mode-selector {
  margin-bottom: 16px;
}

.mode-tabs {
  display: flex;
  gap: 8px;
}

.mode-tab {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 16px;
  border: 1px solid var(--border);
  border-radius: var(--border-radius);
  background: var(--bg-secondary);
  cursor: pointer;
  transition: all 0.2s ease;
}

.mode-tab:hover {
  background: var(--bg-primary);
  border-color: var(--primary);
}

.mode-tab.active {
  background: var(--primary);
  border-color: var(--primary);
  color: white;
}

.mode-tab.active .mode-desc {
  color: rgba(255, 255, 255, 0.85);
}

.mode-label {
  font-size: 14px;
  font-weight: 500;
}

.mode-desc {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 2px;
}
</style>