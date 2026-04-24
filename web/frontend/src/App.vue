<template>
  <div class="app-container">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>ProfiRAG</h1>
      </div>
      <nav class="sidebar-nav">
        <router-link to="/pdf" class="nav-item" active-class="active">
          <span class="icon">📄</span>
          <span class="label">PDF转MD</span>
        </router-link>
        <router-link to="/split" class="nav-item" active-class="active">
          <span class="icon">🔪</span>
          <span class="label">Splitter</span>
        </router-link>
        <router-link to="/import" class="nav-item" active-class="active">
          <span class="icon">📥</span>
          <span class="label">Import</span>
        </router-link>
        <router-link to="/chat" class="nav-item" active-class="active">
          <span class="icon">💬</span>
          <span class="label">Chat</span>
        </router-link>
      </nav>
    </aside>

    <!-- Main Content -->
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup>
</script>

<style>
:root {
  --primary: #3b82f6;
  --primary-dark: #2563eb;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-sidebar: #1e293b;
  --text-primary: #1e293b;
  --text-secondary: #64748b;
  --text-sidebar: #e2e8f0;
  --border: #e2e8f0;
  --border-radius: 8px;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: var(--bg-secondary);
  color: var(--text-primary);
  line-height: 1.5;
}

.app-container {
  display: flex;
  min-height: 100vh;
}

/* Sidebar */
.sidebar {
  width: 200px;
  background: var(--bg-sidebar);
  color: var(--text-sidebar);
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.sidebar-header h1 {
  font-size: 20px;
  font-weight: 600;
}

.sidebar-nav {
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  color: var(--text-sidebar);
  text-decoration: none;
  border-radius: var(--border-radius);
  transition: background 0.2s;
}

.nav-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.nav-item.active {
  background: var(--primary);
  color: white;
}

.nav-item .icon {
  font-size: 18px;
}

.nav-item .label {
  font-size: 14px;
}

/* Main Content */
.main-content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
}

/* Common Components */
.card {
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
  padding: 20px;
  margin-bottom: 16px;
}

.card-header {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: var(--text-primary);
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: background 0.2s;
}

.btn-primary {
  background: var(--primary);
  color: white;
}

.btn-primary:hover {
  background: var(--primary-dark);
}

.btn-success {
  background: var(--success);
  color: white;
}

.btn-secondary {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

input, select, textarea {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--border-radius);
  font-size: 14px;
  width: 100%;
}

input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--primary);
}

label {
  display: block;
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.form-group {
  margin-bottom: 16px;
}

.form-row {
  display: flex;
  gap: 16px;
}

.form-row .form-group {
  flex: 1;
}

/* Upload Area */
.upload-area {
  border: 2px dashed var(--border);
  border-radius: var(--border-radius);
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
}

.upload-area:hover {
  border-color: var(--primary);
}

.upload-area.dragover {
  border-color: var(--primary);
  background: rgba(59, 130, 246, 0.05);
}

/* Preview Panel */
.preview-panel {
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  padding: 16px;
  max-height: 400px;
  overflow-y: auto;
}

.preview-panel pre {
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 13px;
  white-space: pre-wrap;
}

/* Progress Bar */
.progress-bar {
  height: 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar .progress {
  height: 100%;
  background: var(--primary);
  transition: width 0.3s;
}

/* Metadata Table */
.metadata-table {
  width: 100%;
  border-collapse: collapse;
}

.metadata-table th,
.metadata-table td {
  padding: 8px 12px;
  border: 1px solid var(--border);
  text-align: left;
}

.metadata-table th {
  background: var(--bg-secondary);
  font-weight: 500;
}

/* Split View */
.split-view {
  display: flex;
  gap: 16px;
}

.split-view > * {
  flex: 1;
}

/* Checkbox */
.checkbox {
  display: flex;
  align-items: center;
  gap: 8px;
}

.checkbox input[type="checkbox"] {
  width: 18px;
  height: 18px;
}

/* Tabs for chunk navigation */
.chunk-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  overflow-x: auto;
}

.chunk-tab {
  padding: 8px 16px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 13px;
  transition: background 0.2s;
}

.chunk-tab:hover {
  background: var(--border);
}

.chunk-tab.active {
  background: var(--primary);
  color: white;
}

/* Status Badge */
.status-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.status-badge.pending {
  background: var(--warning);
  color: white;
}

.status-badge.running {
  background: var(--primary);
  color: white;
}

.status-badge.completed {
  background: var(--success);
  color: white;
}

.status-badge.failed {
  background: var(--danger);
  color: white;
}
</style>