# Chat Knowledge Q&A Mode Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a mode selector to the Chat page allowing users to choose between Pipeline, Agent, and Plan query modes.

**Architecture:** Backend routes by mode to different query methods; frontend adds ModeSelector component with tab-style UI; mode passed through API layer.

**Tech Stack:** FastAPI (backend), Vue 3 (frontend), Pydantic (schemas)

---

## File Structure

**Backend:**
- `web/api/schemas.py` - Add ChatMode enum, update ChatRequest
- `web/api/services.py` - Update ChatService.query to route by mode, add _format_agent_response
- `web/api/routes/chat.py` - Pass mode parameter to service

**Frontend:**
- `web/frontend/src/components/ModeSelector.vue` - New component (create)
- `web/frontend/src/views/ChatView.vue` - Import ModeSelector, add mode state, pass to API
- `web/frontend/src/api/index.js` - Update chatApi.query to accept mode

---

### Task 1: Backend Schema - Add ChatMode Enum and Update ChatRequest

**Files:**
- Modify: `web/api/schemas.py:154-159`

- [ ] **Step 1: Add ChatMode enum to schemas.py**

Add the enum after the existing OutputFormat enum (around line 33):

```python
class ChatMode(str, Enum):
    PIPELINE = "pipeline"
    AGENT = "agent"
    PLAN = "plan"
```

- [ ] **Step 2: Update ChatRequest to include mode field**

Modify the ChatRequest class (around line 155):

```python
class ChatRequest(BaseModel):
    """Request for RAG chat query."""
    query: str = Field(..., description="User question")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to retrieve")
    mode: ChatMode = Field(ChatMode.PIPELINE, description="Query mode")
    env_file: str = Field(".env", description="Path to .env config file")
```

- [ ] **Step 3: Verify schemas are valid**

Run a quick Python check:

```bash
cd /Users/mac/graphrag/ProfiRAG && python -c "from web.api.schemas import ChatMode, ChatRequest; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit backend schema changes**

```bash
git add web/api/schemas.py
git commit -m "$(cat <<'EOF'
feat: add ChatMode enum and mode field to ChatRequest

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Backend Service - Update ChatService.query to Route by Mode

**Files:**
- Modify: `web/api/services.py:446-492`

- [ ] **Step 1: Update ChatService.query method signature and implementation**

Replace the existing `query` method in ChatService (lines 449-492):

```python
@staticmethod
def query(
    query_str: str,
    top_k: int = 10,
    mode: str = "pipeline",
    env_file: str = ".env",
) -> Dict[str, Any]:
    """Execute RAG query and return response with images.

    Args:
        query_str: User query string
        top_k: Number of results to retrieve
        mode: Query mode (pipeline, agent, plan)
        env_file: Path to environment config file

    Returns:
        Dictionary with response, sources, and images
    """
    # Load configuration
    config_path = PROJECT_ROOT / env_file
    config = load_config(str(config_path))

    # Initialize pipeline
    pipeline = RAGPipeline(config)

    # Route based on mode
    if mode == "plan":
        result = pipeline.query_with_agent(query_str, mode="plan", auto_approve=True)
        return ChatService._format_agent_response(result)
    elif mode == "agent":
        result = pipeline.query_with_agent(query_str, mode="agent")
        return ChatService._format_agent_response(result)
    else:
        # Pipeline mode - default behavior
        return pipeline.query_with_images(query_str, top_k=top_k, include_images=True)
```

- [ ] **Step 2: Add _format_agent_response helper method**

Add this method at the end of ChatService class (after query method):

```python
@staticmethod
def _format_agent_response(result: Dict) -> Dict:
    """Normalize agent response to ChatResponse format.

    Args:
        result: Agent query result dictionary

    Returns:
        Normalized dictionary with response, source_nodes, images, metadata
    """
    # Extract response text
    response = result.get("response", "")

    # Extract sources from agent result
    sources = []
    if "sources" in result:
        for src in result["sources"]:
            sources.append({
                "node_id": src.get("node_id", ""),
                "text": src.get("text", "")[:300] if len(src.get("text", "")) > 300 else src.get("text", ""),
                "score": src.get("score", 0.0),
                "source_file": src.get("source_file"),
                "header_path": src.get("header_path"),
            })

    # Also check source_nodes key (alternative format)
    if "source_nodes" in result and not sources:
        for src in result["source_nodes"]:
            sources.append({
                "node_id": src.get("node_id", ""),
                "text": src.get("text", "")[:300] if len(src.get("text", "")) > 300 else src.get("text", ""),
                "score": src.get("score", 0.0),
                "source_file": src.get("source_file"),
                "header_path": src.get("header_path"),
            })

    return {
        "query": result.get("question", result.get("query", "")),
        "response": response,
        "source_nodes": sources,
        "images": [],  # Agent mode doesn't return images currently
        "metadata": {"mode": result.get("mode", "agent")},
    }
```

- [ ] **Step 3: Verify service imports are correct**

Ensure RAGPipeline is imported (should already be there at line 32):

```python
from profirag.pipeline.rag_pipeline import RAGPipeline
```

- [ ] **Step 4: Verify Python syntax is valid**

```bash
cd /Users/mac/graphrag/ProfiRAG && python -c "from web.api.services import ChatService; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit backend service changes**

```bash
git add web/api/services.py
git commit -m "$(cat <<'EOF'
feat: update ChatService.query to route by mode (pipeline/agent/plan)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Backend Route - Pass Mode Parameter to Service

**Files:**
- Modify: `web/api/routes/chat.py:11-22`

- [ ] **Step 1: Update the query route to pass mode**

Replace the existing route handler (lines 11-22):

```python
@router.post("/query", response_model=schemas.ChatResponse, summary="Execute RAG query")
async def query(request: schemas.ChatRequest):
    """Execute RAG query and return response with source references."""
    try:
        result = services.ChatService.query(
            query_str=request.query,
            top_k=request.top_k,
            mode=request.mode.value,  # Convert enum to string
            env_file=request.env_file,
        )
        return schemas.ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 2: Verify route syntax is valid**

```bash
cd /Users/mac/graphrag/ProfiRAG && python -c "from web.api.routes.chat import router; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit backend route changes**

```bash
git add web/api/routes/chat.py
git commit -m "$(cat <<'EOF'
feat: pass mode parameter from ChatRequest to ChatService

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Frontend - Create ModeSelector Component

**Files:**
- Create: `web/frontend/src/components/ModeSelector.vue`

- [ ] **Step 1: Create components directory if needed**

```bash
mkdir -p /Users/mac/graphrag/ProfiRAG/web/frontend/src/components
```

- [ ] **Step 2: Create ModeSelector.vue with template, script, and styles**

```vue
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
```

- [ ] **Step 3: Commit ModeSelector component**

```bash
git add web/frontend/src/components/ModeSelector.vue
git commit -m "$(cat <<'EOF'
feat: create ModeSelector component for chat query modes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Frontend - Update chatApi to Accept Mode Parameter

**Files:**
- Modify: `web/frontend/src/api/index.js:85-90`

- [ ] **Step 1: Update chatApi.query to accept mode parameter**

Replace the chatApi section (lines 85-90):

```js
// Chat endpoints
export const chatApi = {
  query: async (query, topK = 10, mode = 'pipeline') => {
    return api.post('/chat/query', { query, top_k: topK, mode })
  },
}
```

- [ ] **Step 2: Commit API changes**

```bash
git add web/frontend/src/api/index.js
git commit -m "$(cat <<'EOF'
feat: add mode parameter to chatApi.query

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Frontend - Integrate ModeSelector into ChatView

**Files:**
- Modify: `web/frontend/src/views/ChatView.vue`

- [ ] **Step 1: Import ModeSelector component**

Add import at line 87 (after existing imports):

```js
import { ref, nextTick } from 'vue'
import { marked } from 'marked'
import { chatApi } from '../api'
import ModeSelector from '../components/ModeSelector.vue'
```

- [ ] **Step 2: Add selectedMode state and handler**

Add after existing refs (around line 94):

```js
const messages = ref([])
const query = ref('')
const loading = ref(false)
const topK = ref(10)
const selectedMode = ref('pipeline')
const messagesContainer = ref(null)
```

Add handler function after existing functions (around line 167):

```js
function onModeChange(modeValue) {
  selectedMode.value = modeValue
}
```

- [ ] **Step 3: Add ModeSelector component to template**

Add after the title (around line 5, after `<h2 class="workspace-title">知识问答</h2>`):

```vue
    <ModeSelector @change="onModeChange" />
```

- [ ] **Step 4: Update sendQuery to pass mode**

Modify the chatApi.query call in sendQuery function (line 135):

```js
const res = await chatApi.query(userQuery, topK.value, selectedMode.value)
```

- [ ] **Step 5: Verify frontend builds without errors**

```bash
cd /Users/mac/graphrag/ProfiRAG/web/frontend && npm run build 2>&1 | head -20
```

Expected: Build completes without errors

- [ ] **Step 6: Commit ChatView integration**

```bash
git add web/frontend/src/views/ChatView.vue
git commit -m "$(cat <<'EOF'
feat: integrate ModeSelector into ChatView with mode state

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Manual Testing - Verify All Modes Work

**Files:**
- None (testing task)

- [ ] **Step 1: Start backend server**

```bash
cd /Users/mac/graphrag/ProfiRAG/web/api && python main.py
```

Expected: Server starts on port 8000

- [ ] **Step 2: Start frontend dev server**

```bash
cd /Users/mac/graphrag/ProfiRAG/web/frontend && npm run dev
```

Expected: Dev server starts on port 5173

- [ ] **Step 3: Test Pipeline mode**

1. Open http://localhost:5173 in browser
2. Navigate to Chat page
3. Ensure "直接问答" tab is active by default
4. Enter a query and verify response comes back with sources and images

- [ ] **Step 4: Test Agent mode**

1. Click "Agent" tab to switch mode
2. Enter a query
3. Verify response comes back (may take longer due to agent iteration)
4. Check that mode indicator shows "agent"

- [ ] **Step 5: Test Plan mode**

1. Click "Plan" tab to switch mode
2. Enter a query
3. Verify response comes back (auto-approved plan execution)
4. Check that mode indicator shows "plan"

- [ ] **Step 6: Create final commit with all changes**

```bash
git status
```

Verify all changes are committed. If any uncommitted:

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: complete chat mode selection feature (pipeline/agent/plan)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan implements the chat mode selection feature with:
- 3 backend tasks (schemas, services, routes)
- 3 frontend tasks (ModeSelector component, API update, ChatView integration)
- 1 testing task

All tasks follow TDD principles where applicable, with frequent commits after each logical change.