# Chat Knowledge Q&A Mode Selection Design

**Date:** 2026-04-25
**Status:** Approved

## Overview

Add a mode selector to the Chat page allowing users to choose between three query modes: Pipeline (direct Q&A), Agent (ReAct), and Plan (structured execution).

## Requirements

- User can select query mode before sending a question
- Three modes available: Pipeline, Agent, Plan
- Default mode: Pipeline (fastest, simplest)
- Mode labels include brief description for clarity
- Mode persists during session, resets on page reload

## Architecture

```
Frontend (ModeSelector) → ChatView → API → ChatService → RAGPipeline.query_with_agent(mode)
```

### Integration Points

1. **Frontend**: `ModeSelector.vue` component in `ChatView.vue`
2. **API layer**: `chatApi.query` accepts mode parameter
3. **Backend**: `ChatRequest` schema with mode field, `ChatService.query` routes by mode
4. **Pipeline**: Existing `query_with_agent` method handles mode routing

## Frontend Design

### New Component: ModeSelector.vue

Location: `web/frontend/src/components/ModeSelector.vue`

```vue
<template>
  <div class="mode-selector">
    <div class="mode-tabs">
      <button
        v-for="mode in modes"
        :key="mode.value"
        :class="['mode-tab', { active: currentMode === mode.value }]"
        @click="selectMode(mode.value)"
      >
        <span class="mode-label">{{ mode.label }}</span>
        <span class="mode-desc">{{ mode.description }}</span>
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

const currentMode = ref('pipeline')
const emit = defineEmits(['change'])

function selectMode(modeValue) {
  currentMode.value = modeValue
  emit('change', modeValue)
}
</script>
```

### ChatView.vue Changes

1. Import `ModeSelector` component
2. Add `<ModeSelector @change="onModeChange" />` after title, before messages container
3. Add `const selectedMode = ref('pipeline')` and `onModeChange` handler
4. Pass mode to `chatApi.query(userQuery, topK.value, selectedMode.value)`
5. Display mode in assistant message header (optional)

### API Changes (web/frontend/src/api/index.js)

```js
query: async (query, topK = 10, mode = 'pipeline') => {
  return api.post('/chat/query', { query, top_k: topK, mode })
}
```

## Backend Design

### Schema Changes (web/api/schemas.py)

Add ChatMode enum and update ChatRequest:

```python
class ChatMode(str, Enum):
    PIPELINE = "pipeline"
    AGENT = "agent"
    PLAN = "plan"

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to retrieve")
    mode: ChatMode = Field(ChatMode.PIPELINE, description="Query mode")
    env_file: str = Field(".env", description="Path to .env config file")
```

### Service Changes (web/api/services.py)

Update ChatService.query to route by mode:

```python
@staticmethod
def query(
    query_str: str,
    top_k: int = 10,
    mode: str = "pipeline",
    env_file: str = ".env",
) -> Dict[str, Any]:
    config_path = PROJECT_ROOT / env_file
    config = load_config(str(config_path))
    pipeline = RAGPipeline(config)

    if mode == "plan":
        result = pipeline.query_with_agent(query_str, mode="plan", auto_approve=True)
        return ChatService._format_agent_response(result)
    elif mode == "agent":
        result = pipeline.query_with_agent(query_str, mode="agent")
        return ChatService._format_agent_response(result)
    else:
        return pipeline.query_with_images(query_str, top_k=top_k, include_images=True)

@staticmethod
def _format_agent_response(result: Dict) -> Dict:
    """Normalize agent response to ChatResponse format."""
    # Extract response text
    response = result.get("response", "")

    # Extract sources from tool calls or execution history
    sources = []
    if "sources" in result:
        for src in result["sources"]:
            sources.append({
                "node_id": src.get("node_id", ""),
                "text": src.get("text", "")[:300],
                "score": src.get("score", 0.0),
                "source_file": src.get("source_file"),
                "header_path": src.get("header_path"),
            })

    return {
        "response": response,
        "source_nodes": sources,
        "images": [],  # Agent mode doesn't return images currently
        "metadata": {"mode": result.get("mode", "agent")},
    }
```

### Route Changes (web/api/routes/chat.py)

Pass mode to service:

```python
@router.post("/query", response_model=schemas.ChatResponse)
async def query(request: schemas.ChatRequest):
    result = services.ChatService.query(
        query_str=request.query,
        top_k=request.top_k,
        mode=request.mode.value,
        env_file=request.env_file,
    )
    return schemas.ChatResponse(**result)
```

## Styling

Mode tabs should match existing workspace styling:
- Active tab: `var(--primary)` background, white text
- Inactive tabs: `var(--bg-secondary)` background, `var(--text-secondary)` text
- Hover state: subtle background change
- Compact height (~32px) to not dominate the chat area

## Implementation Order

1. Backend: Update schemas, services, routes
2. Frontend: Create ModeSelector component
3. Frontend: Update ChatView to use ModeSelector
4. Frontend: Update chatApi to pass mode
5. Testing: Verify each mode works correctly

## Notes

- Agent and Plan modes currently don't return images; this is acceptable for MVP
- Plan mode uses `auto_approve=True` to skip interactive approval in web context
- Future enhancement: show execution plan details in Plan mode