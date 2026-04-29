# 文档检索代码格式显示功能设计

## 概述

在 SearchView 的检索结果中，根据文件扩展名自动识别代码文件，使用代码块格式显示内容，包括语法高亮、行号和复制按钮。

## 需求

- 根据文件扩展名自动判断是否为代码文件
- 代码文件内容使用代码块渲染（而非普通文本）
- 显示行号便于定位
- 提供复制按钮，一键复制代码内容
- 支持常见编程语言的语法高亮

## 技术方案

### 新增依赖

```bash
npm install highlight.js
```

项目已有 `marked` 依赖，用于 Markdown 解析。新增 `highlight.js` 用于代码语法高亮。

### 新建组件

路径：`web/frontend/src/components/CodeBlock.vue`

组件职责：
- 接收代码文本和语言标识
- 使用 highlight.js 进行语法高亮
- 渲染带行号的代码块
- 提供复制功能按钮

Props：
- `code: string` - 代码文本内容
- `language: string` - 语言标识（如 'python', 'javascript'）

组件结构：
``┌────────────────────────────────────┐``
│                            [复制]   │
│ 1 │ code line 1                    │
│ 2 │ code line 2                    │
│...│ ...                             │
└────────────────────────────────────┘``

样式要点：
- 行号区域：较深背景色，右对齐，不可选中
- 代码区域：语法高亮，可选中复制
- 复制按钮：hover 显示，点击后短暂变绿色 + "已复制" 文字

### 修改 SearchView

文件：`web/frontend/src/views/SearchView.vue`

修改点：

1. 导入 CodeBlock 组件
2. 新增 `isCodeFile(filename)` 函数，根据扩展名判断是否代码文件
3. 新增 `getLanguage(filename)` 函数，映射扩展名到 highlight.js 语言标识
4. 在 `chunk-full` 区域修改渲染逻辑：
   - 代码文件：使用 `<CodeBlock>` 组件
   - 普通文件：保持原有 `<div class="chunk-text">` 显示

### 语言映射表

| 扩展名 | 语言标识 |
|--------|----------|
| `.py` | python |
| `.java` | java |
| `.js` | javascript |
| `.jsx` | javascript |
| `.ts` | typescript |
| `.tsx` | typescript |
| `.go` | go |
| `.cpp, .cc, .cxx` | cpp |
| `.c` | c |
| `.h` | c |
| `.hpp` | cpp |
| `.rs` | rust |
| `.rb` | ruby |
| `.php` | php |
| `.sh, .bash` | bash |
| `.sql` | sql |
| `.json` | json |
| `.yaml, .yml` | yaml |
| `.xml` | xml |
| `.html, .htm` | html |
| `.css` | css |
| `.scss, .sass` |scss |
| `.md, .markdown` | markdown |
| `.vue` | vue |
| `.kt, .kts` | kotlin |

### 样式设计

CodeBlock 组件样式：

```css
.code-block {
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  position: relative;
  font-family: monospace;
}

.line-numbers {
  background: var(--border);
  color: var(--text-secondary);
  text-align: right;
  user-select: none;
}

.code-content {
  overflow-x: auto;
}

.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  opacity: 0;
  transition: opacity 0.2s;
}

.code-block:hover .copy-btn {
  opacity: 1;
}

.copy-btn.copied {
  background: #22c55e;
  color: white;
}
```

## 数据流

```
SearchService.query() → SearchResponse.chunks[]
  ↓
SearchView.filteredChunks[]
  ↓
chunk.full_text + chunk.source_file
  ↓
isCodeFile(source_file)?
  ↓ yes                ↓ no
CodeBlock            chunk-text
(code, language)
```

## 测试要点

1. 代码文件正确识别（如 `main.py` → 使用 CodeBlock）
2. 非代码文件保持原有显示（如 `readme.txt` → 普通文本）
3. 复制按钮功能正常，点击后显示"已复制"
4. 行号正确对应代码行
5. 语法高亮正确显示
6. 长代码行水平滚动正常

## 文件清单

| 文件 | 操作 |
|------|------|
| `web/frontend/package.json` | 添加 highlight.js 依赖 |
| `web/frontend/src/components/CodeBlock.vue` | 新建 |
| `web/frontend/src/views/SearchView.vue` | 修改 |