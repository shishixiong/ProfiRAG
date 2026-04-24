import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

// Views
import PdfConvert from './views/PdfConvert.vue'
import DocSplitter from './views/DocSplitter.vue'
import DocImport from './views/DocImport.vue'
import ChatView from './views/ChatView.vue'

// Router
const routes = [
  { path: '/', redirect: '/pdf' },
  { path: '/pdf', name: 'PdfConvert', component: PdfConvert },
  { path: '/split', name: 'DocSplitter', component: DocSplitter },
  { path: '/import', name: 'DocImport', component: DocImport },
  { path: '/chat', name: 'ChatView', component: ChatView },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// App
const app = createApp(App)
app.use(router)
app.mount('#app')