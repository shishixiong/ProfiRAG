import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,  // 5 minutes for import operations
})

// PDF endpoints
export const pdfApi = {
  upload: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/pdf/upload', formData)
  },

  convert: async (fileId, options) => {
    return api.post(`/pdf/convert/${fileId}`, options)
  },

  preview: async (fileId) => {
    return api.get(`/pdf/preview/${fileId}`)
  },

  download: async (fileId) => {
    return api.get(`/pdf/download/${fileId}`, { responseType: 'blob' })
  },

  delete: async (fileId) => {
    return api.delete(`/pdf/${fileId}`)
  },
}

// Split endpoints
export const splitApi = {
  upload: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/split/upload', formData)
  },

  preview: async (options) => {
    return api.post('/split/preview', options)
  },

  getChunk: async (fileId, chunkIndex, full = false) => {
    return api.get(`/split/chunks/${fileId}/${chunkIndex}`, { params: { full } })
  },

  download: async (fileId, format = 'json') => {
    return api.post('/split/download', { file_id: fileId, output_format: format }, { responseType: 'blob' })
  },

  delete: async (fileId) => {
    return api.delete(`/split/${fileId}`)
  },
}

// Import endpoints
export const importApi = {
  upload: async (files) => {
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))
    return api.post('/import/upload', formData)
  },

  start: async (fileIds, config) => {
    return api.post('/import/start', { file_ids: fileIds, config })
  },

  progress: async (jobId) => {
    return api.get(`/import/progress/${jobId}`)
  },

  stats: async (jobId) => {
    return api.get(`/import/stats/${jobId}`)
  },

  deleteFile: async (fileId) => {
    return api.delete(`/import/files/${fileId}`)
  },
}

export default api

// Chat endpoints
export const chatApi = {
  query: async (query, topK = 10, mode = 'pipeline') => {
    return api.post('/chat/query', { query, top_k: topK, mode })
  },
}