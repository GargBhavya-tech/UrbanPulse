import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Dev: forward API calls to the FastAPI layer (scripts/11_serve.py)
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
      // Dev: forward /reports to FastAPI static mount (SHAP PNGs, charts)
      '/reports': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
