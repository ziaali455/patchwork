import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

export default defineConfig({
  main: {},
  preload: {},
  renderer: {
    server: { fs: { allow: [__dirname, resolve(__dirname, '../logos')] } },
    plugins: [
      react(),
      {
        name: 'patchwork-dev-refresh',
        apply: 'serve',
        transformIndexHtml: html => html.replace("default-src 'self';", "default-src 'self'; script-src 'self' 'unsafe-inline';"),
      },
    ],
  },
})
