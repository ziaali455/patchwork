import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { app, BrowserWindow, ipcMain } from 'electron'
import { cachedTargets, cancelJob, startJob } from './jobs'
import { saveEditor, updateStore, workspace } from './store'
import type { EditorProfile, JobRequest } from '../shared/types'

let window: BrowserWindow | null = null

function createWindow() {
  window = new BrowserWindow({
    width: 1320,
    height: 900,
    minWidth: 850,
    minHeight: 650,
    backgroundColor: '#171819',
    title: 'Patchwork',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 18, y: 18 },
    webPreferences: { preload: join(__dirname, '../preload/index.js'), sandbox: true, contextIsolation: true },
  })
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  window.webContents.on('will-navigate', event => event.preventDefault())
  window.on('closed', () => { window = null })
  const devUrl = process.env.ELECTRON_RENDERER_URL
  if (!app.isPackaged && devUrl) void window.loadURL(devUrl)
  else void window.loadFile(join(__dirname, '../renderer/index.html'))
}

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.setName('Patchwork')
  app.on('second-instance', () => {
    if (!window) createWindow()
    if (window?.isMinimized()) window.restore()
    window?.focus()
  })
  void app.whenReady().then(() => {
    const env = join(app.getAppPath(), '..', '.env')
    if (existsSync(env)) process.loadEnvFile(env)

    const handle = <Args extends unknown[], Result>(channel: string, fn: (...args: Args) => Result) => {
      ipcMain.handle(channel, (event, ...args) => {
        if (!window || event.sender !== window.webContents || event.senderFrame !== event.sender.mainFrame) {
          throw new Error('Invalid Patchwork window.')
        }
        return fn(...args as Args)
      })
    }

    handle('workspace', workspace)
    handle('cached-targets', cachedTargets)
    handle('save-editor', (profile: EditorProfile, key?: string) => saveEditor(profile, key))
    handle('commit', (runId: string, message: string) => {
      const trimmed = message.trim()
      if (!trimmed || trimmed.length > 500) throw new Error('Enter a commit message of 1–500 characters.')
      updateStore(store => {
        if (!store.runs.some(run => run.id === runId && run.hasPatch)) throw new Error('Select a completed patch.')
        if (store.commits.some(commit => commit.runId === runId)) throw new Error('This patch is already committed.')
        store.commits.unshift({ id: randomUUID(), runId, message: trimmed, createdAt: new Date().toISOString() })
      })
      return workspace()
    })
    handle('start-job', (request: JobRequest) => startJob(request, window!))
    handle('cancel-job', cancelJob)
    createWindow()
    app.on('activate', () => { if (!window) createWindow() })
  })
  app.on('window-all-closed', () => {
    cancelJob()
    if (process.platform !== 'darwin') app.quit()
  })
  app.on('before-quit', cancelJob)
}
