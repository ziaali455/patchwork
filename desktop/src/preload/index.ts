import { contextBridge, ipcRenderer } from 'electron'
import type { DesktopAPI, JobEvent } from '../shared/types'

const api: DesktopAPI = {
  workspace: () => ipcRenderer.invoke('workspace'),
  cachedTargets: () => ipcRenderer.invoke('cached-targets'),
  saveEditor: (profile, key) => ipcRenderer.invoke('save-editor', profile, key),
  commit: (runId, message) => ipcRenderer.invoke('commit', runId, message),
  start: (job) => ipcRenderer.invoke('start-job', job),
  cancel: () => ipcRenderer.invoke('cancel-job'),
  onJob(callback) {
    const handler = (_event: Electron.IpcRendererEvent, job: JobEvent) => callback(job)
    ipcRenderer.on('job-event', handler)
    return () => ipcRenderer.removeListener('job-event', handler)
  },
}

contextBridge.exposeInMainWorld('patchwork', api)
