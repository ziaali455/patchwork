import { randomUUID } from 'node:crypto'
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { app, safeStorage } from 'electron'
import type { EditorProfile, Provider, Workspace } from '../shared/types'

type Store = Omit<Workspace, 'configuredKeys'> & { keys: Partial<Record<Provider, string>> }
export const keyEnv: Record<Provider, string> = {
  anthropic: 'ANTHROPIC_API_KEY',
  openai: 'OPENAI_API_KEY',
  'openai-compatible': 'PATCHWORK_COMPATIBLE_API_KEY',
}

export function readJson<T>(file: string): T {
  return JSON.parse(readFileSync(file, 'utf8'))
}

function file() {
  return join(app.getPath('userData'), 'workspace.json')
}

export function readStore(): Store {
  try {
    return readJson<Store>(file())
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
    return {
      targets: [], runs: [], commits: [], keys: {},
      editor: { provider: 'anthropic', model: 'sonnet', endpoint: '' },
    }
  }
}

export function updateStore(change: (store: Store) => void) {
  const store = readStore()
  change(store)
  const destination = file()
  mkdirSync(dirname(destination), { recursive: true })
  const temporary = join(dirname(destination), `workspace-${randomUUID()}.tmp`)
  writeFileSync(temporary, JSON.stringify(store, null, 2), { mode: 0o600, flag: 'wx' })
  renameSync(temporary, destination)
}

export function workspace(): Workspace {
  const { keys, ...data } = readStore()
  const configuredKeys = Object.fromEntries(
    (Object.keys(keyEnv) as Provider[]).map(provider =>
      [provider, Boolean(keys[provider] || process.env[keyEnv[provider]])]),
  )
  return { ...data, configuredKeys }
}

export function editorKey(provider: Provider) {
  const stored = readStore().keys[provider]
  return stored
    ? safeStorage.decryptString(Buffer.from(stored, 'base64'))
    : process.env[keyEnv[provider]]
}

export function saveEditor(profile: EditorProfile, key?: string) {
  if (!Object.hasOwn(keyEnv, profile.provider)) throw new Error('Choose an editor provider.')
  const model = profile.model.trim()
  const endpoint = profile.endpoint.trim()
  if (!model) throw new Error('Enter an editor model ID.')
  if (profile.provider === 'openai-compatible') {
    if (!endpoint || !['http:', 'https:'].includes(new URL(endpoint).protocol)) {
      throw new Error('Enter an HTTP or HTTPS endpoint.')
    }
  }
  const secret = key?.trim()
  if (secret && (!safeStorage.isEncryptionAvailable() ||
    (process.platform === 'linux' && safeStorage.getSelectedStorageBackend() === 'basic_text'))) {
    throw new Error('Secure credential storage is unavailable. Set the API key in the environment.')
  }
  updateStore(store => {
    store.editor = { provider: profile.provider, model, endpoint }
    if (secret) store.keys[profile.provider] = safeStorage.encryptString(secret).toString('base64')
  })
  return workspace()
}
