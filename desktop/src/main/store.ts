import { randomUUID } from 'node:crypto'
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { app, safeStorage } from 'electron'
import type { EditorProfile, Provider, Workspace } from '../shared/types'

type Store = Omit<Workspace, 'configuredKeys'> & {
  keys: Partial<Record<Provider, string>>
  compatibleKeyEndpoint?: string
}
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
  const { keys, compatibleKeyEndpoint: endpoint, ...data } = readStore()
  const configuredKeys = Object.fromEntries(
    (Object.keys(keyEnv) as Provider[]).map(provider =>
      [provider, Boolean(process.env[keyEnv[provider]] || (
        keys[provider] && (provider !== 'openai-compatible' ||
          (data.editor.provider === provider && (!endpoint || endpoint === data.editor.endpoint)))
      ))]),
  )
  return { ...data, configuredKeys }
}

function normalizeEndpoint(value: string) {
  let endpoint: URL
  try {
    endpoint = new URL(value.trim())
  } catch {
    throw new Error('Enter the API base URL, such as http://localhost:8000/v1.')
  }
  if (!['http:', 'https:'].includes(endpoint.protocol) || endpoint.username || endpoint.password
      || endpoint.search || endpoint.hash) {
    throw new Error('Use an HTTP or HTTPS API base URL without credentials, a query, or a fragment.')
  }
  return endpoint.href.replace(/\/+$/, '')
}

export function editorConnection(connection: Pick<EditorProfile, 'provider' | 'endpoint'>, key?: string) {
  const { provider } = connection
  if (!Object.hasOwn(keyEnv, provider)) throw new Error('Choose an editor provider.')
  const endpoint = provider === 'openai-compatible' ? normalizeEndpoint(connection.endpoint) : ''
  const secret = key?.trim() || editorKey(provider, endpoint)
  if (!secret && provider !== 'openai-compatible') throw new Error('Enter an API key for this provider.')
  return { provider, endpoint, secret }
}

function editorKey(provider: Provider, endpoint: string) {
  const store = readStore()
  const allowedEndpoint = store.compatibleKeyEndpoint ??
    (store.editor.provider === 'openai-compatible' ? store.editor.endpoint.trim().replace(/\/+$/, '') : '')
  const stored = provider !== 'openai-compatible' || endpoint === allowedEndpoint ? store.keys[provider] : undefined
  return stored
    ? safeStorage.decryptString(Buffer.from(stored, 'base64'))
    : process.env[keyEnv[provider]]
}

export function saveEditor(profile: EditorProfile, key?: string) {
  if (!Object.hasOwn(keyEnv, profile.provider)) throw new Error('Choose an editor provider.')
  const model = profile.model.trim()
  if (!model) throw new Error('Enter an editor model ID.')
  const endpoint = profile.provider === 'openai-compatible' ? normalizeEndpoint(profile.endpoint) : ''
  const secret = key?.trim()
  if (secret && (!safeStorage.isEncryptionAvailable() ||
    (process.platform === 'linux' && safeStorage.getSelectedStorageBackend() === 'basic_text'))) {
    throw new Error('Secure credential storage is unavailable. Set the API key in the environment.')
  }
  updateStore(store => {
    if (store.keys['openai-compatible'] && !store.compatibleKeyEndpoint && store.editor.provider === 'openai-compatible') {
      store.compatibleKeyEndpoint = store.editor.endpoint.trim().replace(/\/+$/, '')
    }
    store.editor = { provider: profile.provider, model, endpoint }
    if (secret) {
      store.keys[profile.provider] = safeStorage.encryptString(secret).toString('base64')
      if (profile.provider === 'openai-compatible') store.compatibleKeyEndpoint = endpoint
    }
  })
  return workspace()
}
