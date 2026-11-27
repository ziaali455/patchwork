export type Device = 'cpu' | 'mps' | 'cuda'
export type Dtype = 'float32' | 'float16' | 'bfloat16'
export type Provider = 'anthropic' | 'openai' | 'openai-compatible'

export interface Target {
  id: string
  name: string
  source: string
  path: string
  requestedRevision: string
  revision: string | null
  fingerprint: string
}

export interface CachedTarget {
  source: string
  revision: string
}

export interface EditorProfile {
  provider: Provider
  model: string
  endpoint: string
}

export interface EditorModel {
  id: string
  label: string
}

export interface Score {
  accuracy: number
  reached_token_limit: number
  total: number
}

export interface Run {
  id: string
  targetId: string
  createdAt: string
  goal: string
  provider: Provider
  editorModel: string
  editorSummary?: string
  editedTensors?: { tensor: string; operation: string; parameters_changed: number }[]
  parametersChanged: number
  before: Score
  after: Score | null
  hasPatch: boolean
}

export interface Commit {
  id: string
  runId: string
  createdAt: string
  message: string
}

export interface Workspace {
  targets: Target[]
  runs: Run[]
  commits: Commit[]
  editor: EditorProfile
  configuredKeys: Partial<Record<Provider, boolean>>
}

export interface RuntimeOptions {
  targetId: string
  device: Device
  dtype: Dtype
  tokens: number
  thinking: boolean
}

export type JobRequest =
  | { kind: 'download'; source: string; revision: string }
  | ({ kind: 'use-cached' } & CachedTarget)
  | { kind: 'import' }
  | ({ kind: 'edit'; goal: string; passes: number; cases: number } & RuntimeOptions)
  | ({ kind: 'prompt'; prompt: string; version?: string } & RuntimeOptions)
  | { kind: 'export'; commitId: string }

export interface Answer {
  answer: string
  raw_output: string
  generated_tokens: number
  generation_seconds: number
  reached_token_limit: boolean
}

export interface JobEvent {
  kind: JobRequest['kind']
  status: 'running' | 'success' | 'failed' | 'cancelled'
  log: string
  error?: string
  answers?: { base?: Answer; patched?: Answer }
  runId?: string
  targetId?: string
  exportedTo?: string
}

export interface DesktopAPI {
  workspace(): Promise<Workspace>
  cachedTargets(): Promise<CachedTarget[]>
  listEditorModels(connection: Pick<EditorProfile, 'provider' | 'endpoint'>, key?: string): Promise<EditorModel[]>
  saveEditor(profile: EditorProfile, key?: string): Promise<Workspace>
  commit(runId: string, message: string): Promise<Workspace>
  start(job: JobRequest): Promise<boolean>
  cancel(): Promise<void>
  onJob(callback: (event: JobEvent) => void): () => void
}
