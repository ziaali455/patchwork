import { spawn, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdtemp, mkdir, rename, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { app, BrowserWindow, dialog } from 'electron'
import { editorConnection, keyEnv, readJson, readStore, updateStore } from './store'
import type { CachedTarget, EditorModel, EditorProfile, JobEvent, JobRequest, Run, Target } from '../shared/types'

const repository = join(app.getAppPath(), '..')
let active: { child?: ChildProcess; cancelled: boolean } | undefined

const runPath = (id: string) => join(app.getPath('userData'), 'runs', id)

async function chooseFolder(window: BrowserWindow, title: string) {
  const result = await dialog.showOpenDialog(window, { title, properties: ['openDirectory', 'createDirectory'] })
  return result.filePaths[0]
}

export function cancelJob() {
  if (!active) return
  active.cancelled = true
  const child = active.child
  if (child && child.exitCode === null && child.signalCode === null) {
    child.kill('SIGTERM')
    const force = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL')
    }, 3000)
    force.unref()
  }
}

function runPython(
  args: string[], job: NonNullable<typeof active>, log: (chunk: string) => void,
  env: NodeJS.ProcessEnv, input?: string,
) {
  const python = process.env.PATCHWORK_PYTHON || join(
    repository, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
  )
  if (!existsSync(python)) {
    throw new Error('Patchwork’s Python environment is missing. Run uv sync --extra local from the repository.')
  }
  if (job.cancelled) throw new Error('Cancelled')
  return new Promise<void>((resolve, reject) => {
    const child = spawn(python, ['-u', '-m', 'patchwork.cli', ...args], {
      cwd: repository,
      env,
      stdio: [input === undefined ? 'ignore' : 'pipe', 'pipe', 'pipe'],
      windowsHide: true,
    })
    job.child = child
    child.stdout?.setEncoding('utf8').on('data', log)
    child.stderr?.setEncoding('utf8').on('data', log)
    child.once('error', reject)
    child.once('close', code => {
      job.child = undefined
      if (code === 0 && !job.cancelled) resolve()
      else reject(new Error(job.cancelled ? 'Cancelled' : `Patchwork exited with code ${code}.`))
    })
    child.stdin?.on('error', () => {})
    child.stdin?.end(input, 'utf8')
  })
}

type Report = Pick<Run, 'goal' | 'before' | 'after'> & {
  editor_provider: Run['provider']
  editor_model: string
  editor_summary?: { summary?: string }
  edits?: Run['editedTensors']
  parameters_changed?: number
}

async function readPythonResult<T>(args: string[], env: NodeJS.ProcessEnv) {
  const temporary = await mkdtemp(join(tmpdir(), 'patchwork-query-'))
  const result = join(temporary, 'result.json')
  let log = ''
  try {
    await runPython([...args, '--json', result], { cancelled: false },
      chunk => { log = (log + chunk).slice(-3000) }, env)
    return readJson<T>(result)
  } catch (error) {
    throw new Error(log.trim() || (error as Error).message)
  } finally {
    await rm(temporary, { recursive: true, force: true })
  }
}

export function cachedTargets() {
  return readPythonResult<CachedTarget[]>(['cache'], process.env)
}

export function listEditorModels(connection: Pick<EditorProfile, 'provider' | 'endpoint'>, key?: string) {
  const { provider, endpoint, secret } = editorConnection(connection, key)
  return readPythonResult<EditorModel[]>(['models', '--editor-provider', provider,
    ...(endpoint ? [`--editor-base-url=${endpoint}`] : []),
    ...(secret ? ['--editor-key-env', keyEnv[provider]] : []),
  ], { ...process.env, [keyEnv[provider]]: secret, PYTHONIOENCODING: 'utf-8' })
}

function modelArgs(target: Target) {
  return ['--target-model', target.revision ? target.source : target.path,
    ...(target.revision ? ['--revision', target.revision] : [])]
}

async function perform(request: JobRequest, window: BrowserWindow, selection: string | undefined,
  task: NonNullable<typeof active>, onLog: (chunk: string) => void): Promise<Partial<JobEvent>> {
  const store = readStore()
  const env: NodeJS.ProcessEnv = { ...process.env, PYTHONIOENCODING: 'utf-8' }
  const run = (args: string[], input?: string) => runPython(args, task, onLog, env, input)
  const temporary = await mkdtemp(join(tmpdir(), 'patchwork-'))
  const result = join(temporary, 'result.json')
  try {
    if (request.kind === 'download' || request.kind === 'import' || request.kind === 'use-cached') {
      const source = request.kind === 'import' ? selection! : request.source.trim()
      const requestedRevision = request.kind === 'import' ? '' : request.revision.trim() || 'main'
      if (request.kind !== 'import' && !/^[\w.-]+(?:\/[\w.-]+)?$/.test(source)) {
        throw new Error('Enter a Hugging Face repository ID such as Qwen/Qwen3.5-0.8B.')
      }
      if (request.kind === 'use-cached') onLog('Verifying cached checkpoint…\n')
      await run(['download', `--target-model=${source}`, `--revision=${requestedRevision || 'main'}`, '--json', result,
        ...(request.kind === 'use-cached' ? ['--cached-only'] : [])])
      const info = readJson<Pick<Target, 'path' | 'revision' | 'fingerprint'>>(result)
      const existing = store.targets.find(target =>
        target.source === source && target.revision === info.revision && target.fingerprint === info.fingerprint)
      if (existing) return { targetId: existing.id }
      const target = { ...info, id: randomUUID(), source, name: basename(source), requestedRevision }
      updateStore(data => data.targets.unshift(target))
      return { targetId: target.id }
    }

    if (request.kind === 'export') {
      const commit = store.commits.find(item => item.id === request.commitId)
      const saved = store.runs.find(item => item.id === commit?.runId)
      if (!commit || !saved?.hasPatch) throw new Error('Select a committed patch to export.')
      const output = join(selection!, `patchwork-${commit.id.slice(0, 8)}`)
      if (existsSync(output)) throw new Error('This folder already contains the exported version.')
      // Export into a staging directory on the destination volume so a failure leaves no partial checkpoint.
      const stage = await mkdtemp(join(selection!, '.patchwork-'))
      try {
        const checkpoint = join(stage, 'checkpoint')
        await run(['export', '--patch', runPath(saved.id), '--output', checkpoint])
        await rename(checkpoint, output)
      } finally {
        await rm(stage, { recursive: true, force: true })
      }
      return { exportedTo: output }
    }

    const target = store.targets.find(item => item.id === request.targetId)
    if (!target) throw new Error('Select a target model.')
    const args = [...modelArgs(target), '--expect-fingerprint', target.fingerprint,
      '--device', request.device, '--max-new-tokens', String(request.tokens),
      ...(request.thinking ? ['--thinking'] : [])]

    if (request.kind === 'prompt') {
      let patchArgs: string[] = ['--dtype', request.dtype]
      if (request.version) {
        const commit = store.commits.find(item => item.id === request.version)
        const saved = store.runs.find(item => item.id === commit?.runId && item.targetId === target.id)
        if (!saved?.hasPatch) throw new Error('Select a version for this exact checkpoint.')
        patchArgs = ['--patch', runPath(saved.id), '--compare']
      }
      await run(['prompt', ...args, ...patchArgs, '--json', result], request.prompt)
      return { answers: readJson<JobEvent['answers']>(result) }
    }

    const editor = store.editor
    const { provider, endpoint, secret } = editorConnection(editor)
    if (secret) env[keyEnv[provider]] = secret
    const id = randomUUID()
    const output = runPath(id)
    await mkdir(join(output, '..'), { recursive: true })
    try {
      await run(['patch', ...args, '--dtype', request.dtype, `--prompt=${request.goal}`, '--passes',
        String(request.passes), '--eval', `gsm8k:${request.cases}`, '--output', output,
        '--editor-provider', provider, `--editor-model=${editor.model}`,
        ...(provider === 'openai-compatible' ? [
          `--editor-base-url=${endpoint}`, ...(secret ? ['--editor-key-env', keyEnv[provider]] : []),
        ] : []),
      ])
      const report = readJson<Report>(join(output, 'report.json'))
      const item: Run = {
        id, targetId: target.id, createdAt: new Date().toISOString(), goal: report.goal,
        provider: report.editor_provider, editorModel: report.editor_model, parametersChanged: report.parameters_changed ?? 0,
        editorSummary: report.editor_summary?.summary, editedTensors: report.edits,
        before: report.before, after: report.after, hasPatch: Boolean(report.after && report.parameters_changed),
      }
      updateStore(data => data.runs.unshift(item))
      return { runId: id }
    } catch (error) {
      await rm(output, { recursive: true, force: true })
      throw error
    }
  } finally {
    await rm(temporary, { recursive: true, force: true })
  }
}

export async function startJob(request: JobRequest, window: BrowserWindow) {
  if (active) throw new Error('Wait for the active job or cancel it first.')
  const task = { cancelled: false }
  active = task
  let selection: string | undefined
  try {
    if (request.kind === 'import' || request.kind === 'export') {
      selection = await chooseFolder(window, request.kind === 'import' ? 'Import Transformers checkpoint' : 'Export checkpoint to folder')
      if (!selection || task.cancelled) {
        active = undefined
        return false
      }
    }
  } catch (error) {
    active = undefined
    throw error
  }

  let log = ''
  let timer: ReturnType<typeof setTimeout> | undefined
  const publish = (event: Partial<JobEvent> & Pick<JobEvent, 'status'>) => {
    if (!window.isDestroyed()) window.webContents.send('job-event', { kind: request.kind, log, ...event })
  }
  publish({ status: 'running' })
  void perform(request, window, selection, task, chunk => {
    log = (log + chunk).slice(-18000)
    timer ??= setTimeout(() => {
      timer = undefined
      publish({ status: 'running' })
    }, 70)
  }).then(
    value => {
      clearTimeout(timer)
      active = undefined
      publish({ status: 'success', ...value })
    },
    (error: Error) => {
      clearTimeout(timer)
      active = undefined
      publish({ status: task.cancelled ? 'cancelled' : 'failed', error: error.message })
    },
  )
  return true
}
