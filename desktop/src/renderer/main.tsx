import { StrictMode, useEffect, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  ArrowDownToLine, ArrowRight, Box, ChevronRight, CircleHelp, Cpu, FolderOpen,
  GitCommitHorizontal, History, Layers3, MessageSquare, Play, Settings2, Terminal,
  WandSparkles, X,
} from 'lucide-react'
import logo from '../../../logos/patchwork-logo.svg'
import type {
  Answer, CachedTarget, DesktopAPI, Device, Dtype, EditorProfile, JobEvent, JobRequest, Run, Target, Workspace,
} from '../shared/types'
import './style.css'

declare global {
  interface Window { patchwork: DesktopAPI }
}

const api = window.patchwork
const pages = [
  { label: 'Playground', icon: MessageSquare, subtitle: 'Prompt a local model and compare responses across versions.' },
  { label: 'Models', icon: Layers3, subtitle: 'Download and manage the checkpoints in your workspace.' },
  { label: 'Edit', icon: WandSparkles, subtitle: 'Describe a behavior change and evaluate the resulting weight patch.' },
  { label: 'History', icon: History, subtitle: 'Review edit runs, write commits, and revisit saved model versions.' },
  { label: 'Editor settings', icon: Settings2, subtitle: 'Connect the model that proposes and evaluates weight edits.' },
] as const
type Page = typeof pages[number]['label']

const providerLabels = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  'openai-compatible': 'OpenAI-compatible',
}

const date = (value: string) => new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value))
const percent = (value: number) => new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 }).format(value)

function Panel({ title, icon, children, action, className = '' }: {
  title: string; icon?: ReactNode; children: ReactNode; action?: ReactNode; className?: string
}) {
  return <section className={`panel ${className}`}>
    <header className="panel-head"><span>{icon}{title}</span>{action}</header>
    {children}
  </section>
}

function Empty({ icon, title, detail, action }: { icon: ReactNode; title: string; detail: string; action?: ReactNode }) {
  return <div className="empty">
    <div className="empty-icon">{icon}</div>
    <strong>{title}</strong>
    <p>{detail}</p>
    {action}
  </div>
}

function Response({ answer, label, running }: { answer?: Answer; label: string; running: boolean }) {
  return <Panel title={label} icon={<MessageSquare size={15} />} className="response"
    action={answer && <small>{answer.generated_tokens} tokens · {answer.generation_seconds.toFixed(1)}s</small>}>
    {answer ? <div className="answer">
      <div>{answer.answer || 'The model returned no final text.'}</div>
      {answer.reached_token_limit && <small>Token limit reached</small>}
      {answer.raw_output !== answer.answer && <details><summary>Raw output</summary><pre>{answer.raw_output}</pre></details>}
    </div> : <Empty icon={<MessageSquare size={19} />}
      title={running ? 'Generating response' : 'No response yet'}
      detail={running ? 'Follow progress in the job log below.' : 'Run a prompt to see the model’s output here.'} />}
  </Panel>
}

function EditorForm({ data, save }: { data: Workspace; save(profile: EditorProfile, key?: string): Promise<boolean> }) {
  const [profile, setProfile] = useState(data.editor)
  const [key, setKey] = useState('')
  const [saving, setSaving] = useState(false)
  async function submit(event: FormEvent) {
    event.preventDefault()
    setSaving(true)
    if (await save(profile, key)) setKey('')
    setSaving(false)
  }
  return <Panel title="Editor connection" icon={<Settings2 size={17} />} className="narrow">
    <form className="form-body" onSubmit={submit}>
      <label>Provider<select value={profile.provider} onChange={event => {
        const provider = event.target.value as EditorProfile['provider']
        setProfile({ provider, model: provider === data.editor.provider ? data.editor.model : provider === 'anthropic' ? 'sonnet' : '',
          endpoint: provider === data.editor.provider ? data.editor.endpoint : '' })
        setKey('')
      }}>
        {Object.entries(providerLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select></label>
      <label>Editor model<input required value={profile.model} placeholder="Provider model ID"
        onChange={event => setProfile({ ...profile, model: event.target.value })} /></label>
      {profile.provider === 'openai-compatible' && <label>API base URL<input type="url" required value={profile.endpoint}
        placeholder="http://localhost:8000/v1" onChange={event => setProfile({ ...profile, endpoint: event.target.value })} /></label>}
      <label>API key <input type="password" autoComplete="off" value={key} onChange={event => setKey(event.target.value)}
        placeholder={data.configuredKeys[profile.provider] ? 'Configured · enter a replacement key' : 'Enter an API key'} />
        <small>{profile.provider === 'openai-compatible' ? 'Optional for endpoints without authentication.' : 'Required to run edits with this provider.'}</small>
      </label>
      <button className="primary" disabled={saving}>Save connection</button>
    </form>
  </Panel>
}

function HistoryView({ data, busy, commit, exportVersion }: {
  data: Workspace; busy: boolean; commit(id: string, message: string): Promise<boolean>; exportVersion(id: string): void
}) {
  const [openRun, setOpenRun] = useState('')
  const [message, setMessage] = useState('')
  const [saving, setSaving] = useState(false)
  const commits = new Map(data.commits.map(item => [item.runId, item]))
  const targets = new Map(data.targets.map(item => [item.id, item]))
  const showRun = (run: Run) => {
    setOpenRun(openRun === run.id ? '' : run.id)
    setMessage('')
  }
  return <Panel title="Model timeline" icon={<GitCommitHorizontal size={18} />}
    action={<small>{data.commits.length} {data.commits.length === 1 ? 'commit' : 'commits'}</small>}>
    {!data.runs.length && <Empty icon={<GitCommitHorizontal size={23} />} title="No edit runs yet"
      detail="Completed runs appear here with their benchmark and patch status." />}
    {data.runs.map(run => {
      const saved = commits.get(run.id)
      const model = targets.get(run.targetId)
      return <article className="history-row" key={run.id}>
        <button className="row-button" onClick={() => showRun(run)} aria-expanded={openRun === run.id}>
          <span className="row-icon"><GitCommitHorizontal size={19} /></span>
          <span className="row-text"><strong>{saved?.message || run.goal}</strong>
            <small>{model?.name} · {run.editorModel} · {date(saved?.createdAt || run.createdAt)}</small></span>
          <span className={`tag ${saved ? 'confirmed' : ''}`}>{saved ? saved.id.slice(0, 8) : run.hasPatch ? 'Uncommitted' : 'No patch'}</span>
          <ChevronRight size={16} className={openRun === run.id ? 'rotated' : ''} />
        </button>
        {openRun === run.id && <div className="history-detail">
          {saved && <p>{run.goal}</p>}
          <dl className="metrics">
            <div><dt>Baseline</dt><dd>{percent(run.before.accuracy)}</dd></div>
            <div><dt>Patched</dt><dd>{run.after ? percent(run.after.accuracy) : '—'}</dd></div>
            <div><dt>Weights changed</dt><dd>{run.parametersChanged.toLocaleString()}</dd></div>
            <div><dt>Token-limit hits</dt><dd>{run.before.reached_token_limit}{run.after ? ` → ${run.after.reached_token_limit}` : ''}</dd></div>
          </dl>
          {model && <p className="mono muted">Base {model.revision?.slice(0, 10) ?? 'local'} · {model.fingerprint.slice(0, 16)}</p>}
          {run.editorSummary && <p className="muted">{run.editorSummary}</p>}
          {Boolean(run.editedTensors?.length) && <details>
            <summary>Tensor changes</summary>
            <ul className="tensor-list">
              {run.editedTensors?.map((edit, index) => <li key={index}>
                <code>{edit.tensor}</code>
                <span>{edit.operation} · {edit.parameters_changed.toLocaleString()} weights</span>
              </li>)}
            </ul>
          </details>}
          {saved ? <button className="secondary" disabled={busy} onClick={() => exportVersion(saved.id)}>
            <ArrowDownToLine size={16} /> Export checkpoint
          </button> : run.hasPatch && <form className="commit-form" onSubmit={async event => {
            event.preventDefault()
            setSaving(true)
            if (await commit(run.id, message)) setMessage('')
            setSaving(false)
          }}>
            <label>Commit message<input required maxLength={500} value={message} onChange={event => setMessage(event.target.value)}
              placeholder="Describe what changed in this version" /></label>
            <button className="primary" disabled={saving || !message.trim()}><GitCommitHorizontal size={17} /> Commit patch</button>
          </form>}
        </div>}
      </article>
    })}
  </Panel>
}

function App() {
  const [data, setData] = useState<Workspace>()
  const [cached, setCached] = useState<CachedTarget[]>()
  const [cacheError, setCacheError] = useState('')
  const [page, setPage] = useState<Page>('Playground')
  const [selection, setSelection] = useState('')
  const [version, setVersion] = useState('')
  const [prompt, setPrompt] = useState('')
  const [goal, setGoal] = useState('')
  const [repo, setRepo] = useState('')
  const [revision, setRevision] = useState('')
  const [device, setDevice] = useState<Device>('cpu')
  const [dtype, setDtype] = useState<Dtype>('float32')
  const [tokens, setTokens] = useState(512)
  const [thinking, setThinking] = useState(false)
  const [passes, setPasses] = useState(10)
  const [cases, setCases] = useState(5)
  const [job, setJob] = useState<JobEvent>()
  const [answers, setAnswers] = useState<JobEvent['answers']>()
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [logOpen, setLogOpen] = useState(false)

  useEffect(() => {
    if (!api) {
      setError('Launch Patchwork through Electron to connect to the local runtime.')
      return
    }
    void api.workspace().then(setData, err => setError(err.message))
    void api.cachedTargets().then(setCached, err => {
      setCached([])
      setCacheError(err.message)
    })
    return api.onJob(event => {
      setJob(event)
      if (event.status === 'running') return
      setRunning(false)
      if (event.status === 'failed') {
        setError(event.error || 'The job failed.')
        setLogOpen(true)
      }
      if (event.status === 'cancelled') setNotice('Job cancelled.')
      if (event.status !== 'success') return
      if (event.answers) setAnswers(event.answers)
      if (event.targetId) {
        setSelection(event.targetId)
        setVersion('')
        setAnswers(undefined)
        setNotice(event.kind === 'import' ? 'Checkpoint imported.' : 'Checkpoint is ready.')
        if (event.kind !== 'use-cached') setPage('Models')
      }
      if (event.runId) {
        setPage('History')
        setNotice('Edit run complete. Open it to review the benchmark.')
      }
      if (event.exportedTo) setNotice(`Checkpoint exported to ${event.exportedTo}`)
      void api.workspace().then(setData, err => setError(err.message))
    })
  }, [])

  const target = data?.targets.find(item => item.id === selection) ?? data?.targets[0]
  const cachedOptions = cached?.filter(item =>
    !data?.targets.some(saved => saved.source === item.source && saved.revision === item.revision),
  ) ?? []
  const versions = data?.commits.filter(item => data.runs.some(run => run.id === item.runId && run.targetId === target?.id)) ?? []
  const selectedVersion = versions.find(item => item.id === version)
  const activePage = pages.find(item => item.label === page)!
  const onPlayground = page === 'Playground'
  const isPrompting = running && job?.kind === 'prompt'

  function selectTarget(item: Target | undefined) {
    setSelection(item?.id ?? '')
    setVersion('')
    setAnswers(undefined)
  }

  async function start(request: JobRequest) {
    setError('')
    setNotice('')
    setRunning(true)
    if (request.kind === 'prompt') setAnswers(undefined)
    if (request.kind !== 'prompt') setLogOpen(true)
    try {
      if (!await api.start(request)) setRunning(false)
    } catch (err) {
      setError((err as Error).message)
      setRunning(false)
    }
  }

  async function update(operation: () => Promise<Workspace>, message: string) {
    setError('')
    setNotice('')
    try {
      setData(await operation())
      setNotice(message)
      return true
    } catch (err) {
      setError((err as Error).message)
      return false
    }
  }

  function submitModel(event: FormEvent) {
    event.preventDefault()
    if (!target) return
    const runtime = { targetId: target.id, device, dtype, tokens, thinking }
    void start(onPlayground ? { kind: 'prompt', ...runtime, prompt, version: selectedVersion?.id }
      : { kind: 'edit', ...runtime, goal, passes, cases })
  }

  return <div className="shell">
    <div className="titlebar" />
    <aside className="sidebar">
      <div className="brand"><img src={logo} alt="" /><span>patchwork</span></div>
      <nav aria-label="Workspace">
        <div className="nav-label">Workspace</div>
        {pages.slice(0, 4).map(({ label, icon: Icon }) => <button key={label} onClick={() => setPage(label)}
          aria-current={page === label ? 'page' : undefined} className={page === label ? 'selected' : ''}>
          <Icon size={17} />{label}
        </button>)}
        <div className="nav-label configuration">Configuration</div>
        <button onClick={() => setPage('Editor settings')} className={page === 'Editor settings' ? 'selected' : ''}
          aria-current={page === 'Editor settings' ? 'page' : undefined}><Settings2 size={17} />Editor settings</button>
      </nav>
      <div className="sidebar-foot"><span className="local-dot" /><span>Local workspace</span></div>
    </aside>
    <main>
      <div className="topbar"><span>Workspace</span><ChevronRight size={14} /><strong>{page}</strong></div>
      <div className="content">
        <header className="page-head">
          <div className="page-icon"><activePage.icon size={21} /></div>
          <div><h1>{page}</h1><p>{activePage.subtitle}</p></div>
        </header>
        {error && <div className="banner error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError('')}><X size={15} /></button></div>}
        {notice && <div className="banner success" role="status">{notice}<button aria-label="Dismiss notice" onClick={() => setNotice('')}><X size={15} /></button></div>}
        {!data && !error && <p role="status" className="muted">Loading workspace…</p>}
        {data && (onPlayground || page === 'Edit') && <>
          <form className="work-grid" onSubmit={submitModel}>
            <Panel title={onPlayground ? 'Prompt' : 'Edit goal'} icon={onPlayground ? <Terminal size={16} /> : <WandSparkles size={16} />}>
              <textarea className="prompt-input" aria-label={onPlayground ? 'Prompt' : 'Edit goal'}
                value={onPlayground ? prompt : goal} onChange={event => (onPlayground ? setPrompt : setGoal)(event.target.value)}
                placeholder={onPlayground ? 'Ask the target model something…' : 'Describe the behavior you want to change…'} required />
              <footer className="panel-footer">
                <span>{onPlayground ? 'Single-turn inference' : `${providerLabels[data.editor.provider]} · ${data.editor.model}`}</span>
                <button className="primary" disabled={!target || running || !(onPlayground ? prompt : goal).trim()}>
                  {onPlayground ? <Play size={15} /> : <WandSparkles size={15} />}
                  {onPlayground ? 'Run prompt' : 'Run edit'}
                </button>
              </footer>
            </Panel>
            <Panel title="Run configuration" icon={<Settings2 size={16} />}>
              <div className="form-body compact">
                <label>Target model<select value={target?.id ?? ''} disabled={running || (!data.targets.length && !cachedOptions.length)}
                  onChange={event => {
                    const id = event.target.value
                    const match = cachedOptions.find(item => `${item.source}@${item.revision}` === id)
                    if (match) void start({ kind: 'use-cached', ...match })
                    else selectTarget(data.targets.find(item => item.id === id))
                  }}>
                  {!data.targets.length && <option value="" disabled>
                    {cachedOptions.length ? 'Select a target model' : cached ? 'No models yet' : 'Checking model cache…'}
                  </option>}
                  {data.targets.map(item => <option value={item.id} key={item.id}>
                    {item.name} · {(item.revision || item.fingerprint).slice(0, 8)}
                  </option>)}
                  {cachedOptions.length > 0 && <optgroup label="Hugging Face cache">
                    {cachedOptions.map(item => {
                      const id = `${item.source}@${item.revision}`
                      return <option key={id} value={id}>{item.source} · {item.revision.slice(0, 8)}</option>
                    })}
                  </optgroup>}
                </select></label>
                {cacheError && <small title={cacheError}>Couldn’t scan the Hugging Face cache.</small>}
                {!target && <button type="button" className="text-button" onClick={() => setPage('Models')}>
                  Add a target model <ArrowRight size={14} />
                </button>}
                {onPlayground && <label>Compare with<select value={version} disabled={running || !versions.length} onChange={event => {
                  setVersion(event.target.value)
                  setAnswers(undefined)
                }}>
                  <option value="">Base only</option>
                  {versions.map(item => <option key={item.id} value={item.id}>{item.message}</option>)}
                </select></label>}
                <div className="split-fields">
                  <label>Device<select value={device} onChange={event => setDevice(event.target.value as Device)}>
                    <option value="cpu">CPU</option><option value="mps">Apple MPS</option><option value="cuda">CUDA</option>
                  </select></label>
                  <label>Max tokens<input type="number" min={1} max={32768} required value={tokens || ''} onChange={event => setTokens(+event.target.value)} /></label>
                </div>
                <details><summary>Advanced options</summary>
                  <div className="advanced">
                    <label>Weight precision<select value={dtype} disabled={onPlayground && Boolean(selectedVersion)}
                      onChange={event => setDtype(event.target.value as Dtype)}>
                      <option value="float32">Float32</option><option value="float16">Float16</option><option value="bfloat16">BFloat16</option>
                    </select></label>
                    <label className="checkbox"><input type="checkbox" checked={thinking}
                      onChange={event => setThinking(event.target.checked)} /> Enable thinking</label>
                    {!onPlayground && <div className="split-fields">
                      <label>Editor passes<input type="number" min={1} required value={passes || ''} onChange={event => setPasses(+event.target.value)} /></label>
                      <label>Eval cases<input type="number" min={1} required value={cases || ''} onChange={event => setCases(+event.target.value)} /></label>
                    </div>}
                  </div>
                </details>
              </div>
            </Panel>
          </form>
          {onPlayground && <div className={`responses ${selectedVersion ? 'compare' : ''}`}>
            <Response label="Base response" answer={answers?.base} running={isPrompting} />
            {selectedVersion && <Response label="Patched response" answer={answers?.patched} running={isPrompting} />}
          </div>}
          {!onPlayground && !data.configuredKeys[data.editor.provider] && data.editor.provider !== 'openai-compatible' &&
            <div className="banner"><CircleHelp size={17} /> Connect your editor to run an edit.
              <button className="text-button" onClick={() => setPage('Editor settings')}>Editor settings <ArrowRight size={14} /></button></div>}
        </>}
        {data && page === 'Models' && <>
          <Panel title="Add a target" icon={<ArrowDownToLine size={17} />}
            action={<button className="secondary" disabled={running} onClick={() => void start({ kind: 'import' })}><FolderOpen size={16} /> Import local</button>}>
            <form className="download-form" onSubmit={event => {
              event.preventDefault()
              void start({ kind: 'download', source: repo, revision })
            }}>
              <label>Hugging Face repository<input required value={repo} onChange={event => setRepo(event.target.value)} placeholder="Qwen/Qwen3.5-0.8B" /></label>
              <label>Revision<input value={revision} onChange={event => setRevision(event.target.value)} placeholder="main" /></label>
              <button className="primary" disabled={running || !repo.trim()}><ArrowDownToLine size={16} /> Download</button>
            </form>
          </Panel>
          <Panel title="Model library" icon={<Layers3 size={17} />} action={<small>{data.targets.length} {data.targets.length === 1 ? 'checkpoint' : 'checkpoints'}</small>}>
            {!data.targets.length && <Empty icon={<Box size={22} />} title="No target models" detail="Download a Hugging Face repo or import a local Transformers checkpoint."
              action={<button className="text-button" onClick={() => setRepo('Qwen/Qwen3.5-0.8B')}>Use Qwen3.5-0.8B <ArrowRight size={14} /></button>} />}
            {data.targets.map(item => <button key={item.id} className="row-button" disabled={running} onClick={() => {
              selectTarget(item)
              setPage('Playground')
            }}>
              <span className="row-icon"><Box size={18} /></span>
              <span className="row-text"><strong>{item.name}</strong><small title={item.source}>{item.source}</small></span>
              <span className="tag">{item.revision?.slice(0, 8) ?? 'Local'}</span>
              <ArrowRight size={16} />
            </button>)}
          </Panel>
        </>}
        {data && page === 'History' && <HistoryView data={data} busy={running}
          commit={(id, message) => update(() => api.commit(id, message), 'Version committed.')}
          exportVersion={commitId => void start({ kind: 'export', commitId })} />}
        {data && page === 'Editor settings' && <EditorForm data={data}
          save={(profile, key) => update(() => api.saveEditor(profile, key), 'Editor connection saved.')} />}
        {job && <Panel title="Job log" icon={<Terminal size={16} />}
          action={<div className="job-actions">
            <span className={`tag ${job.status === 'success' ? 'confirmed' : ''}`}>{job.status}</span>
            {running && <button className="secondary" onClick={() => void api.cancel().catch(err => setError(err.message))}>Cancel</button>}
            <button className="secondary" aria-expanded={logOpen} onClick={() => setLogOpen(!logOpen)}>{logOpen ? 'Hide log' : 'Show log'}</button>
          </div>}>
          {logOpen && <pre className="job-log">{job.log.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').replace(/\r/g, '\n').trim() || 'Waiting for runtime output…'}</pre>}
        </Panel>}
      </div>
      <footer className="statusbar"><span><Cpu size={13} /> {device.toUpperCase()}</span>
        <span>{running ? 'Job running' : 'Local runtime'} <i className={running ? 'busy-dot' : 'local-dot'} /></span></footer>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
