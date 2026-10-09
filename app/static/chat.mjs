const STORAGE_KEY = 'pubmed-chat.v1';
const MODELS = ['qwen3.5-2b', 'qwen3-0.6b'];
export const MODEL_LABELS = { 'qwen3.5-2b': 'Qwen3.5 · 2B', 'qwen3-0.6b': 'Qwen3 · 0.6B' };
/** Small local models have a few thousand tokens of context, so attachments are capped. */
export const ATTACHMENT_LIMIT = 12000;
const DEFAULT_FILE_PROMPT = '첨부한 파일의 내용을 요약해 줘.';
export const DEFAULT_SETTINGS = Object.freeze({ systemPrompt: '', temperature: 0.2, maxTokens: 512 });
const id = () => globalThis.crypto.randomUUID();
const freshChat = () => ({ id: id(), title: '새 대화', model: MODELS[0],
  usePubmed: true, thinking: false, messages: [], updatedAt: Date.now() });
/** Reasoning spends tokens before the answer starts, so thinking requests get a larger floor. */
export const THINKING_MIN_TOKENS = 1024;
const STOPPED = '답변 생성을 중지했습니다.';

function normalizeTools(calls) {
  return Array.isArray(calls) ? calls.filter(call => call && typeof call.name === 'string')
    .map(call => ({ name: call.name,
      arguments: call.arguments && typeof call.arguments === 'object' ? call.arguments : {},
      ...(typeof call.error === 'string' ? { error: call.error } : {}) })) : [];
}

const CITATIONS = '\n\nPubMed 검색·조회 출처:\n';

/** Keep only well-formed PubMed sources; links are rebuilt from the PMID rather than trusted. */
export function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];
  const text = (value, limit) => typeof value === 'string' && value.trim() ? value.trim().slice(0, limit) : undefined;
  return sources.filter(item => item && /^[0-9]+$/.test(item.pmid)).slice(0, 20).map(item => Object.fromEntries(
    Object.entries({ pmid: item.pmid, url: 'https://pubmed.ncbi.nlm.nih.gov/' + item.pmid + '/',
      title: text(item.title, 300), authors: text(item.authors, 200), journal: text(item.journal, 120),
      year: text(item.year, 4), abstractFetched: item.abstractFetched === true || item.abstract_fetched === true })
      .filter(([, value]) => value !== undefined)));
}

/** The answer without the server's plain citation list, for when source cards show the same links. */
export function withoutCitationList(content) {
  const index = content.lastIndexOf(CITATIONS);
  return index === -1 ? content : content.slice(0, index);
}

function restoreAttachments(files) {
  return Array.isArray(files) ? files.filter(file => file && typeof file.name === 'string'
    && typeof file.text === 'string').slice(0, 5).map(file => ({ name: file.name.slice(0, 200),
    text: file.text.slice(0, ATTACHMENT_LIMIT), truncated: file.truncated === true })) : [];
}

/** What the model sees for a user turn: the question followed by each attached file. */
export function messageText(message) {
  const files = message.attachments || [];
  return [message.content, ...files.map(file => '[첨부 파일: ' + file.name
    + (file.truncated ? ' (앞부분만)' : '') + ']\n```\n' + file.text + '\n```')].join('\n\n');
}

/** Read text files for attaching; the total across files stays within ATTACHMENT_LIMIT. */
export async function readAttachments(files, used = 0) {
  const result = [];
  let budget = ATTACHMENT_LIMIT - used;
  for (const file of files) {
    if (budget <= 0) break;
    const text = (await file.text()).replace(/\r\n/g, '\n');
    if (text.includes('\u0000')) throw new Error(file.name + '은(는) 텍스트 파일이 아닙니다.');
    result.push({ name: file.name, text: text.slice(0, budget), truncated: text.length > budget });
    budget -= Math.min(text.length, budget);
  }
  return result;
}

export function chatMarkdown(chat) {
  const lines = ['# ' + chat.title, ''];
  for (const message of chat.messages) {
    if (message.role === 'user') lines.push('## 나', '', messageText(message), '');
    else lines.push('## ' + (MODEL_LABELS[message.model] || 'PubMed Chat'), '', message.content, '');
  }
  return lines.join('\n');
}

const MAX_VERSIONS = 10;
const VERSION_FIELDS = ['content', 'toolCalls', 'truncated', 'stopped', 'reasoning', 'thinkMs', 'model', 'dropped', 'sources', 'tokens', 'elapsedMs'];

/** The parts of an answer that differ between regenerated versions. */
function snapshot(message) {
  return Object.fromEntries(VERSION_FIELDS.filter(key => message[key] !== undefined).map(key => [key, message[key]]));
}

function restoreVersions(versions) {
  if (!Array.isArray(versions)) return [];
  return versions.filter(item => item && typeof item.content === 'string').slice(-MAX_VERSIONS)
    .map(item => snapshot({ ...item, toolCalls: normalizeTools(item.toolCalls), truncated: item.truncated === true,
      stopped: item.stopped === true, reasoning: typeof item.reasoning === 'string' && item.reasoning ? item.reasoning : undefined,
      thinkMs: Number.isFinite(item.thinkMs) ? item.thinkMs : undefined, model: MODELS.includes(item.model) ? item.model : undefined,
      dropped: Number.isInteger(item.dropped) && item.dropped > 0 ? item.dropped : undefined,
      sources: normalizeSources(item.sources).length ? normalizeSources(item.sources) : undefined,
      tokens: Number.isInteger(item.tokens) && item.tokens >= 0 ? item.tokens : undefined,
      elapsedMs: Number.isFinite(item.elapsedMs) && item.elapsedMs >= 0 ? item.elapsedMs : undefined }));
}

function restoreSettings(value) {
  const number = (input, min, max, fallback) =>
    Number.isFinite(input) && input >= min && input <= max ? input : fallback;
  return { systemPrompt: typeof value?.systemPrompt === 'string' ? value.systemPrompt.slice(0, 4000) : '',
    temperature: number(value?.temperature, 0, 2, DEFAULT_SETTINGS.temperature),
    maxTokens: Number.isInteger(value?.maxTokens)
      ? number(value.maxTokens, 1, 2048, DEFAULT_SETTINGS.maxTokens) : DEFAULT_SETTINGS.maxTokens };
}

/** Validate a stored chat. In-progress requests are kept as-is; settleInterrupted() decides if they died. */
function restoreChat(chat) {
  if (!chat || typeof chat.id !== 'string' || typeof chat.title !== 'string'
      || !Array.isArray(chat.messages)) return null;
  const messages = chat.messages.filter(message => message && typeof message.id === 'string'
    && ['user', 'assistant'].includes(message.role) && typeof message.content === 'string')
    .map(message => ({ id: message.id, role: message.role, content: message.content,
      toolCalls: normalizeTools(message.toolCalls), truncated: message.truncated === true,
      stopped: message.stopped === true,
      ...(message.role === 'assistant' && message.streaming === true ? { streaming: true } : {}),
      ...(Number.isInteger(message.dropped) && message.dropped > 0 ? { dropped: message.dropped } : {}),
      ...(typeof message.reasoning === 'string' && message.reasoning ? { reasoning: message.reasoning } : {}),
      ...(Number.isFinite(message.thinkMs) && message.thinkMs >= 0 ? { thinkMs: message.thinkMs } : {}),
      ...(MODELS.includes(message.model) ? { model: message.model } : {}),
      ...(normalizeSources(message.sources).length ? { sources: normalizeSources(message.sources) } : {}),
      ...(Number.isInteger(message.tokens) && message.tokens >= 0 ? { tokens: message.tokens } : {}),
      ...(Number.isFinite(message.elapsedMs) && message.elapsedMs >= 0 ? { elapsedMs: message.elapsedMs } : {}),
      ...(message.role === 'assistant' && restoreVersions(message.versions).length > 1 ? {
        versions: restoreVersions(message.versions),
        version: Math.min(Math.max(Number.isInteger(message.version) ? message.version : 0, 0),
          restoreVersions(message.versions).length - 1) } : {}),
      ...(message.role === 'user' && restoreAttachments(message.attachments).length
        ? { attachments: restoreAttachments(message.attachments) } : {}),
      ...(typeof message.error === 'string' ? { error: message.error } : {}),
      ...(message.role === 'user' && message.pending === true ? { pending: true } : {}) }));
  return { id: chat.id, title: chat.title, model: MODELS.includes(chat.model) ? chat.model : MODELS[0],
    usePubmed: chat.usePubmed !== false, thinking: chat.thinking === true, pinned: chat.pinned === true, messages,
    updatedAt: Number(chat.updatedAt) || Date.now(),
    rev: Number.isFinite(chat.rev) ? chat.rev : Number(chat.updatedAt) || 0 };
}

const inProgress = chat => chat.messages.some(message => message.pending || message.streaming);
const LOCK_PREFIX = 'pubmed-chat:';

/** Turn a request whose tab is gone into a stopped answer or a retryable question; true if changed. */
function settleInterrupted(chat) {
  if (!inProgress(chat)) return false;
  // A draft that never received text was interrupted before it said anything.
  chat.messages = chat.messages.filter(message => !(message.streaming && !message.content.trim()));
  chat.messages.forEach((message, index) => {
    if (message.streaming) {
      delete message.streaming;
      message.stopped = true;
    }
    if (message.pending) {
      delete message.pending;
      if (chat.messages[index + 1]?.role !== 'assistant') {
        message.error = '페이지가 닫혀 응답을 받지 못했습니다. 다시 시도해 주세요.';
      }
    }
  });
  return true;
}

function errorMessage(error, status) {
  const hints = {
    context_overflow: '대화가 모델의 길이 한도를 넘었습니다. 새 대화를 시작하거나 질문을 줄여 주세요.',
    model_missing: '모델 파일이 없습니다. 서버에서 모델 준비를 완료해 주세요.',
    device_unavailable: '설정된 실행 장치를 사용할 수 없습니다.',
    pubmed_timeout: 'PubMed 연결 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.',
    pubmed_unavailable: 'PubMed에 연결할 수 없습니다. 잠시 후 다시 시도하거나 PubMed 검색을 꺼 주세요.',
    tool_limit: '논문 검색 단계가 너무 많아 중단됐습니다. 요청을 더 구체적으로 줄여 주세요.',
  };
  if (hints[error?.code]) return hints[error.code];
  if (status === 413) return hints.context_overflow;
  if (status === 504) return hints.pubmed_timeout;
  if (status === 502) return 'PubMed 검색을 완료하지 못했습니다. 다시 시도하거나 PubMed 검색을 꺼 주세요.';
  return typeof error?.message === 'string' ? error.message : '응답을 받지 못했습니다. 다시 시도해 주세요.';
}

/** Yields the JSON payload of each server-sent event in a fetch response body. */
export async function* readEvents(body) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value.replace(/\r\n/g, '\n');
      let end;
      while ((end = buffer.indexOf('\n\n')) !== -1) {
        const data = buffer.slice(0, end).split('\n').filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).trimStart()).join('\n');
        buffer = buffer.slice(end + 2);
        if (data) yield JSON.parse(data);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function finish(draft, data) {
  const choice = data.choices?.[0];
  const content = choice?.message?.content;
  if (typeof content !== 'string') throw new Error('빈 응답을 받았습니다. 다시 시도해 주세요.');
  draft.content = content;
  draft.toolCalls = normalizeTools(data.tool_calls);
  if (normalizeSources(data.sources).length) draft.sources = normalizeSources(data.sources);
  if (Number.isInteger(data.usage?.completion_tokens)) draft.tokens = data.usage.completion_tokens;
  draft.truncated = choice.finish_reason === 'length';
  draft.stopped = choice.finish_reason === 'cancelled';
  if (typeof choice.message.reasoning_content === 'string' && choice.message.reasoning_content) {
    draft.reasoning = choice.message.reasoning_content;
  }
  if (Number.isInteger(data.dropped_messages) && data.dropped_messages > 0) draft.dropped = data.dropped_messages;
  if (!content.trim() && !draft.stopped && !draft.reasoning) throw new Error('빈 응답을 받았습니다. 다시 시도해 주세요.');
}

export class ChatStore {
  /**
   * locks: the Web Locks API. The answering tab holds a lock per chat, so other tabs can tell a request
   * that is still running elsewhere from one whose tab was closed. Without it, requests found in storage
   * are treated as interrupted.
   */
  constructor(storage = globalThis.localStorage, fetcher = globalThis.fetch, { locks = globalThis.navigator?.locks } = {}) {
    this.storage = storage;
    this.fetcher = fetcher;
    this.locks = locks || null;
    this.liveChats = new Set();  // chats another tab is answering right now
    this.busy = false;
    this.controller = null;
    this.notice = '';
    this.onChange = () => {};
    this.storageWarning = '';
    this.chats = [];
    this.deleted = new Set();
    this.requestChat = null;
    this.settings = { ...DEFAULT_SETTINGS };
    try {
      const saved = JSON.parse(storage.getItem(STORAGE_KEY) || 'null');
      if (Array.isArray(saved?.chats)) this.chats = saved.chats.map(restoreChat).filter(Boolean);
      if (Array.isArray(saved?.deleted)) this.deleted = new Set(saved.deleted.filter(item => typeof item === 'string'));
      this.settings = restoreSettings(saved?.settings);
      this.activeId = this.chats.some(chat => chat.id === saved?.activeId)
        ? saved.activeId : this.chats[0]?.id;
    } catch {
      this.storageWarning = '대화 기록을 불러오지 못했습니다. 새 대화로 시작합니다.';
    }
    if (!this.chats.length) {
      this.chats = [freshChat()];
      this.activeId = this.chats[0].id;
    }
    if (this.locks) this.chats.filter(inProgress).forEach(chat => this.liveChats.add(chat.id));  // until checked
    else this.chats.forEach(settleInterrupted);
  }

  get current() { return this.chats.find(chat => chat.id === this.activeId); }

  /** Whether this tab may change a chat: not while it is answering, nor while another tab answers it. */
  canModify(chat = this.current) {
    return Boolean(chat) && !this.busy && !this.liveChats.has(chat.id);
  }

  touch(chat) {
    chat.rev = Math.max(Date.now(), (chat.rev || 0) + 1);
  }

  /** Ask the browser which chats another tab is still answering; settle the rest. */
  async refreshLive() {
    if (!this.locks) return;
    let held;
    try { held = new Set((await this.locks.query()).held.map(lock => lock.name)); } catch { return; }
    let changed = false;
    const live = new Set();
    for (const chat of this.chats) {
      if (chat === this.requestChat || !inProgress(chat)) continue;
      if (held.has(LOCK_PREFIX + chat.id)) live.add(chat.id);
      else if (settleInterrupted(chat)) {
        this.touch(chat);
        changed = true;
      }
    }
    changed ||= live.size !== this.liveChats.size || [...live].some(chatId => !this.liveChats.has(chatId));
    this.liveChats = live;
    if (changed) this.emit();
  }

  /**
   * Fold in what other tabs saved: their new chats, newer versions of idle chats and deletions.
   * The chat this tab is answering keeps its in-memory object, which the request still writes to.
   */
  merge(saved, { adoptSettings = false } = {}) {
    if (!saved || !Array.isArray(saved.chats)) return;
    if (Array.isArray(saved.deleted)) saved.deleted.forEach(item => typeof item === 'string' && this.deleted.add(item));
    for (const stored of saved.chats.map(restoreChat).filter(Boolean)) {
      if (this.deleted.has(stored.id)) continue;
      const index = this.chats.findIndex(chat => chat.id === stored.id);
      if (index === -1) {
        this.chats.push(stored);
        if (inProgress(stored)) this.liveChats.add(stored.id);  // refreshLive() confirms or settles it
      } else if (stored.rev > this.chats[index].rev && this.chats[index] !== this.requestChat) {
        this.chats[index] = stored;
        if (inProgress(stored)) this.liveChats.add(stored.id);
        else this.liveChats.delete(stored.id);
      }
    }
    if (this.requestChat && this.deleted.has(this.requestChat.id)) this.deleted.delete(this.requestChat.id);
    this.chats = this.chats.filter(chat => !this.deleted.has(chat.id));
    if (!this.chats.length) this.chats = [freshChat()];
    if (!this.current) this.activeId = this.chats[0].id;
    if (adoptSettings) this.settings = restoreSettings(saved.settings);
  }

  /** Called when another tab wrote to storage. */
  sync() {
    try {
      this.merge(JSON.parse(this.storage.getItem(STORAGE_KEY) || 'null'), { adoptSettings: true });
    } catch { return; }
    this.onChange();
    return this.refreshLive();
  }

  save() {
    try {
      try { this.merge(JSON.parse(this.storage.getItem(STORAGE_KEY) || 'null')); } catch { /* unreadable: overwrite */ }
      this.storage.setItem(STORAGE_KEY, JSON.stringify({ chats: this.chats, activeId: this.activeId,
        settings: this.settings, deleted: [...this.deleted].slice(-300) }));
    } catch {
      this.storageWarning = '브라우저에 기록을 저장할 수 없습니다. 페이지를 닫으면 대화가 사라질 수 있습니다.';
    }
  }

  emit() { this.save(); this.onChange(); }

  newChat() {
    if (this.busy) return;
    const empty = this.chats.find(chat => !chat.messages.length);
    if (empty) this.activeId = empty.id;
    else {
      const chat = freshChat();
      this.chats.unshift(chat);
      this.activeId = chat.id;
    }
    this.emit();
  }

  select(chatId) {
    if (this.chats.some(chat => chat.id === chatId)) {
      this.activeId = chatId;
      this.emit();
    }
  }

  rename(chatId, title) {
    const chat = this.chats.find(item => item.id === chatId);
    const clean = String(title).replace(/\s+/g, ' ').trim().slice(0, 80);
    if (chat && clean && this.canModify(chat)) {
      chat.title = clean;
      this.touch(chat);
      this.emit();
    }
  }

  togglePin(chatId) {
    const chat = this.chats.find(item => item.id === chatId);
    if (!chat || !this.canModify(chat)) return;
    chat.pinned = !chat.pinned;
    this.touch(chat);
    this.emit();
  }

  /** Per-chat options set from the UI: model, usePubmed, thinking. */
  updateChat(patch) {
    const chat = this.current;
    if (!this.canModify(chat)) return;
    for (const key of ['model', 'usePubmed', 'thinking']) {
      if (!(key in patch)) continue;
      if (key === 'model' && !MODELS.includes(patch.model)) continue;
      chat[key] = key === 'model' ? patch.model : patch[key] === true;
    }
    this.touch(chat);
    this.emit();
  }

  remove(chatId) {
    const chat = this.chats.find(item => item.id === chatId);
    if (this.busy || (chat && !this.canModify(chat))) return;
    this.deleted.add(chatId);
    this.chats = this.chats.filter(chat => chat.id !== chatId);
    if (!this.chats.length) this.chats = [freshChat()];
    if (!this.chats.some(chat => chat.id === this.activeId)) this.activeId = this.chats[0].id;
    this.emit();
  }

  updateSettings(patch) {
    this.settings = restoreSettings({ ...this.settings, ...patch });
    this.emit();
  }

  async send(content, attachments = []) {
    const files = restoreAttachments(attachments);
    const chat = this.current;
    if (!this.canModify(chat) || (!content.trim() && !files.length)) return;
    const message = { id: id(), role: 'user', content: content.trim() || DEFAULT_FILE_PROMPT,
      ...(files.length ? { attachments: files } : {}) };
    chat.messages.push(message);
    if (chat.title === '새 대화') chat.title = message.content.replace(/\s+/g, ' ').slice(0, 36);
    return this.request(chat, message);
  }

  canRetry(messageId) {
    const message = this.current.messages.at(-1);
    return this.canModify() && message?.id === messageId && message.role === 'user' && Boolean(message.error);
  }

  async retry(messageId) {
    if (this.canRetry(messageId)) return this.request(this.current, this.current.messages.at(-1));
  }

  canRegenerate(messageId) {
    const messages = this.current.messages;
    return this.canModify() && messages.at(-1)?.id === messageId && messages.at(-1).role === 'assistant'
      && messages.at(-2)?.role === 'user';
  }

  /** Ask again, keeping earlier answers as versions; if the new attempt fails the old answer returns. */
  async regenerate(messageId) {
    if (!this.canRegenerate(messageId)) return;
    const chat = this.current;
    const previous = chat.messages.pop();
    const versions = previous.versions?.length ? previous.versions : [snapshot(previous)];
    return this.request(chat, chat.messages.at(-1), { versions, fallback: previous });
  }

  canSelectVersion(messageId) {
    const message = this.current.messages.at(-1);
    return this.canModify() && message?.id === messageId && (message.versions?.length || 0) > 1;
  }

  selectVersion(messageId, index) {
    if (!this.canSelectVersion(messageId)) return;
    const message = this.current.messages.at(-1);
    const chosen = message.versions[index];
    if (!chosen) return;
    for (const key of VERSION_FIELDS) delete message[key];
    Object.assign(message, { toolCalls: [], truncated: false, stopped: false }, structuredClone(chosen));
    message.version = index;
    this.touch(this.current);
    this.emit();
  }

  /** Replace a question and discard everything after it, as answers to the old wording no longer apply. */
  async edit(messageId, content) {
    const chat = this.current;
    const index = chat.messages.findIndex(message => message.id === messageId);
    if (!this.canModify(chat) || !content.trim() || chat.messages[index]?.role !== 'user') return;
    const message = chat.messages[index];
    chat.messages.splice(index + 1);
    message.content = content.trim();
    return this.request(chat, message);
  }

  canContinue(messageId) {
    const message = this.current.messages.at(-1);
    return this.canModify() && message?.id === messageId && message.role === 'assistant'
      && (message.truncated || message.stopped) && Boolean(message.content.trim())
      && this.current.messages.at(-2)?.role === 'user';
  }

  /** Extend a cut-off answer in place; if that fails the answer is left exactly as it was. */
  async continueAnswer(messageId) {
    if (!this.canContinue(messageId)) return;
    const chat = this.current;
    const target = chat.messages.at(-1);
    return this.request(chat, chat.messages.at(-2), { continueFrom: target });
  }

  stop() { this.controller?.abort(); }

  async request(chat, message, { versions = null, fallback = null, continueFrom = null } = {}) {
    this.busy = true;
    this.requestChat = chat;
    let releaseLock = () => {};
    if (this.locks) {
      // Hold the chat's lock before announcing the request, so no tab can mistake it for a dead one.
      const granted = await new Promise(resolve => {
        this.locks.request(LOCK_PREFIX + chat.id, { ifAvailable: true }, lock => {
          if (!lock) return resolve(false);
          return new Promise(release => { releaseLock = release; resolve(true); });
        }).catch(() => resolve(true));
      });
      if (!granted) {
        this.busy = false;
        this.requestChat = null;
        this.notice = '다른 탭에서 이 대화의 답변을 만들고 있어요.';
        if (fallback) chat.messages.push(fallback);
        this.onChange();
        return;
      }
    }
    delete message.error;
    message.pending = true;
    chat.updatedAt = Date.now();
    this.touch(chat);
    const history = chat.messages.filter(item => !item.error && item.content.trim())
      .map(item => ({ role: item.role, content: item.role === 'user' ? messageText(item) : item.content }));
    if (this.settings.systemPrompt.trim()) {
      history.unshift({ role: 'system', content: this.settings.systemPrompt.trim() });
    }
    const base = continueFrom ? { ...snapshot(continueFrom), content: continueFrom.content } : null;
    let draft;
    if (continueFrom) {
      draft = continueFrom;  // stream straight into the answer being extended
      delete draft.truncated;
      delete draft.stopped;
      draft.streaming = true;
    } else {
      draft = { id: id(), role: 'assistant', content: '', toolCalls: [], streaming: true, model: chat.model };
      chat.messages.push(draft);
    }
    const controller = new AbortController();
    this.controller = controller;
    let finished = false;
    const started = Date.now();
    let shared = started;
    this.emit();
    try {
      const fetcher = this.fetcher;
      const response = await fetcher('/v1/chat/completions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ model: chat.model, messages: history,
          max_tokens: chat.thinking && !continueFrom ? Math.max(this.settings.maxTokens, THINKING_MIN_TOKENS)
            : this.settings.maxTokens,
          temperature: this.settings.temperature, use_pubmed: chat.usePubmed && !continueFrom, stream: true,
          truncate_history: true, ...(chat.thinking && !continueFrom ? { thinking: true } : {}),
          ...(continueFrom ? { continue_final: true } : {}) }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(errorMessage(data.error, response.status));
      }
      const complete = data => {
        finish(draft, data);
        if (base) {  // the server returns only the added text; keep what the answer already had
          draft.content = base.content + draft.content;
          for (const key of ['sources', 'reasoning', 'thinkMs', 'dropped']) if (draft[key] === undefined && base[key] !== undefined) draft[key] = base[key];
          if (!draft.toolCalls.length) draft.toolCalls = base.toolCalls || [];
          draft.tokens = (base.tokens || 0) + (draft.tokens || 0);
        }
        finished = true;
      };
      if (!(response.headers?.get('content-type') || '').includes('text/event-stream')) {
        complete(await response.json());
      } else {
        for await (const event of readEvents(response.body)) {
          if (event.type === 'reasoning') {
            draft.thinkStart ??= Date.now();
            draft.reasoning = (draft.reasoning || '') + event.content;
          } else if (event.type === 'delta') {
            if (draft.thinkStart && draft.thinkMs === undefined) draft.thinkMs = Date.now() - draft.thinkStart;
            draft.content += event.content;
          }
          else if (event.type === 'status') draft.stage = event.stage;
          else if (event.type === 'tool') {
            if (event.status === 'running') {
              draft.content = '';  // text before a tool call is superseded by the grounded answer
              if (draft.reasoning) draft.reasoning += '\n\n';
              draft.toolCalls.push({ name: event.name, arguments: event.arguments || {}, running: true });
            } else {
              const call = draft.toolCalls.findLast(item => item.running && item.name === event.name);
              if (call) delete call.running;
              if (call && event.error) call.error = event.error;
            }
          } else if (event.type === 'done') {
            complete(event);
          } else if (event.type === 'error') throw new Error(errorMessage(event.error, event.status));
          if (Date.now() - shared > 1500) {  // let other tabs follow the answer as it grows
            shared = Date.now();
            this.touch(chat);
            this.save();
          }
          this.onChange();
        }
        if (!finished) throw new Error('응답이 중간에 끊겼습니다. 다시 시도해 주세요.');
      }
    } catch (error) {
      if (finished) { /* stopped after the complete answer arrived: keep it as finished */ }
      else if (base && !(controller.signal.aborted && draft.content.length > base.content.length)) {
        for (const key of VERSION_FIELDS) delete draft[key];
        Object.assign(draft, base);  // a failed continuation leaves the answer as it was
        if (!controller.signal.aborted) this.notice = error instanceof TypeError
          ? '서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하고 다시 시도해 주세요.'
          : error.message || '이어서 생성하지 못했습니다.';
      } else if (controller.signal.aborted && draft.content.trim()) draft.stopped = true;
      else {
        chat.messages.splice(chat.messages.indexOf(draft), 1);
        const reason = controller.signal.aborted ? STOPPED : error instanceof TypeError
          ? '서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하고 다시 시도해 주세요.'
          : error.message || '응답을 받지 못했습니다. 다시 시도해 주세요.';
        if (fallback) {
          chat.messages.push(fallback);  // a failed regeneration must not cost the answer we had
          this.notice = reason;
        } else message.error = reason;
      }
    } finally {
      if (draft.thinkStart && draft.thinkMs === undefined) draft.thinkMs = Date.now() - draft.thinkStart;
      if (finished) draft.elapsedMs = Date.now() - started + (base?.elapsedMs || 0);
      delete draft.thinkStart;
      delete draft.streaming;
      delete draft.stage;
      draft.toolCalls = normalizeTools(draft.toolCalls.map(({ running, ...call }) => call));
      if (versions && chat.messages.includes(draft)) {
        draft.versions = [...versions, snapshot(draft)].slice(-MAX_VERSIONS);
        draft.version = draft.versions.length - 1;
      }
      if (base && draft.versions?.length) draft.versions[draft.version] = snapshot(draft);
      delete message.pending;
      this.busy = false;
      this.requestChat = null;
      this.controller = null;
      chat.updatedAt = Date.now();
      this.touch(chat);
      this.emit();
      releaseLock();
    }
  }
}

export const escapeHtml = value => String(value).replace(/[&<>"']/g,
  char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

const TRAILING = /(?:&quot;|&#39;|&gt;|[.,;:!?)\]])+$/;

function inlineMarkdown(text) {
  // Escape first; only our own tags and validated HTTP(S) links may enter the DOM.
  return escapeHtml(text).replace(
    /`([^`\n]+)`|\[([^\]\n]+)\]\(([^\s)]+)\)|(https?:\/\/[^\s<]+)|\*\*([^*\n]+)\*\*|\*(?!\s)([^*\n]+?)(?<!\s)\*|~~([^~\n]+)~~/g,
    (match, code, label, target, url, bold, italic, strike) => {
      if (code !== undefined) return '<code>' + code + '</code>';
      if (strike !== undefined) return '<del>' + strike + '</del>';
      if (bold !== undefined) return '<strong>' + bold + '</strong>';
      if (italic !== undefined) return '<em>' + italic + '</em>';
      if (url !== undefined) {
        const tail = url.match(TRAILING)?.[0] || '';
        const href = url.slice(0, url.length - tail.length);
        return '<a href="' + href + '" target="_blank" rel="noopener noreferrer">' + href + '</a>' + tail;
      }
      if (!/^https?:\/\//i.test(target)) return label;
      return '<a href="' + target + '" target="_blank" rel="noopener noreferrer">' + label + '</a>';
    });
}

const tableCells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
const isTableRule = line => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line);

function renderTable(rows) {
  const [head, , ...body] = rows.map(tableCells);
  const cells = (row, tag) => row.map(cell => '<' + tag + '>' + inlineMarkdown(cell) + '</' + tag + '>').join('');
  return '<div class="table-wrap"><table><thead><tr>' + cells(head, 'th') + '</tr></thead><tbody>'
    + body.map(row => '<tr>' + cells(row, 'td') + '</tr>').join('') + '</tbody></table></div>';
}

function renderBlocks(part) {
  const lines = part.split('\n');
  const lists = [];  // open lists, innermost last; each leaves its current <li> open for nesting
  let output = '';
  const closeLists = (indent = -1) => {
    while (lists.length && lists.at(-1).indent > indent) output += '</li></' + lists.pop().type + '>';
  };
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (line.includes('|') && isTableRule(lines[index + 1] || '')) {
      closeLists();
      const rows = [line, lines[index + 1]];
      index += 2;
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) rows.push(lines[index++]);
      index--;
      output += renderTable(rows);
      continue;
    }
    const bullet = line.match(/^(\s*)(?:([-*+])|(\d+)\.)\s+(.+)/);
    if (bullet && !/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      const indent = bullet[1].replace(/\t/g, '    ').length;
      const type = bullet[3] === undefined ? 'ul' : 'ol';
      closeLists(indent);
      const top = lists.at(-1);
      if (top?.indent === indent && top.type === type) output += '</li>';
      else {
        if (top?.indent === indent) output += '</li></' + lists.pop().type + '>';  // list type changed
        // Numbered items split by blank lines would otherwise restart at 1.
        const start = type === 'ol' && Number(bullet[3]) > 1 ? ' start="' + Number(bullet[3]) + '"' : '';
        output += '<' + type + start + '>';
        lists.push({ type, indent });
      }
      output += '<li>' + inlineMarkdown(bullet[4]);
      continue;
    }
    if (lists.length && /^\s{2,}\S/.test(line)) {  // an indented continuation of the current item
      output += '<br>' + inlineMarkdown(line.trim());
      continue;
    }
    if (!line.trim()) {
      if (lists.length && /^\s*(?:[-*+]|\d+\.)\s+/.test(lines.slice(index + 1).find(next => next.trim()) || '')) continue;
      closeLists();
      continue;
    }
    closeLists();
    const heading = line.match(/^(#{1,6})\s+(.+)/);
    const quote = line.match(/^\s*>\s?(.*)/);
    if (heading) {
      const tag = heading[1].length <= 2 ? 'h3' : 'h4';
      output += '<' + tag + '>' + inlineMarkdown(heading[2]) + '</' + tag + '>';
    } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) output += '<hr>';
    else if (quote) output += '<blockquote>' + inlineMarkdown(quote[1]) + '</blockquote>';
    else output += '<p>' + inlineMarkdown(line) + '</p>';
  }
  closeLists();
  return output;
}

export function renderMarkdown(text) {
  return String(text).split(/(```[\s\S]*?(?:```|$))/g).map(part => {
    if (!part.startsWith('```')) return renderBlocks(part);
    const language = part.match(/^```([\w+#.-]*)/)?.[1] || '';
    const code = part.replace(/^```[^\n]*\n?/, '').replace(/```$/, '').replace(/\n$/, '');
    return '<div class="code-block"><div class="code-header"><span>' + escapeHtml(language || 'code')
      + '</span><button type="button" class="code-copy">복사</button></div><pre><code>'
      + escapeHtml(code) + '</code></pre></div>';
  }).join('');
}
