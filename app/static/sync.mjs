/**
 * Keeps a ChatStore in step with the chats the server keeps for the signed-in user. The browser copy is
 * a cache: the higher revision wins, a deletion always wins, and one request is in flight at a time.
 */
import { CHAT_ID, DEFAULT_SETTINGS, FILE_URL, adoptSettings, inProgress, isImage, restoreChat } from './chat.mjs';

export const LEGACY_KEY = 'pubmed-chat.v1';
const VISIBLE_PULL_MS = 5000;
const HIDDEN_PULL_MS = 30000;
const BACKOFF_MS = [5000, 10000, 20000, 30000];
// A device that finished answering uploads its final copy just after its stream ends. For this long a
// chat that is no longer reported live is not treated as interrupted yet, and is pulled more often.
const GRACE_MS = 10000;
const GRACE_PULL_MS = 1500;
const MAX_ROUNDS = 5;

class SyncError extends Error {
  constructor(kind) {
    super(kind);
    this.kind = kind;  // 'offline': try again later; 'unauthorized': the session is gone
  }
}

/** Where the sync cursor for a cache key is kept; user names cannot contain ':', so keys never collide. */
export const syncKey = storageKey => storageKey + ':sync';

/** Remove a user's cached chats and sync state from this browser, e.g. when signing out. */
export function forgetCache(storage, storageKey) {
  for (const key of [storageKey, syncKey(storageKey)]) {
    try { storage.removeItem(key); } catch { /* nothing to remove */ }
  }
}

function eachAttachment(messages, visit) {
  for (const message of messages) {
    for (const branch of message.branches || []) eachAttachment(branch.tail || [], visit);
    for (const file of message.attachments || []) visit(file);
  }
}

const inlineImage = file => isImage(file) && typeof file.dataUrl === 'string' && !file.url;

function hasInlineImages(chat) {
  let found = false;
  eachAttachment(chat.messages, file => { found ||= inlineImage(file); });
  return found;
}

/** A copy without pictures stored inline, for a chat too large for the server; the words are kept. */
function withoutInlineImages(chat) {
  const copy = structuredClone(chat);
  eachAttachment(copy.messages, file => {
    if (inlineImage(file)) {
      delete file.dataUrl;
      file.removed = true;
    }
  });
  return copy;
}

/** Upload a picture; resolves to the attachment fields that point at the server's copy. */
export async function uploadImage(fetcher, blob) {
  const response = await fetcher('/api/files', { method: 'POST', headers: { 'Content-Type': blob.type }, body: blob });
  const data = await response.json().catch(() => null);
  const stored = typeof data?.url === 'string' ? data.url.match(FILE_URL) : null;
  if (!response.ok || !stored) {
    throw Object.assign(new Error(data?.error?.message || 'Upload failed'),
      { status: response.status, code: data?.error?.code });
  }
  return { fileId: stored[1], url: data.url };
}

function dataUrlBlob(dataUrl) {
  const [, type, base64] = dataUrl.match(/^data:([^;,]+);base64,(.*)$/s);
  return new Blob([Uint8Array.from(atob(base64), char => char.charCodeAt(0))], { type });
}

/** Chats this browser saved before history moved to the server, unless they were imported or declined. */
export function legacyChats(storage) {
  try {
    if (storage.getItem(LEGACY_KEY + '.skip') === '1') return [];
    const saved = JSON.parse(storage.getItem(LEGACY_KEY) || 'null');
    return Array.isArray(saved?.chats) ? saved.chats.map(restoreChat).filter(chat => chat?.messages.length) : [];
  } catch { return []; }
}

/** The user declined: the old history is not offered again in this browser. */
export function skipLegacy(storage) {
  try { storage.setItem(LEGACY_KEY + '.skip', '1'); } catch { /* offered again next time */ }
}

/** Keep the old history as a backup under another name, so it is never offered again. */
function retireLegacy(storage, raw) {
  try { storage.removeItem(LEGACY_KEY); } catch { return skipLegacy(storage); }
  try { storage.setItem(LEGACY_KEY + '.imported', raw); } catch {
    // No room for a second copy; the original fits again, as its space was just freed.
    try { storage.setItem(LEGACY_KEY, raw); } catch { /* the chats are in the store now */ }
    skipLegacy(storage);
  }
}

/**
 * Move the pre-server history into the store. Inline pictures are uploaded and replaced by references
 * (one stays inline if its upload fails) and the chats are touched, so the sync uploads them.
 * Resolves to counts { chats, images, uploaded }, or null when there was nothing to import.
 */
export async function importLegacy(store, { fetcher = globalThis.fetch, storage = store.storage,
  onProgress = () => {} } = {}) {
  let raw;
  try { raw = storage.getItem(LEGACY_KEY); } catch { return null; }
  const chats = legacyChats(storage);
  if (!raw || !chats.length) return null;
  const images = [];
  for (const chat of chats) {
    if (!CHAT_ID.test(chat.id)) chat.id = globalThis.crypto.randomUUID();  // the server would refuse the old id
    eachAttachment(chat.messages, file => { if (inlineImage(file)) images.push(file); });
  }
  let uploaded = 0;
  for (const [index, file] of images.entries()) {
    try {
      Object.assign(file, await uploadImage(fetcher, dataUrlBlob(file.dataUrl)));
      delete file.dataUrl;
      uploaded++;
    } catch { /* stays inline, inside the chat */ }
    onProgress(index + 1, images.length);
  }
  store.importChats(chats);
  // The old history carried custom instructions too; keep them unless settings were already chosen.
  const saved = JSON.parse(raw);
  if (saved?.settings && JSON.stringify(store.settings) === JSON.stringify(DEFAULT_SETTINGS)) {
    adoptSettings(store, saved.settings);
  }
  retireLegacy(storage, raw);
  return { chats: chats.length, images: images.length, uploaded };
}

export class RemoteSync {
  /**
   * fetcher: fetch for the API (called without a receiver). document: for page visibility; pulls run
   * every 5 s while visible and every 30 s while hidden. Timers and clock can be replaced for tests.
   * onStatus(status) reports 'synced' | 'syncing' | 'offline' | 'unauthorized'; onNotice(text) reports
   * a problem the user should know about, in Korean.
   */
  constructor(store, { fetcher = globalThis.fetch, storage = store.storage, document = globalThis.document,
    setTimeout: setTimer = globalThis.setTimeout, clearTimeout: clearTimer = globalThis.clearTimeout,
    now = Date.now, pushDelay = 400, onStatus = () => {}, onNotice = () => {} } = {}) {
    this.store = store;
    this.fetcher = fetcher;
    this.storage = storage;
    this.document = document || null;
    this.setTimer = (callback, ms) => setTimer(callback, ms);
    this.clearTimer = timer => { if (timer) clearTimer(timer); };
    this.now = now;
    this.pushDelay = pushDelay;
    this.onStatus = onStatus;
    this.onNotice = onNotice;
    this.stateKey = syncKey(store.storageKey);
    this.seq = 0;  // the server change sequence this cache has caught up to
    this.known = new Map();  // chat id -> the revision the server holds
    this.confirmed = new Set();  // local deletions the server has recorded
    this.refused = new Map();  // chat id -> a revision the server refused; tried again once the chat changes
    this.sentSettings = null;  // JSON of the settings the server holds, once known
    this.settingsDirty = false;  // changed while offline in an earlier visit: these win over the server's
    this.fading = new Map();  // chat id -> when the server stopped reporting it live
    this.status = 'syncing';
    this.ready = false;  // the first pull has finished
    this.error = null;
    this.failures = 0;
    this.running = null;
    this.again = false;
    this.wantPull = false;
    this.wantSettings = true;
    this.stopped = false;
    this.pullTimer = null;
    this.pushTimer = null;
    // Visible again: catch up at once. Hidden: send what is pending now, as the page may be closed next.
    this.onVisibility = () => this.run({ pull: this.document.visibilityState !== 'hidden' });
    this.saved = () => this.changed();
    store.onSaved = this.saved;
    this.loadState();
  }

  /** Chats the server holds for this user, as far as this browser knows. */
  get serverChats() { return this.known.size; }

  /** Pull everything new, sync settings and push local changes; resolves once that first round is done. */
  start() {
    this.document?.addEventListener('visibilitychange', this.onVisibility);
    return this.run({ pull: true });
  }

  stop() {
    this.stopped = true;
    this.clearTimer(this.pullTimer);
    this.clearTimer(this.pushTimer);
    this.document?.removeEventListener('visibilitychange', this.onVisibility);
    if (this.store.onSaved === this.saved) this.store.onSaved = () => {};
  }

  /** Pull and push right away, e.g. when the browser is back online. */
  syncNow() { return this.run({ pull: true }); }

  /** Push pending changes now; resolves to whether the server has all of them. */
  async flush() {
    // A cycle already running may have finished its last round before the newest save.
    for (let attempt = 0; attempt < 3 && !this.stopped; attempt++) {
      await this.run();
      if (this.status !== 'syncing') break;
    }
    return this.status === 'synced';
  }

  /** The store saved: push soon. The answering tab saves every 1.5 s, so other devices follow the answer. */
  changed() {
    if (this.stopped) return;
    this.updateStatus();
    if (this.running) {
      this.again = true;
      return;
    }
    if (this.error === 'offline') return;  // the backoff timer retries; an unreachable server is not hammered
    this.clearTimer(this.pushTimer);
    this.pushTimer = this.setTimer(() => this.run(), this.pushDelay);
  }

  run({ pull = false } = {}) {
    if (this.stopped) return Promise.resolve();
    this.wantPull ||= pull;
    if (this.running) {
      this.again = true;
      return this.running;
    }
    this.clearTimer(this.pushTimer);
    this.running = this.cycle().finally(() => {
      this.running = null;
      this.schedule();
      if (this.again) {  // saved after the last round of this cycle
        this.again = false;
        this.changed();
      }
    });
    return this.running;
  }

  async cycle() {
    try {
      // Rounds repeat while saves arrive meanwhile; the cap keeps an unexpected answer from looping fast.
      for (let round = 0; round === 0 || (this.again && !this.stopped && round < MAX_ROUNDS); round++) {
        this.again = false;
        if (this.wantPull) {
          this.wantPull = false;
          await this.pull();
        }
        if (this.wantSettings) {
          await this.syncSettings();
          this.wantSettings = false;
        }
        await this.pushDeletions();
        await this.pushChats();
        if (this.settingsChanged()) await this.pushSettings();
      }
      this.error = null;
      this.failures = 0;
    } catch (error) {
      if (!(error instanceof SyncError)) console.error(error);
      if (error.kind === 'unauthorized') {
        this.error = 'unauthorized';
        this.stop();
      } else {
        this.error = 'offline';
        this.failures++;
        this.wantPull = true;
      }
    } finally {
      this.saveState();
      this.updateStatus();
    }
  }

  schedule() {
    this.clearTimer(this.pullTimer);
    if (this.stopped) return;
    const delay = this.error === 'offline' ? BACKOFF_MS[Math.min(this.failures, BACKOFF_MS.length) - 1]
      : this.fading.size ? GRACE_PULL_MS
        : this.document?.visibilityState === 'hidden' ? HIDDEN_PULL_MS : VISIBLE_PULL_MS;
    this.pullTimer = this.setTimer(() => this.run({ pull: true }), delay);
  }

  async request(method, url, body) {
    const fetcher = this.fetcher;
    let response;
    try {
      response = await fetcher(url, { method, cache: 'no-store',
        ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body }) });
    } catch { throw new SyncError('offline'); }
    if (response.status === 401) throw new SyncError('unauthorized');
    // A proxy in front of a stopped server answers 5xx (Cloudflare uses 502 and 530): wait and retry.
    if (response.status >= 500 || response.status === 429) throw new SyncError('offline');
    let data = null;
    if (response.status !== 204) {
      data = await response.json().catch(() => undefined);
      if (data === undefined) throw new SyncError('offline');  // not this API answering
    }
    return { ok: response.ok, status: response.status, data };
  }

  async pull() {
    const since = this.seq;
    const { ok, data } = await this.request('GET', '/api/chats?since=' + since);
    if (!ok || !Number.isInteger(data?.seq) || !Array.isArray(data.chats)) throw new SyncError('offline');
    if (data.seq < since) {
      // The server's history was replaced: catch up from the start and upload what only this browser has.
      this.seq = 0;
      this.known.clear();
      this.confirmed.clear();
      return this.pull();
    }
    const chats = data.chats.filter(chat => chat && typeof chat.id === 'string');
    const deleted = Array.isArray(data.deleted) ? data.deleted.filter(chatId => typeof chatId === 'string') : [];
    for (const chat of chats) if (Number.isInteger(chat.rev)) this.known.set(chat.id, chat.rev);
    for (const chatId of deleted) {
      this.known.delete(chatId);
      this.confirmed.add(chatId);
    }
    if (chats.length || deleted.length) {
      this.store.merge({ chats, deleted });
      this.store.emit();
    }
    this.seq = data.seq;
    this.ready = true;
    if (this.applyLive(data.live) || chats.length || deleted.length) await this.store.refreshLive();
  }

  /** Mark the chats the server is answering right now; returns whether that set changed. */
  applyLive(live) {
    const store = this.store;
    const reported = new Set(Array.isArray(live) ? live.filter(chatId => typeof chatId === 'string') : []);
    const next = new Set(reported);
    const now = this.now();
    for (const chatId of store.remoteLive) {
      if (reported.has(chatId)) continue;
      const chat = store.chats.find(item => item.id === chatId);
      const since = this.fading.get(chatId) ?? now;
      if (chat && chat !== store.requestChat && inProgress(chat) && now - since < GRACE_MS) {
        this.fading.set(chatId, since);
        next.add(chatId);
      }
    }
    for (const chatId of [...this.fading.keys()]) {
      if (!next.has(chatId) || reported.has(chatId)) this.fading.delete(chatId);
    }
    const changed = next.size !== store.remoteLive.size || [...next].some(chatId => !store.remoteLive.has(chatId));
    store.remoteLive = next;
    return changed;
  }

  /** Chats with changes the server lacks: the one being answered first, then the open one, then newest. */
  dirty() {
    const store = this.store;
    const rank = chat => (chat === store.requestChat ? 2 : chat.id === store.activeId ? 1 : 0);
    // A chat another tab or device is answering is uploaded by that tab.
    return store.chats.filter(chat => chat.messages.length && CHAT_ID.test(chat.id) && !store.liveChats.has(chat.id)
      && chat.rev > (this.known.get(chat.id) ?? -1) && this.refused.get(chat.id) !== chat.rev)
      .sort((a, b) => rank(b) - rank(a) || b.updatedAt - a.updatedAt);
  }

  async pushChats() {
    const pushed = new Set();
    for (let chat; !this.stopped && (chat = this.dirty().find(item => !pushed.has(item.id)));) {
      pushed.add(chat.id);
      await this.pushChat(chat);
    }
  }

  async pushChat(chat) {
    const store = this.store;
    const rev = chat.rev;
    const path = '/api/chats/' + chat.id;
    let response = await this.request('PUT', path, JSON.stringify(chat));
    if (response.status === 413 && response.data?.error?.code === 'chat_too_large' && hasInlineImages(chat)) {
      // Pictures stored inline (from before uploads, or when one failed) can make a chat too large.
      response = await this.request('PUT', path, JSON.stringify(withoutInlineImages(chat)));
    }
    const { ok, status, data } = response;
    if (ok) {
      this.known.set(chat.id, rev);
      this.refused.delete(chat.id);
    } else if (status === 409 && Number.isInteger(data?.chat?.rev)) {
      // Another device saved a newer copy, and it wins; unless this tab is answering the chat right now.
      this.known.set(chat.id, data.chat.rev);
      store.merge({ chats: [data.chat] });
      const local = store.chats.find(item => item.id === chat.id);
      if (local && local === store.requestChat && local.rev <= data.chat.rev) local.rev = data.chat.rev + 1;
      store.emit();
    } else if (status === 410) {  // deleted on another device: a deletion always wins
      this.known.delete(chat.id);
      this.confirmed.add(chat.id);
      // A chat this tab is answering stays until the answer ends; its next save is refused the same way.
      this.refused.set(chat.id, rev);
      store.merge({ chats: [], deleted: [chat.id] });
      store.emit();
    } else {
      this.refused.set(chat.id, rev);
      const code = data?.error?.code;
      this.onNotice(code === 'storage_quota' ? '서버의 내 저장 공간이 가득 차서 대화를 저장하지 못했어요. 오래된 대화를 지워 주세요.'
        : code === 'chat_too_large' ? '“' + chat.title + '” 대화가 너무 커서 서버에 저장하지 못했어요.'
          : '“' + chat.title + '” 대화를 서버에 저장하지 못했어요.');
    }
  }

  unsentDeletions() {
    return [...this.store.deleted].filter(chatId => !this.confirmed.has(chatId));
  }

  /** Each local deletion is sent once; the server keeps it, so it is remembered as confirmed. */
  async pushDeletions() {
    for (const chatId of this.unsentDeletions()) {
      if (this.stopped) return;
      // Any answer but a retryable failure (thrown) is final: retrying could not change it.
      if (CHAT_ID.test(chatId)) await this.request('DELETE', '/api/chats/' + chatId);
      this.confirmed.add(chatId);
      this.known.delete(chatId);
    }
  }

  settingsChanged() {
    return this.sentSettings !== null && JSON.stringify(this.store.settings) !== this.sentSettings;
  }

  /** At start the server's settings win, unless this browser changed them while it could not send them. */
  async syncSettings() {
    if (!this.settingsDirty) {
      const { ok, data } = await this.request('GET', '/api/settings');
      if (!ok) throw new SyncError('offline');
      if (data?.settings && typeof data.settings === 'object') {
        adoptSettings(this.store, data.settings);
        this.sentSettings = JSON.stringify(this.store.settings);
        return;
      }
    }
    await this.pushSettings();
  }

  async pushSettings() {
    const body = JSON.stringify(this.store.settings);
    const { ok } = await this.request('PUT', '/api/settings', body);
    if (!ok) this.onNotice('설정을 서버에 저장하지 못했어요.');
    this.sentSettings = body;  // a refused copy would be refused again; the next change tries anew
    this.settingsDirty = false;
  }

  pending() {
    return !this.ready || this.wantSettings || this.settingsChanged() || this.unsentDeletions().length > 0
      || this.dirty().length > 0;
  }

  updateStatus() {
    const status = this.error || (this.pending() ? 'syncing' : 'synced');
    if (status === this.status) return;
    this.status = status;
    this.onStatus(status);
  }

  loadState() {
    const store = this.store;
    // Without a cached history the cursor would skip chats this browser never stored.
    const cached = store.deleted.size > 0 || store.chats.some(chat => chat.messages.length);
    try {
      const state = JSON.parse(this.storage.getItem(this.stateKey) || 'null');
      if (cached && Number.isInteger(state?.seq) && state.seq > 0) {
        this.seq = state.seq;
        for (const [chatId, rev] of Object.entries(state.known || {})) {
          if (Number.isInteger(rev)) this.known.set(chatId, rev);
        }
      }
      for (const chatId of Array.isArray(state?.confirmed) ? state.confirmed : []) {
        if (typeof chatId === 'string') this.confirmed.add(chatId);
      }
      this.settingsDirty = state?.settingsDirty === true;
    } catch { /* start with a full pull */ }
  }

  saveState() {
    // If the browser copy could not be written, it lags behind the cursor: catch up from the start next time.
    const behind = this.store.cacheFailed;
    const state = { seq: behind ? 0 : this.seq, known: behind ? {} : Object.fromEntries(this.known),
      confirmed: [...this.confirmed].filter(chatId => this.store.deleted.has(chatId)).slice(-300),
      settingsDirty: this.wantSettings ? this.settingsDirty : this.settingsChanged() };
    try { this.storage.setItem(this.stateKey, JSON.stringify(state)); } catch { /* an older cursor is safe */ }
  }
}
