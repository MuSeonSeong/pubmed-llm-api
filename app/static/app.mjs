import { ATTACHMENT_LIMIT, ChatStore, MAX_ATTACHMENTS, MAX_IMAGES, attachmentTokenBudget, chatMarkdown, estimateTokens,
  imageSource, isImage, modelInfo, modelLabel, modelList, readAttachments, renderMarkdown, revealItem, serverModels,
  serverState, withoutCitationList } from './chat.mjs';
import { RemoteSync, forgetCache, importLegacy, legacyChats, skipLegacy, uploadImage } from './sync.mjs';

const $ = id => document.getElementById(id);
const icon = name => {
  const element = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  element.classList.add('icon');
  element.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#i-' + name);
  element.append(use);
  return element;
};
const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const iconButton = (name, label, onClick) => {
  const button = element('button', 'icon-button');
  button.type = 'button';
  button.title = label;
  button.setAttribute('aria-label', label);
  button.append(icon(name));
  button.addEventListener('click', onClick);
  return button;
};

let storage;
try { storage = window.localStorage; }
catch { storage = { getItem: () => null, setItem: () => { throw Error('Storage unavailable'); }, removeItem: () => {} }; }

let leaving = false;
/** The server answers 401 once a session expired or was signed out: sign in again. */
function signIn() {
  if (leaving) return;
  leaving = true;
  location.replace('/login');
}

/** Who is signed in decides whose history this page shows, so nothing starts before the server says. */
async function loadSession() {
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await fetch('/api/session', { cache: 'no-store' });
      const data = await response.json();
      if (response.ok && typeof data?.auth_enabled === 'boolean') return data;
    } catch { /* tried again below */ }
    setupNotice(['서버에 연결할 수 없습니다. 다시 연결하는 중이에요…']);
    await new Promise(resolve => setTimeout(resolve, Math.min(2000 * 2 ** attempt, 30000)));
  }
}

const session = await loadSession();
setupNotice(null);
if (session.auth_enabled && !session.user) {
  signIn();
  await new Promise(() => {});  // the page is being replaced
}
const accounts = session.auth_enabled;  // false: one implicit user on this PC, no sign-in
const userName = typeof session.user?.name === 'string' ? session.user.name : 'local';

/** fetch for this server's API; any 401 (sync, answers, uploads, PDFs) leads to the sign-in page. */
async function api(url, init) {
  const response = await fetch(url, init);
  if (response.status === 401) signIn();
  return response;
}

/** The models the server offers, with their labels and states; checkServer keeps them current. */
const modelSync = serverModels(api);
await modelSync.load();  // before the store, so the chats it restores and starts see this server's models

const store = new ChatStore(storage, api, { autoTitles: true, storageKey: 'pubmed-chat.v2:' + userName,
  cacheOnly: true });
const sync = new RemoteSync(store, { fetcher: api, storage, onStatus: showSyncStatus, onNotice: text => toast(text) });
const prompt = $('prompt');
const conversation = $('conversation');
const THEME_KEY = 'pubmed-chat.theme';
let lastChatId;
let shownIds = new Set();
let nodeCache = new Map();
let editingId = null;
let editingText = '';
let renamingId = null;
let followOutput = true;
let toastTimer;
let renderQueued = false;
let pendingFiles = [];
let attaching = 0;  // attachFiles calls still reading or uploading
let healthContextTokens = 8192;  // the server's context budget from /health, for models that state none
const contextTokensFor = model => modelInfo(model)?.contextTokens || healthContextTokens;
let speakingId = null;
const canSpeak = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
const reasoningOpen = new Map();  // message id -> the user's own expand/collapse choice

function toast(text) {
  $('toast').textContent = text;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2500);
}

async function copyText(text, done) {
  try {
    await navigator.clipboard.writeText(text);
    toast(done);
  } catch { toast('복사할 수 없습니다. 내용을 선택해서 복사해 주세요.'); }
}

function hideSidebar() {
  document.body.classList.remove('sidebar-open');
  $('sidebar-backdrop').hidden = true;
}

function dateGroup(time) {
  const day = 24 * 60 * 60 * 1000;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  if (time >= today.getTime()) return '오늘';
  if (time >= today.getTime() - day) return '어제';
  if (time >= today.getTime() - 7 * day) return '지난 7일';
  if (time >= today.getTime() - 30 * day) return '지난 30일';
  return '이전';
}

function renameField(chat) {
  const input = element('input', 'rename-input');
  input.value = chat.title;
  input.setAttribute('aria-label', '대화 이름');
  let finished = false;
  const finish = save => {
    if (finished) return;
    finished = true;
    renamingId = null;
    if (save) store.rename(chat.id, input.value);
    scheduleRender();
  };
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.isComposing) finish(true);
    if (event.key === 'Escape') { event.stopPropagation(); finish(false); }
  });
  input.addEventListener('blur', () => finish(true));
  requestAnimationFrame(() => { input.focus(); input.select(); });
  return input;
}

function renderHistory() {
  const history = $('chat-list');
  if (renamingId && document.activeElement?.classList.contains('rename-input')) return;  // keep the field alive
  history.replaceChildren();
  const query = $('chat-search').value.trim().toLowerCase();
  const chats = store.chats.filter(chat => chat.messages.length)
    .filter(chat => !query || chat.title.toLowerCase().includes(query)
      || chat.messages.some(message => message.content.toLowerCase().includes(query)))
    .sort((a, b) => b.updatedAt - a.updatedAt);
  if (!chats.length) {
    history.append(element('p', 'history-empty', query ? '검색 결과가 없습니다.' : '첫 대화를 시작해 보세요.'));
  }
  chats.sort((a, b) => Number(b.pinned === true) - Number(a.pinned === true));  // stable: keeps recency within groups
  let group;
  for (const chat of chats) {
    const heading = chat.pinned ? '고정됨' : dateGroup(chat.updatedAt);
    if (heading !== group) {
      group = heading;
      history.append(element('div', 'history-heading', group));
    }
    const row = element('div', 'chat-row' + (chat.id === store.activeId ? ' active' : ''));
    if (renamingId === chat.id) {
      row.append(renameField(chat));
      history.append(row);
      continue;
    }
    const select = element('button', 'chat-select');
    select.title = chat.title;
    if (chat.id === store.activeId) select.setAttribute('aria-current', 'true');
    select.append(icon('chat'), element('span', 'chat-title', chat.title));
    select.addEventListener('click', () => {
      store.select(chat.id);
      render();
      hideSidebar();
      prompt.focus();
    });
    select.addEventListener('dblclick', () => { renamingId = chat.id; render(); });
    const rename = iconButton('edit', chat.title + ' 이름 바꾸기', () => { renamingId = chat.id; render(); });
    rename.classList.add('chat-action');
    const pin = iconButton('pin', chat.pinned ? chat.title + ' 고정 해제' : chat.title + ' 고정', () => store.togglePin(chat.id));
    pin.classList.add('chat-action');
    if (chat.pinned) pin.classList.add('pinned');
    const remove = iconButton('trash', chat.title + ' 대화 삭제', () => {
      if (confirm('이 대화를 삭제할까요? 서버에서 지워져 모든 기기에서 사라집니다.')) store.remove(chat.id);
    });
    remove.classList.add('chat-action');
    remove.disabled = !store.canModify(chat);
    rename.disabled = pin.disabled = !store.canModify(chat);
    row.append(select, pin, rename, remove);
    history.append(row);
  }
}

function assistantHeading(message) {
  const heading = element('div', 'assistant-heading');
  const avatar = element('span', 'assistant-avatar');
  avatar.append(icon('spark'));
  heading.append(avatar, element('span', '', 'PubMed Chat'));
  const label = modelLabel(message.model);
  if (label) heading.append(element('span', 'model-tag', label));
  return heading;
}

function fileChips(files, onRemove) {
  const list = element('div', 'attachment-list');
  files.forEach((file, index) => {
    const chip = element('span', 'attachment-chip' + (isImage(file) ? ' image-chip' : ''));
    // A local preview while composing; then the uploaded copy or a base64 raster data URL, as checked on restore.
    const source = isImage(file) ? file.preview || imageSource(file) : null;
    if (source) {
      chip.title = file.name;
      const picture = element('img', 'attachment-image');
      picture.src = source;
      picture.alt = file.name;
      picture.addEventListener('error', () => {  // e.g. the file is gone from the server
        chip.classList.remove('image-chip');
        chip.title = file.name + ' · 이미지를 불러오지 못했어요';
        picture.replaceWith(icon('file'), element('span', 'attachment-name', file.name),
          element('span', 'attachment-size', '불러올 수 없음'));
      }, { once: true });
      chip.append(picture);
    } else if (isImage(file)) {
      chip.title = file.name + ' · 저장 공간이 부족해 이미지를 지웠어요';
      chip.append(icon('file'), element('span', 'attachment-name', file.name), element('span', 'attachment-size', '삭제됨'));
    } else {
      chip.title = file.name + (file.truncated ? ' · 길어서 앞부분만 첨부' : '');
      chip.append(icon('file'), element('span', 'attachment-name', file.name),
        element('span', 'attachment-size', file.text.length.toLocaleString() + '자' + (file.truncated ? '+' : '')));
    }
    if (onRemove) chip.append(iconButton('close', file.name + ' 첨부 취소', () => onRemove(index)));
    list.append(chip);
  });
  return list;
}

function describeCall(call) {
  const kind = call.name === 'pubmed_search_articles' ? '논문 검색' : '초록 조회';
  const query = typeof call.arguments?.query === 'string' ? call.arguments.query
    : Array.isArray(call.arguments?.pmids) ? 'PMID ' + call.arguments.pmids.join(', ') : '';
  return { kind, query };
}

function progressLine(message, chat) {
  const line = element('div', 'loading');
  const dots = element('span', 'loading-dots');
  dots.setAttribute('aria-hidden', 'true');
  dots.append(element('span'), element('span'), element('span'));
  const running = message.toolCalls.findLast(call => call.running);
  let text = '답변을 준비하고 있어요';
  if (running) {
    const { kind, query } = describeCall(running);
    text = 'PubMed ' + kind + ' 중' + (query ? ' · ' + query : '');
  } else if (message.stage === 'queued') text = '다른 요청이 끝나기를 기다리고 있어요';
  else if (message.stage === 'loading_model') {
    const label = modelLabel(message.model);
    text = (label ? label + ' 모델을' : '모델을') + ' 준비하고 있어요. 처음에는 시간이 걸릴 수 있어요';
  }
  else if (message.retrying) {
    // A rejected tool call is being rewritten; with search off the model is answering directly instead.
    text = message.retrying === 'tools_disabled' ? '답변을 다시 준비하고 있어요' : '검색 조건을 고쳐서 다시 시도하고 있어요';
  }
  else if (message.toolCalls.length) text = '찾은 논문을 읽고 답변을 쓰고 있어요';
  else if (!message.stage && chat.usePubmed) text = 'PubMed 연결을 준비하고 있어요';
  line.append(dots, document.createTextNode(text));
  return line;
}

function toolDetails(calls, open) {
  const details = element('details', 'tool-details');
  details.open = open;
  details.append(element('summary', '', 'PubMed 검색·조회 ' + calls.length + '회'));
  const list = element('ul');
  for (const call of calls) {
    const { kind, query } = describeCall(call);
    const state = call.running ? ' · 진행 중' : call.error ? ' · 실패' : '';
    list.append(element('li', call.running ? 'running' : '', kind + (query ? ' · ' + query : '') + state));
  }
  details.append(list);
  return details;
}

function reasoningPanel(message) {
  const thinking = message.streaming && !message.content && !message.toolCalls.some(call => call.running);
  const details = element('details', 'reasoning-panel' + (thinking ? ' active' : ''));
  const initial = reasoningOpen.has(message.id) ? reasoningOpen.get(message.id) : thinking;
  details.open = initial;
  const seconds = Math.max(1, Math.round((message.thinkMs || 0) / 1000));
  details.append(element('summary', '', thinking ? '생각하는 중…'
    : message.thinkMs !== undefined ? seconds + '초 동안 생각함' : '생각 과정'));
  details.append(element('div', 'reasoning-text', message.reasoning));
  // Setting .open above also fires a toggle event; only a change from that state is the user's choice.
  details.addEventListener('toggle', () => { if (details.open !== initial) reasoningOpen.set(message.id, details.open); });
  return details;
}

function startEdit(message) {
  editingId = message.id;
  editingText = message.content;
  render();
}

function sourceCards(sources) {
  const section = element('section', 'sources');
  section.setAttribute('aria-label', '참고한 논문');
  section.append(element('div', 'sources-heading', '참고한 논문 ' + sources.length + '편'));
  const list = element('div', 'source-list');
  for (const source of sources) {
    const card = element('a', 'source-card');
    card.href = source.url;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';
    card.append(element('span', 'source-title', source.title || 'PMID ' + source.pmid));
    const meta = [source.journal, source.year, source.authors].filter(Boolean).join(' · ');
    if (meta) card.append(element('span', 'source-meta', meta));
    const foot = element('span', 'source-foot');
    foot.append(element('span', '', 'PMID ' + source.pmid));
    if (source.abstractFetched) foot.append(element('span', 'source-badge', '초록 확인'));
    card.append(foot);
    list.append(card);
  }
  section.append(list);
  return section;
}

/** Plain text for speech: no code, tables read cell by cell, links read by their label. */
function speechText(message) {
  const holder = element('div');
  holder.innerHTML = renderMarkdown(withoutCitationList(message.content));
  holder.querySelectorAll('.code-block').forEach(block => block.replaceWith('코드는 생략합니다. '));
  holder.querySelectorAll('p, li, h2, h3, blockquote, tr').forEach(node =>
    node.append(/[.!?。]\s*$/.test(node.textContent) ? ' ' : '. '));  // pause between blocks
  return holder.textContent.replace(/\s+/g, ' ').replace(/(\.\s*){2,}/g, '. ').trim();
}

function toggleSpeech(message) {
  speechSynthesis.cancel();
  if (speakingId === message.id) { speakingId = null; render(); return; }
  // Chrome stops long utterances after ~15 seconds, so speak sentence-sized pieces in a queue.
  const pieces = speechText(message).match(/[^.!?。]+[.!?。]*\s*/g)?.map(piece => piece.trim()).filter(Boolean) || [];
  if (!pieces.length) return;
  const lang = /[가-힣]/.test(message.content) ? 'ko-KR' : 'en-US';
  speakingId = message.id;
  pieces.forEach((piece, index) => {
    const utterance = new SpeechSynthesisUtterance(piece);
    utterance.lang = lang;
    if (index === pieces.length - 1) utterance.onend = utterance.onerror = () => {
      if (speakingId === message.id) { speakingId = null; scheduleRender(); }
    };
    speechSynthesis.speak(utterance);
  });
  render();
}

function editForm(message) {
  const form = element('form', 'edit-form');
  const field = element('textarea', 'edit-input');
  field.value = editingText;
  field.setAttribute('aria-label', '질문 수정');
  field.addEventListener('input', () => { editingText = field.value; });
  const resize = () => { field.style.height = 'auto'; field.style.height = Math.min(field.scrollHeight, 260) + 'px'; };
  field.addEventListener('input', resize);
  field.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      form.requestSubmit();
    }
    if (event.key === 'Escape') { event.stopPropagation(); cancel.click(); }
  });
  const buttons = element('div', 'edit-buttons');
  const cancel = element('button', 'text-button', '취소');
  cancel.type = 'button';
  cancel.addEventListener('click', () => { editingId = null; render(); prompt.focus(); });
  const save = element('button', 'primary-button', '보내기');
  save.type = 'submit';
  buttons.append(cancel, save);
  form.append(field, buttons);
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (!field.value.trim()) return;
    if (!store.canModify()) { toast('지금 답변이 끝난 뒤에 보낼 수 있어요.'); return; }
    editingId = null;
    followOutput = true;
    store.edit(message.id, field.value);
  });
  requestAnimationFrame(() => { field.focus(); field.setSelectionRange(field.value.length, field.value.length); resize(); });
  return form;
}

function renderUser(message) {
  const article = element('article', 'message message-user');
  article.setAttribute('aria-label', '내 메시지');
  if (editingId === message.id) {
    article.append(editForm(message));
    return article;
  }
  if (message.attachments?.length) article.append(fileChips(message.attachments));
  article.append(element('div', 'user-content', message.content));
  const actions = element('div', 'message-actions user-actions');
  actions.append(iconButton('copy', '질문 복사', () => copyText(message.content, '질문을 복사했습니다.')));
  const edit = iconButton('edit', '질문 수정', () => startEdit(message));
  edit.disabled = !store.canModify();
  actions.append(edit);
  if (message.branches?.length > 1) {
    // Earlier wordings stay reachable, each with the conversation that followed it.
    const pager = element('div', 'version-pager');
    const back = iconButton('left', '이전 질문', () => store.switchBranch(message.id, message.branch - 1));
    back.disabled = message.branch === 0 || !store.canSwitchBranch(message.id);
    const next = iconButton('right', '다음 질문', () => store.switchBranch(message.id, message.branch + 1));
    next.disabled = message.branch === message.branches.length - 1 || !store.canSwitchBranch(message.id);
    pager.append(back, element('span', '', (message.branch + 1) + ' / ' + message.branches.length), next);
    actions.prepend(pager);
    actions.classList.add('has-branches');
  }
  article.append(actions);
  if (message.error) {
    const error = element('div', 'message-error');
    error.append(element('div', '', message.error));
    if (store.canRetry(message.id)) {
      const retry = element('button', 'retry-button');
      retry.append(icon('refresh'), document.createTextNode('다시 시도'));
      retry.addEventListener('click', () => { followOutput = true; store.retry(message.id); });
      error.append(retry);
    }
    article.append(error);
  }
  return article;
}

function renderAssistant(message, chat) {
  const article = element('article', 'message message-assistant');
  article.setAttribute('aria-label', '답변');
  article.append(assistantHeading(message));
  const body = element('div', 'assistant-body');
  if (message.reasoning) body.append(reasoningPanel(message));
  if (message.streaming && message.toolCalls.length) body.append(toolDetails(message.toolCalls, false));
  if (message.content) {
    const text = element('div', 'markdown' + (message.streaming ? ' streaming' : ''));
    text.innerHTML = renderMarkdown(message.sources?.length ? withoutCitationList(message.content) : message.content);
    body.append(text);
  }
  if (message.streaming && !message.content && !(message.reasoning && !message.toolCalls.some(call => call.running))) {
    body.append(progressLine(message, chat));
  }
  if (message.stopped) body.append(element('div', 'truncated-note', '답변 생성을 중지했습니다.'));
  if (message.dropped) body.append(element('div', 'truncated-note',
    '대화가 길어 앞부분 메시지 ' + message.dropped + '개는 참고하지 않고 답했어요. 새 주제라면 새 대화를 시작해 보세요.'));
  if (message.truncated) body.append(element('div', 'truncated-note', !message.content.trim() && message.reasoning
    ? '생각하는 데 길이를 모두 써서 답을 쓰지 못했어요. 추론 없이 다시 생성하거나 입력창에서 추론을 낮춰 보세요.'
    : '답변 길이 한도에 도달했습니다. 아래 “이어서 생성”을 누르거나 설정에서 답변 길이를 늘려 보세요.'));
  if (!message.streaming && message.sources?.length) body.append(sourceCards(message.sources));
  if (!message.streaming && message.toolCalls.length) body.append(toolDetails(message.toolCalls, false));
  article.append(body);
  if (!message.streaming) {
    const actions = element('div', 'message-actions');
    actions.append(iconButton('copy', '답변 복사', () => copyText(message.content, '답변을 복사했습니다.')));
    if (canSpeak && message.content.trim()) {
      const speaking = speakingId === message.id;
      const speak = iconButton(speaking ? 'stop' : 'speaker', speaking ? '읽기 중지' : '소리 내어 읽기', () => toggleSpeech(message));
      if (speaking) speak.classList.add('speaking');
      actions.append(speak);
    }
    if (store.canSelectVersion(message.id)) {
      const pager = element('div', 'version-pager');
      const index = message.version ?? message.versions.length - 1;
      const back = iconButton('left', '이전 답변', () => store.selectVersion(message.id, index - 1));
      back.disabled = index === 0;
      const next = iconButton('right', '다음 답변', () => store.selectVersion(message.id, index + 1));
      next.disabled = index === message.versions.length - 1;
      pager.append(back, element('span', '', (index + 1) + ' / ' + message.versions.length), next);
      actions.append(pager);
    }
    // Thinking used up the whole length before any answer: offer the way out prominently.
    const thoughtOnly = Boolean(message.truncated && !message.content.trim() && message.reasoning);
    if (store.canRegenerate(message.id) && thoughtOnly) {
      const plain = element('button', 'continue-button');
      plain.type = 'button';
      plain.append(icon('refresh'), document.createTextNode('추론 없이 다시 생성'));
      plain.addEventListener('click', () => { followOutput = true; store.regenerate(message.id, { effort: 'none' }); });
      actions.append(plain);
    }
    if (store.canContinue(message.id)) {
      const more = element('button', 'continue-button');
      more.type = 'button';
      more.append(icon('right'), document.createTextNode('이어서 생성'));
      more.addEventListener('click', () => { followOutput = true; store.continueAnswer(message.id); });
      actions.append(more);
    }
    if (store.canRegenerate(message.id)) {
      actions.append(iconButton('refresh', '답변 다시 생성', () => { followOutput = true; store.regenerate(message.id); }));
      if (!thoughtOnly && store.effortOf() !== 'none') {
        const plain = element('button', 'other-model-button', '추론 없이 다시 생성');
        plain.type = 'button';
        plain.title = '생각하는 과정 없이 같은 질문에 바로 다시 답하게 합니다';
        plain.addEventListener('click', () => { followOutput = true; store.regenerate(message.id, { effort: 'none' }); });
        actions.append(plain);
      }
      const current = store.modelOf();
      const asksAboutImages = store.current.messages.at(-2)?.attachments?.some(isImage);
      for (const model of modelList()) {
        if (model.id === current || model.state === 'missing' || (asksAboutImages && !model.vision)) continue;
        const other = element('button', 'other-model-button', model.label + '로 다시');
        other.type = 'button';
        other.title = model.label + ' 모델로 같은 질문에 다시 답하게 합니다';
        other.addEventListener('click', () => { followOutput = true; store.regenerate(message.id, { model: model.id }); });
        actions.append(other);
      }
    }
    if (message.elapsedMs !== undefined) {
      const seconds = (message.elapsedMs / 1000).toFixed(1);
      const stats = element('span', 'answer-stats', (message.tokens !== undefined ? message.tokens + '토큰 · ' : '') + seconds + '초');
      stats.title = '생성한 토큰 수와 걸린 시간 (모델 로딩·논문 검색 포함)';
      actions.append(stats);
    }
    article.append(actions);
  }
  return article;
}

/** Scrollable parts of a message; rebuilding a streaming message would otherwise reset them to the start. */
const SCROLLERS = '.reasoning-text, .code-block pre, .table-wrap';

function captureScroll(node) {
  return [...node.querySelectorAll(SCROLLERS)].map(element => ({ top: element.scrollTop, left: element.scrollLeft,
    atBottom: element.scrollHeight - element.scrollTop - element.clientHeight < 8 }));
}

function restoreScroll(node, saved, streaming) {
  node.querySelectorAll(SCROLLERS).forEach((element, index) => {
    const state = saved[index];
    if (state) element.scrollLeft = state.left;
    if (!element.classList.contains('reasoning-text')) {
      if (state) element.scrollTop = state.top;
      return;
    }
    // While reasoning streams, follow the newest lines unless the reader scrolled up to look back.
    element.scrollTop = streaming && (!state || state.atBottom) ? element.scrollHeight : state?.top ?? 0;
  });
}

function renderMessages() {
  const chat = store.current;
  const switched = lastChatId !== chat.id;
  if (switched) {
    nodeCache = new Map();
    editingId = null;
    followOutput = true;
    if (speakingId) { speechSynthesis.cancel(); speakingId = null; }
  }
  const nextCache = new Map();
  const restores = [];
  // Model names label every answer; the last one also offers the other models and "without reasoning".
  const labels = JSON.stringify(modelList().map(model => model.label));
  const choices = store.modelOf(chat) + store.effortOf(chat) + JSON.stringify(modelList());
  const nodes = chat.messages.map((message, index) => {
    // Rebuild a message only when its content or available actions change; streaming touches one node.
    // An open edit form is never rebuilt, so typed text and the caret survive unrelated updates.
    const last = index === chat.messages.length - 1;
    const key = editingId === message.id ? 'editing' : JSON.stringify(message) + store.canModify()
      + last + chat.usePubmed + reasoningOpen.get(message.id) + (speakingId === message.id) + labels + (last ? choices : '');
    const cached = nodeCache.get(message.id);
    const node = cached?.key === key ? cached.node
      : message.role === 'user' ? renderUser(message) : renderAssistant(message, chat);
    if (node !== cached?.node && (cached || message.streaming)) {
      restores.push([node, cached ? captureScroll(cached.node) : [], Boolean(message.streaming)]);
    }
    if (!switched && !shownIds.has(message.id)) node.classList.add('is-new');
    nextCache.set(message.id, { key, node });
    return node;
  });
  nodeCache = nextCache;
  shownIds = new Set(chat.messages.map(message => message.id));
  // Move only what changed: reinserting an unchanged node would replay its entrance animation.
  const messages = $('messages');
  let cursor = messages.firstChild;
  for (const node of nodes) {
    if (cursor === node) cursor = cursor.nextSibling;
    else messages.insertBefore(node, cursor);
  }
  while (cursor) {
    const next = cursor.nextSibling;
    cursor.remove();
    cursor = next;
  }
  for (const [node, saved, streaming] of restores) restoreScroll(node, saved, streaming);  // needs layout, so after insertion
  $('welcome').hidden = chat.messages.length > 0;
  if (switched || followOutput) conversation.scrollTop = conversation.scrollHeight;
  lastChatId = chat.id;
  updateScrollButton();
}

function updateScrollButton() {
  const distance = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight;
  $('scroll-bottom').hidden = distance < 120;
}

function renderPendingFiles() {
  $('attachments').hidden = !pendingFiles.length;
  $('attachments').replaceChildren(...fileChips(pendingFiles, index => {
    revokePreviews(pendingFiles.splice(index, 1));
    renderPendingFiles();
    updateComposer();
  }).children);
}

const revokePreviews = files => files.forEach(file => file.preview && URL.revokeObjectURL(file.preview));

/**
 * Downscale to at most 1024 px and re-encode as JPEG (the model sees ~512 px anyway), then upload it so
 * chats refer to the server's copy. If the upload fails the picture travels inline inside the chat.
 */
async function readImage(file) {
  let bitmap;
  try { bitmap = await createImageBitmap(file); }
  catch { throw new Error(file.name + '은(는) 열 수 없는 이미지 형식이에요. PNG·JPEG·WebP·GIF를 사용해 주세요.'); }
  const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
  const canvas = element('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const context = canvas.getContext('2d');
  context.fillStyle = '#fff';  // transparent areas would otherwise turn black in JPEG
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const name = file.name || '붙여넣은 이미지.jpg';
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
  if (blob) {
    try {
      return { name, kind: 'image', ...await uploadImage(api, blob), preview: URL.createObjectURL(blob) };
    } catch (error) {
      toast(error.code === 'storage_quota' ? '서버의 내 저장 공간이 가득 차서 이미지를 대화에 직접 담았어요.'
        : '이미지를 서버에 올리지 못해 대화에 직접 담았어요.');
    }
  }
  return { name, kind: 'image', dataUrl: canvas.toDataURL('image/jpeg', 0.85) };
}

/** Pictures need a model that can see them: switch this chat (not the choice for new chats) to one. */
function switchToVision() {
  const vision = modelList().find(model => model.vision && model.state !== 'missing');
  if (!vision) {
    toast('이미지를 이해할 수 있는 모델이 없어 이미지는 빼고 보내요.');
    return;
  }
  store.updateChat({ model: vision.id });
  toast('이미지를 이해할 수 있는 ' + vision.label + ' 모델로 바꿨어요.');
}

const isPdf = file => file.type === 'application/pdf' || /\.pdf$/i.test(file.name);

/** PDFs are turned into text by the server; the result is attached like a text file. */
async function readPdf(file) {
  let response;
  try {
    response = await api('/v1/documents/extract', { method: 'POST', body: file,
      headers: { 'Content-Type': 'application/pdf' } });
  } catch { throw new Error('서버에 연결할 수 없어 PDF를 읽지 못했어요.'); }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(file.name + ': ' + (response.status === 413 ? 'PDF는 20MB까지 첨부할 수 있어요.'
      : data.error?.code === 'invalid_document' ? '읽을 수 없는 PDF예요 (암호가 걸렸거나 손상됐을 수 있어요).'
        : 'PDF를 읽지 못했어요.'));
  }
  if (!data.text?.trim()) throw new Error(file.name + '에서 글자를 찾지 못했어요. 스캔한 PDF일 수 있어요.');
  return { name: file.name, text: async () => data.text };
}

async function attachFiles(files) {
  files = [...files];
  if (!files.length) return;
  const images = files.filter(file => file.type.startsWith('image/'));
  const pdfs = files.filter(isPdf);
  const texts = files.filter(file => !file.type.startsWith('image/') && !isPdf(file));
  const pendingTexts = pendingFiles.filter(file => !isImage(file));
  const used = pendingTexts.reduce((sum, file) => sum + file.text.length, 0);
  const usedTokens = pendingTexts.reduce((sum, file) => sum + estimateTokens(file.text), 0);
  attaching++;  // uploads take a moment; a message sent meanwhile would leave pictures behind
  updateComposer();
  try {
    const slots = () => MAX_ATTACHMENTS - pendingFiles.length;
    const room = Math.min(MAX_IMAGES - pendingFiles.filter(isImage).length, slots());
    if (images.length + texts.length + pdfs.length > slots()) toast('첨부는 한 번에 ' + MAX_ATTACHMENTS + '개까지예요.');
    else if (images.length > room) toast('이미지는 한 번에 ' + MAX_IMAGES + '장까지 보낼 수 있어요.');
    for (const file of images.slice(0, Math.max(0, room))) pendingFiles.push(await readImage(file));
    if (images.length && !modelInfo(store.modelOf())?.vision) switchToVision();
    if (pdfs.length) {
      toast('PDF에서 글자를 읽고 있어요…');
      for (const file of pdfs.slice(0, Math.max(0, slots() - texts.length))) texts.push(await readPdf(file));
    }
    if (texts.length) {
      const tokenBudget = attachmentTokenBudget(contextTokensFor(store.modelOf()));
      if (used >= ATTACHMENT_LIMIT || usedTokens >= tokenBudget) toast('텍스트 첨부가 모델이 읽을 수 있는 길이를 다 채웠어요.');
      else {
        const added = await readAttachments(texts.slice(0, Math.max(0, slots())), used, tokenBudget, usedTokens);
        pendingFiles.push(...added);
        if (added.some(file => file.truncated)) toast('파일이 길어 모델이 읽을 수 있는 앞부분(약 ' + tokenBudget.toLocaleString() + '토큰)만 첨부했어요.');
      }
    }
  } catch (error) { toast(error.message || '파일을 읽을 수 없습니다.'); }
  attaching--;
  renderPendingFiles();
  updateComposer();
  prompt.focus();
}

let lastAnnounced = false;
function announce() {
  // Streaming text is not announced token by token; screen readers hear start and finish.
  if (store.busy === lastAnnounced) return;
  lastAnnounced = store.busy;
  $('request-status').textContent = store.busy ? '답변을 생성하고 있어요.' : '답변이 끝났어요.';
}

function updateComposer() {
  $('send').hidden = store.busy;
  $('stop').hidden = !store.busy;
  $('send').disabled = attaching > 0 || store.liveChats.has(store.current.id) || (!prompt.value.trim() && !pendingFiles.length);
  $('attach').classList.toggle('busy', attaching > 0);
  prompt.style.height = 'auto';
  prompt.style.height = Math.min(prompt.scrollHeight, 220) + 'px';
}

function render() {
  renderQueued = false;
  const chat = store.current;
  const locked = !store.canModify();
  const elsewhere = store.liveChats.has(chat.id);
  renderPickers(locked);
  $('use-pubmed').checked = chat.usePubmed;
  $('use-pubmed').disabled = locked;
  $('new-chat').disabled = store.busy;
  $('storage-warning').textContent = store.storageWarning;
  $('storage-warning').hidden = !store.storageWarning;
  $('chat-heading').textContent = $('chat-heading').title = chat.title;
  document.title = chat.messages.length ? chat.title + ' · PubMed Chat' : 'PubMed Chat';
  $('export-chat').hidden = !chat.messages.length;
  if (store.notice) { toast(store.notice); store.notice = ''; }
  $('attach').disabled = locked;
  prompt.placeholder = elsewhere ? '다른 탭이나 기기에서 이 대화의 답변을 만들고 있어요' : '메시지를 입력하세요';
  $('chat-form').classList.toggle('locked', elsewhere);
  announce();
  renderHistory();
  renderMessages();
  updateComposer();
}

function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(render);
}

const EFFORT_TEXT = { none: ['끔', '바로 답변'], light: ['가볍게', '잠깐 생각'], medium: ['보통', '차근차근 생각'],
  high: ['깊게', '오래 생각'], max: ['최대', '가장 오래 생각, 느림'] };
/** What a model's state means for someone about to ask it something. */
const STATE_HINT = { starting: '준비 중…', stopped: '첫 질문 때 불러와요', failed: '시작하지 못했어요', missing: '모델 파일이 없어요' };
let openMenu = null;

/**
 * A composer button with a menu of choices (WAI-ARIA menu button, radio items). Arrow keys, Home and End
 * move, Enter or Space picks, Escape and Tab close; focus returns to the button. choices() gives
 * [{ value, label, hint?, badge?, disabled?, checked }]; pick(value) applies one.
 */
function menuButton(button, menu, title, choices, pick) {
  let shown = '';  // the choices on display, so unrelated renders leave an open menu and its focus alone
  const items = () => [...menu.querySelectorAll('.menu-item')];
  // Focus never scrolls the page; the menu alone scrolls to the item, which at 200% zoom or on a phone held
  // sideways may lie past the part of the menu that fits on screen.
  const focusAt = index => {
    const list = items();
    const at = (index + list.length) % list.length;
    list[at]?.focus({ preventScroll: true });
    revealItem(menu, list, at);
  };
  const item = choice => {
    const node = element('button', 'menu-item');
    node.type = 'button';
    node.tabIndex = -1;
    node.dataset.value = choice.value;
    node.setAttribute('role', 'menuitemradio');
    node.setAttribute('aria-checked', String(choice.checked));
    // The visible parts, with pauses between them that their markup alone may not give a screen reader.
    node.setAttribute('aria-label', [choice.label, choice.badge, choice.hint].filter(Boolean).join(', '));
    if (choice.disabled) node.setAttribute('aria-disabled', 'true');
    const name = element('span', 'menu-item-label', choice.label);
    if (choice.badge) name.append(element('span', 'menu-badge', choice.badge));
    const text = element('span', 'menu-item-text');
    text.append(name);
    if (choice.hint) text.append(element('span', 'menu-item-hint', choice.hint));
    const check = icon('check');
    check.classList.add('menu-check');
    node.append(text, check);
    node.addEventListener('click', () => {
      if (choice.disabled) return;
      close(true);
      pick(choice.value);
    });
    // The pointer and the keyboard share one highlight.
    node.addEventListener('pointermove', event => {
      if (event.pointerType === 'mouse' && document.activeElement !== node) node.focus({ preventScroll: true });
    });
    return node;
  };
  const build = () => {
    const list = choices();
    const key = JSON.stringify(list);
    if (key === shown) return;
    shown = key;
    const focused = menu.contains(document.activeElement) ? document.activeElement.dataset.value : undefined;
    const heading = element('div', 'menu-heading', title);
    heading.setAttribute('aria-hidden', 'true');  // the menu is named by its aria-label
    menu.replaceChildren(heading, ...list.map(item));
    if (focused !== undefined) items().find(node => node.dataset.value === focused)?.focus({ preventScroll: true });
  };
  // Keep the menu inside the window: it opens upwards unless only the space below fits it (a short
  // landscape screen, a long draft), and shifts left when its button sits near the right edge.
  const place = () => {
    menu.style.left = menu.style.maxHeight = '';
    const anchor = button.getBoundingClientRect();
    const above = anchor.top - 20;
    const below = document.documentElement.clientHeight - anchor.bottom - 20;
    const down = menu.scrollHeight > above && below > above;
    menu.classList.toggle('below', down);
    menu.style.maxHeight = Math.max(120, Math.min(400, Math.floor(down ? below : above))) + 'px';
    const { left, right } = menu.getBoundingClientRect();
    const shift = Math.min(right - (document.documentElement.clientWidth - 12), left - 12);
    if (shift > 0) menu.style.left = -shift + 'px';
    const list = items();
    revealItem(menu, list, list.indexOf(document.activeElement));  // a menu made shorter still shows its focus
  };
  const open = (target = 'checked') => {
    if (openMenu && openMenu !== control) openMenu.close();
    shown = '';
    build();
    menu.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    openMenu = control;
    place();
    const list = items();
    const checked = list.findIndex(node => node.getAttribute('aria-checked') === 'true');
    focusAt(target === 'last' ? -1 : Math.max(checked, 0));
  };
  const close = (returnFocus = false) => {
    if (menu.hidden) return;
    menu.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    if (openMenu === control) openMenu = null;
    if (returnFocus) button.focus();
  };
  button.addEventListener('click', () => (menu.hidden ? open() : close(true)));
  button.addEventListener('keydown', event => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    open(event.key === 'ArrowUp' ? 'last' : 'checked');
  });
  menu.addEventListener('keydown', event => {
    const index = items().indexOf(document.activeElement);
    const moves = { ArrowDown: index + 1, ArrowUp: index < 0 ? -1 : index - 1, Home: 0, End: -1 };
    if (Object.hasOwn(moves, event.key)) {
      event.preventDefault();
      focusAt(moves[event.key]);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      close(true);
    } else if (event.key === 'Tab') close(true);  // the browser then moves on from the button
  });
  const control = { close, place, refresh: () => { if (!menu.hidden) build(); },
    contains: node => menu.contains(node) || button.contains(node) };
  return control;
}

const modelMenu = menuButton($('model-button'), $('model-menu'), '모델', () => {
  const current = store.modelOf();
  return modelList().map(model => ({ value: model.id, label: model.label, badge: model.vision ? '이미지' : '',
    hint: [model.summary, STATE_HINT[model.state]].filter(Boolean).join(' · '), disabled: model.state === 'missing',
    checked: model.id === current }));
}, model => {
  store.updateChat({ model }, { remember: true });
  const hasImages = store.current.messages.some(message => message.attachments?.some(isImage));
  if (hasImages && !modelInfo(model)?.vision) {
    toast(modelLabel(model) + '은(는) 이미지를 볼 수 없어 이 대화의 이미지는 빼고 보내요.');
  }
});
const effortMenu = menuButton($('effort-button'), $('effort-menu'), '추론', () => {
  const current = store.effortOf();
  return ['none', ...(modelInfo(store.modelOf())?.efforts || [])].map(level => ({ value: level,
    label: EFFORT_TEXT[level][0], hint: EFFORT_TEXT[level][1], checked: level === current }));
}, effort => store.updateChat({ effort }, { remember: true }));

/** The composer's model and reasoning buttons show what the next question is sent with. */
function renderPickers(locked) {
  const model = store.modelOf();
  const info = modelInfo(model);
  const effort = store.effortOf();
  const reasons = Boolean(info?.efforts.length);
  $('model-label').textContent = modelLabel(model);
  $('model-button').title = [info?.summary, STATE_HINT[info?.state]].filter(Boolean).join(' · ') || '답할 모델 고르기';
  $('model-button').disabled = locked;
  $('effort-label').textContent = '추론: ' + EFFORT_TEXT[effort][0];
  $('effort-button').classList.toggle('active', effort !== 'none');
  $('effort-button').disabled = locked || !reasons;
  $('effort-button').title = reasons ? '답하기 전에 얼마나 생각할지 고릅니다. 깊을수록 정확할 수 있지만 느립니다'
    : '이 모델은 추론을 지원하지 않아요';
  if (locked) modelMenu.close();
  if (locked || !reasons) effortMenu.close();
  modelMenu.refresh();
  effortMenu.refresh();
}

store.onChange = scheduleRender;
window.addEventListener('storage', event => { if (event.key === store.storageKey) store.sync(); });
$('chat-form').addEventListener('submit', event => {
  event.preventDefault();
  if (attaching) { toast('첨부를 준비하고 있어요. 잠시 후 보내 주세요.'); return; }
  if (!store.canModify() || (!prompt.value.trim() && !pendingFiles.length)) return;
  // Never send a picture to a model that cannot see it.
  if (pendingFiles.some(isImage) && !modelInfo(store.modelOf())?.vision) switchToVision();
  const content = prompt.value;
  const files = pendingFiles;
  prompt.value = '';
  pendingFiles = [];
  renderPendingFiles();
  followOutput = true;
  updateComposer();
  store.send(content, files);  // keeps the uploaded copies; local previews are no longer shown
  revokePreviews(files);
  prompt.focus();
});
$('stop').addEventListener('click', () => { store.stop(); prompt.focus(); });
$('attach').addEventListener('click', () => $('file-input').click());
prompt.addEventListener('paste', event => {
  const files = [...(event.clipboardData?.files || [])].filter(file => file.type.startsWith('image/'));
  if (!files.length || !store.canModify()) return;
  event.preventDefault();  // a pasted screenshot becomes an attachment, not text
  attachFiles(files);
});
$('file-input').addEventListener('change', event => { attachFiles(event.target.files); event.target.value = ''; });
const composer = $('chat-form');
composer.addEventListener('dragover', event => {
  if (![...event.dataTransfer.types].includes('Files')) return;
  event.preventDefault();
  composer.classList.add('dragging');
});
composer.addEventListener('dragleave', () => composer.classList.remove('dragging'));
composer.addEventListener('drop', event => {
  composer.classList.remove('dragging');
  if (!event.dataTransfer.files.length) return;
  event.preventDefault();
  if (store.canModify()) attachFiles(event.dataTransfer.files);
});
$('export-chat').addEventListener('click', () => {
  const chat = store.current;
  const name = chat.title.replace(/[\\/:*?"<>|]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 60).replace(/[.\s]+$/, '');
  download(chatMarkdown(chat), 'text/markdown', (name || 'pubmed-chat') + '.md');
});
prompt.addEventListener('input', updateComposer);
prompt.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    $('chat-form').requestSubmit();
  }
  if (event.key === 'ArrowUp' && !prompt.value && store.canModify()) {
    const last = store.current.messages.findLast(message => message.role === 'user');
    if (last) { event.preventDefault(); startEdit(last); }
  }
});
conversation.addEventListener('scroll', () => {
  const distance = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight;
  followOutput = distance < 80;
  updateScrollButton();
}, { passive: true });
$('scroll-bottom').addEventListener('click', () => {
  followOutput = true;
  conversation.scrollTo({ top: conversation.scrollHeight, behavior: 'smooth' });
});
$('messages').addEventListener('click', event => {
  const button = event.target.closest('.code-copy');
  if (!button) return;
  copyText(button.closest('.code-block').querySelector('code').textContent, '코드를 복사했습니다.');
  button.textContent = '복사됨';
  setTimeout(() => { button.textContent = '복사'; }, 1500);
});
$('use-pubmed').addEventListener('change', event => store.updateChat({ usePubmed: event.target.checked }));
// A tap or click anywhere else closes an open composer menu; a resize (e.g. the on-screen keyboard) moves it.
document.addEventListener('pointerdown', event => {
  if (openMenu && !openMenu.contains(event.target)) openMenu.close();
}, true);
window.addEventListener('resize', () => openMenu?.place());
$('chat-search').addEventListener('input', renderHistory);
function newChat() {
  if (store.busy) return;
  openMenu?.close();
  store.newChat();
  render();  // controls must reflect the new chat before the next input event
  prompt.value = '';
  updateComposer();
  hideSidebar();
  prompt.focus();
}
$('new-chat').addEventListener('click', newChat);
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    newChat();
  }
  if (event.key === 'Escape') {
    if (store.busy && document.activeElement === prompt) store.stop();
    openMenu?.close();
    hideSidebar();
  }
});
for (const suggestion of document.querySelectorAll('.suggestion')) {
  suggestion.addEventListener('click', () => {
    if (!store.canModify()) return;
    prompt.value = suggestion.dataset.prompt;
    store.updateChat({ usePubmed: suggestion.dataset.pubmed !== 'false' });
    prompt.focus();
  });
}
$('close-sidebar').addEventListener('click', () => {
  document.body.classList.add('sidebar-collapsed');
  $('open-sidebar').hidden = false;
  hideSidebar();
});
$('open-sidebar').addEventListener('click', () => {
  document.body.classList.remove('sidebar-collapsed');
  $('open-sidebar').hidden = true;
  if (matchMedia('(max-width: 760px)').matches) {
    document.body.classList.add('sidebar-open');
    $('sidebar-backdrop').hidden = false;
  }
});
$('sidebar-backdrop').addEventListener('click', hideSidebar);

function applyTheme(theme) {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  try { storage.setItem(THEME_KEY, theme); } catch { /* the theme just won't persist */ }
}
function savedTheme() {
  try { return storage.getItem(THEME_KEY) || 'system'; } catch { return 'system'; }
}
const MODEL_TEMPERATURE = 0.7;  // where the slider rests while the model's recommended sampling applies
/** null: the model's own recommended sampling, which differs with and without reasoning. */
function showTemperature(value = store.settings.temperature) {
  const chosen = value !== null;
  $('temperature-value').textContent = chosen ? Number(value).toFixed(1) : '모델 기본값';
  $('temperature-reset').hidden = !chosen;
  if (chosen) $('temperature').removeAttribute('aria-valuetext');
  else $('temperature').setAttribute('aria-valuetext', '모델 기본값');
}
$('open-settings').addEventListener('click', () => {
  $('system-prompt').value = store.settings.systemPrompt;
  $('temperature').value = store.settings.temperature ?? MODEL_TEMPERATURE;
  $('max-tokens').value = String(store.settings.maxTokens);
  $('theme').value = savedTheme();
  showTemperature();
  $('settings-dialog').showModal();
});
$('settings-dialog').addEventListener('click', event => {
  if (event.target === $('settings-dialog')) $('settings-dialog').close();  // backdrop click
});
$('system-prompt').addEventListener('input', event => store.updateSettings({ systemPrompt: event.target.value }));
$('temperature').addEventListener('input', event => showTemperature(Number(event.target.value)));
$('temperature').addEventListener('change', event => store.updateSettings({ temperature: Number(event.target.value) }));
$('temperature-reset').addEventListener('click', () => {
  store.updateSettings({ temperature: null });
  $('temperature').value = MODEL_TEMPERATURE;
  showTemperature();
  $('temperature').focus();  // the button that had focus is hidden now
});
$('max-tokens').addEventListener('change', event => store.updateSettings({ maxTokens: Number(event.target.value) }));
$('theme').addEventListener('change', event => applyTheme(event.target.value));
function download(text, type, name) {
  const link = element('a');
  link.href = URL.createObjectURL(new Blob([text], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
$('export-chats').addEventListener('click', () => {
  download(JSON.stringify({ exportedAt: new Date().toISOString(),
    chats: store.chats.filter(chat => chat.messages.length) }, null, 2),
  'application/json', 'pubmed-chat-' + new Date().toISOString().slice(0, 10) + '.json');
});
$('clear-chats').addEventListener('click', () => {
  if (store.busy || !confirm('모든 대화를 삭제할까요? 서버에서 지워져 모든 기기에서 사라지며, 되돌릴 수 없습니다.')) return;
  for (const chat of [...store.chats]) store.remove(chat.id);
  $('settings-dialog').close();
  toast('모든 대화를 삭제했습니다.');
});

function setupNotice(lines, command) {
  const notice = $('setup-notice');
  notice.hidden = !lines;
  if (!lines) return;
  notice.replaceChildren(...lines.map(line => element('p', '', line)));
  if (command) notice.append(element('code', '', command));
}

const RUNTIME_NAME = { mlx: 'MLX', transformers: 'Transformers' };
const STATE_TEXT = { ready: '준비됨', starting: '준비 중', stopped: '대기 중 · 첫 질문 때 불러와요', failed: '시작 실패',
  missing: '파일 없음' };
const SERVER_TEXT = { ready: '서버 연결됨', starting: '모델 준비 중', unavailable: '모델 준비 필요' };
let serverTimer;

/** One model's line in the server status tooltip. */
function modelStatus(model) {
  const runtime = Object.hasOwn(RUNTIME_NAME, model.runtime) ? ' (' + RUNTIME_NAME[model.runtime] + ')' : '';
  return model.label + runtime + ': ' + (STATE_TEXT[model.state] || '확인 전') + (model.error ? ' — ' + model.error : '');
}

async function checkServer() {
  clearTimeout(serverTimer);
  let starting = false;
  const status = $('server-status');
  try {
    const response = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw Error('offline');
    const health = await response.json();
    const inference = health.inference || {};
    const states = inference.models && typeof inference.models === 'object' ? inference.models : null;
    if (Number.isInteger(inference.max_context_tokens)) healthContextTokens = inference.max_context_tokens;
    if (await modelSync.follow(states)) scheduleRender();
    const state = serverState(health);
    starting = state === 'starting';
    status.classList.toggle('starting', starting);
    status.classList.toggle('offline', state === 'unavailable');  // missing files or a failed start: see the notice
    status.lastElementChild.textContent = SERVER_TEXT[state];
    const models = states ? modelList().map(modelStatus)
      : ['메모리의 모델: ' + (modelLabel(inference.loaded_model) || '없음 (첫 질문 때 불러옵니다)')];
    status.title = [...models, 'PubMed: ' + ({ available: '연결됨', unavailable: '연결 안 됨' }[health.pubmed?.status] || '확인 전'),
      '실행 장치: ' + String(inference.device || '?').toUpperCase()].join('\n');
    // Only the models this server offers count; the files of others may well be absent.
    const missing = states ? modelList().filter(model => model.state === 'missing').map(model => model.id)
      : Object.entries(inference.model_files || {}).filter(([, ok]) => !ok).map(([name]) => name);
    const failed = modelList().filter(model => model.state === 'failed');
    // People signing in from elsewhere cannot run commands on the server PC.
    if (missing.length && accounts) setupNotice(['서버에 모델이 아직 준비되지 않았습니다. 서버 관리자에게 알려 주세요.']);
    else if (missing.length) {
      setupNotice(['모델 파일이 준비되지 않았습니다: ' + missing.join(', '),
        '프로젝트 폴더에서 아래 명령을 실행한 뒤 이 페이지를 새로고침하세요.'], 'uv run python scripts/prepare_models.py');
    } else if (failed.length && accounts) setupNotice(['모델을 시작하지 못했어요. 서버 관리자에게 알려 주세요.']);
    else if (failed.length) {
      setupNotice(['모델을 시작하지 못했어요: ' + failed.map(model => model.label
        + (model.error ? ' (' + model.error + ')' : '')).join(', '), '서버 로그를 확인해 주세요.']);
    } else setupNotice(null);
  } catch {
    status.classList.remove('starting');
    status.classList.add('offline');
    status.lastElementChild.textContent = '서버 연결 끊김';
    if (accounts) setupNotice(['서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.']);
    else {
      setupNotice(['서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.'],
        'uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1');
    }
  }
  serverTimer = setTimeout(checkServer, starting ? 3000 : 15000);  // a starting model is followed closely
}

const SYNC_TEXT = { synced: '서버에 저장됨', syncing: '저장하는 중…', offline: '오프라인 — 연결되면 저장해요',
  unauthorized: '다시 로그인해 주세요' };
let syncTimer;
function showSyncStatus(status) {
  if (status === 'unauthorized') signIn();
  clearTimeout(syncTimer);
  const show = () => {
    $('sync-status').dataset.state = status;
    $('sync-text').textContent = SYNC_TEXT[status];
  };
  // Most saves reach the server in a moment; only one that takes a while is worth showing.
  if (status === 'syncing' && $('sync-status').dataset.state === 'synced') syncTimer = setTimeout(show, 800);
  else show();
}

function showAccount() {
  $('account-name').textContent = accounts ? userName : '이 PC 전용';
  $('account').title = accounts ? '로그인한 계정' : '로그인 없이 이 PC의 서버를 쓰고 있어요';
  $('account-avatar').textContent = accounts ? [...userName][0].toUpperCase() : '';
  $('account-avatar').hidden = !accounts;
  $('local-dot').hidden = accounts;
  $('logout').hidden = !accounts;
  $('history-note').textContent = accounts
    ? '대화 기록은 서버의 내 계정에 저장돼요. 다른 기기에서 로그인해도 이어서 볼 수 있어요.'
    : '대화 기록은 이 PC에서 실행 중인 서버에 저장돼요.';
  $('api-docs').hidden = accounts;
}

$('logout').addEventListener('click', async () => {
  if (store.busy) { toast('답변이 끝난 뒤에 로그아웃할 수 있어요.'); return; }
  if (!await sync.flush() && !confirm('아직 서버에 저장하지 못한 변경이 있어요. 로그아웃하면 이 기기에서 사라져요. 그래도 로그아웃할까요?')) return;
  let response;
  try { response = await fetch('/api/logout', { method: 'POST' }); }
  catch { toast('서버에 연결할 수 없어 로그아웃하지 못했어요.'); return; }
  if (!response.ok && response.status !== 401) { toast('로그아웃하지 못했어요. 잠시 후 다시 시도해 주세요.'); return; }
  sync.stop();
  // Nothing more is cached for this user: a save still on its way would bring the history back.
  store.storage = sync.storage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  forgetCache(storage, store.storageKey);
  signIn();
});

/** History kept in this browser before chats moved to the server is offered once, if the account is empty. */
async function offerLegacyImport() {
  if (!sync.ready || sync.serverChats > 0) return;
  const count = legacyChats(storage).length;
  if (!count) return;
  const target = accounts ? '내 계정(' + userName + ')으로' : '이 PC의 서버로';
  if (!confirm('이 브라우저에 서버 저장 전의 대화 ' + count + '개가 남아 있어요. ' + target + ' 옮길까요?\n\n'
    + '취소하면 다시 묻지 않아요.')) {
    skipLegacy(storage);
    return;
  }
  toast('예전 대화를 옮기고 있어요…');
  try {
    const result = await importLegacy(store, { fetcher: api, storage,
      onProgress: (done, total) => toast('예전 대화의 이미지를 올리고 있어요… ' + done + ' / ' + total) });
    if (!result) return;
    const kept = result.images - result.uploaded;
    toast('예전 대화 ' + result.chats + '개를 옮겼어요.'
      + (kept ? ' 이미지 ' + kept + '장은 올리지 못해 대화 안에 그대로 담았어요.' : ''));
  } catch { toast('예전 대화를 옮기지 못했어요.'); }
}

showAccount();
render();
checkServer();  // and again every 15 s, or every 3 s while a model is starting
window.addEventListener('online', () => sync.syncNow());
sync.start().then(async () => {
  await offerLegacyImport();
  // Answers found running at load are settled only now that the server said which ones still are.
  store.refreshLive();
  setInterval(() => store.refreshLive(), 4000);  // notices when another tab answering a chat closes
});
