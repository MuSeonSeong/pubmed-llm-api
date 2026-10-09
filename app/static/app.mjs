import { ChatStore, renderMarkdown } from './chat.mjs';

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

let storage;
try { storage = window.localStorage; }
catch { storage = { getItem: () => null, setItem: () => { throw Error('Storage unavailable'); } }; }
const store = new ChatStore(storage);
const prompt = $('prompt');
let lastChatId;
let lastMessageKey;
let shownIds = new Set();
let toastTimer;

function toast(text) {
  $('toast').textContent = text;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2500);
}

function hideSidebar() {
  document.body.classList.remove('sidebar-open');
  $('sidebar-backdrop').hidden = true;
}

function renderHistory() {
  const history = $('chat-list');
  history.replaceChildren();
  const chats = [...store.chats].filter(chat => chat.messages.length)
    .sort((a, b) => b.updatedAt - a.updatedAt);
  if (!chats.length) history.append(element('p', 'history-empty', '첫 대화를 시작해 보세요.'));
  for (const chat of chats) {
    const row = element('div', 'chat-row' + (chat.id === store.activeId ? ' active' : ''));
    const select = element('button', 'chat-select');
    select.title = chat.title;
    select.setAttribute('aria-label', chat.title);
    if (chat.id === store.activeId) select.setAttribute('aria-current', 'true');
    select.append(icon('chat'), element('span', 'chat-title', chat.title));
    select.addEventListener('click', () => {
      store.select(chat.id);
      hideSidebar();
      prompt.focus();
    });
    const remove = element('button', 'chat-delete icon-button');
    remove.setAttribute('aria-label', chat.title + ' 대화 삭제');
    remove.title = '대화 삭제';
    remove.disabled = store.busy;
    remove.append(icon('trash'));
    remove.addEventListener('click', () => {
      if (confirm('이 대화를 삭제할까요? 브라우저에 저장된 기록이 삭제됩니다.')) store.remove(chat.id);
    });
    row.append(select, remove);
    history.append(row);
  }
}

function assistantHeading() {
  const heading = element('div', 'assistant-heading');
  const avatar = element('span', 'assistant-avatar');
  avatar.append(icon('spark'));
  heading.append(avatar, element('span', '', 'PubMed Chat'));
  return heading;
}

function renderMessage(message) {
  const article = element('article', 'message message-' + message.role);
  article.setAttribute('aria-label', message.role === 'user' ? '내 메시지' : '답변');
  if (message.role === 'user') {
    article.append(element('div', 'user-content', message.content));
    if (message.error) {
      const error = element('div', 'message-error');
      error.append(element('div', '', message.error));
      const retry = element('button', 'retry-button');
      retry.append(icon('refresh'), document.createTextNode('다시 시도'));
      retry.disabled = store.busy;
      retry.addEventListener('click', () => store.retry(message.id));
      if (store.current.messages.at(-1).id === message.id) error.append(retry);
      article.append(error);
    }
  } else {
    article.append(assistantHeading());
    const body = element('div', 'assistant-body');
    body.innerHTML = renderMarkdown(message.content);
    if (message.truncated) body.append(element('div', 'truncated-note',
      '답변 길이 한도에 도달했습니다. “이어서 설명해줘”라고 요청할 수 있어요.'));
    if (Array.isArray(message.toolCalls) && message.toolCalls.length) {
      const details = element('details', 'tool-details');
      details.append(element('summary', '', 'PubMed 검색·조회 ' + message.toolCalls.length + '회'));
      const list = element('ul');
      for (const call of message.toolCalls) {
        const kind = call.name === 'pubmed_search_articles' ? '논문 검색' : '초록 조회';
        const query = typeof call.arguments?.query === 'string' ? call.arguments.query
          : Array.isArray(call.arguments?.pmids) ? 'PMID ' + call.arguments.pmids.join(', ') : '';
        list.append(element('li', '', kind + (query ? ' · ' + query : '') + (call.error ? ' · 실패' : '')));
      }
      details.append(list);
      body.append(details);
    }
    const actions = element('div', 'message-actions');
    const copy = element('button', 'icon-button');
    copy.title = '답변 복사';
    copy.setAttribute('aria-label', '답변 복사');
    copy.append(icon('copy'));
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(message.content);
        toast('답변을 복사했습니다.');
      } catch { toast('복사할 수 없습니다. 답변을 선택해서 복사해 주세요.'); }
    });
    actions.append(copy);
    article.append(body, actions);
  }
  return article;
}

function renderMessages() {
  const chat = store.current;
  const messageKey = chat.id + JSON.stringify(chat.messages);
  if (messageKey === lastMessageKey) return;
  const conversation = $('conversation');
  const atBottom = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < 100;
  const switched = lastChatId !== chat.id;
  const messages = $('messages');
  // Only messages that just arrived animate in; rerenders must not flash the whole log.
  const rendered = chat.messages.map(message => {
    const node = renderMessage(message);
    if (!switched && !shownIds.has(message.id)) node.classList.add('is-new');
    return node;
  });
  shownIds = new Set(chat.messages.map(message => message.id));
  messages.replaceChildren(...rendered);
  $('welcome').hidden = chat.messages.length > 0;
  const pending = chat.messages.some(message => message.pending);
  if (pending) {
    const loading = element('div', 'message message-assistant is-new');
    loading.append(assistantHeading());
    const line = element('div', 'loading');
    const dots = element('span', 'loading-dots');
    dots.setAttribute('aria-hidden', 'true');
    dots.append(element('span'), element('span'), element('span'));
    line.append(dots, document.createTextNode(chat.usePubmed ? '논문을 살펴보며 답변을 준비하고 있어요' : '답변을 준비하고 있어요'));
    loading.append(line);
    messages.append(loading);
  }
  if (atBottom || switched || pending) conversation.scrollTop = conversation.scrollHeight;
  lastMessageKey = messageKey;
  lastChatId = chat.id;
}

function updateSendButton() {
  $('send').disabled = store.busy || !prompt.value.trim();
  prompt.style.height = 'auto';
  prompt.style.height = Math.min(prompt.scrollHeight, 180) + 'px';
}

function render() {
  const chat = store.current;
  $('model').value = chat.model;
  $('model').disabled = store.busy;
  $('use-pubmed').checked = chat.usePubmed;
  $('use-pubmed').disabled = store.busy;
  $('new-chat').disabled = store.busy;
  $('storage-warning').textContent = store.storageWarning;
  $('storage-warning').hidden = !store.storageWarning;
  $('request-status').textContent = store.busy
    ? '응답을 기다리는 중입니다. 처음 실행하거나 모델을 바꾸면 로딩 시간이 필요해요.' : '';
  document.title = chat.messages.length ? chat.title + ' · PubMed Chat' : 'PubMed Chat';
  renderHistory();
  renderMessages();
  updateSendButton();
}

store.onChange = render;
$('chat-form').addEventListener('submit', event => {
  event.preventDefault();
  if (store.busy || !prompt.value.trim()) return;
  const content = prompt.value;
  prompt.value = '';
  updateSendButton();
  store.send(content);
  prompt.focus();
});
prompt.addEventListener('input', updateSendButton);
prompt.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    $('chat-form').requestSubmit();
  }
});
$('model').addEventListener('change', event => { store.current.model = event.target.value; store.emit(); });
$('use-pubmed').addEventListener('change', event => { store.current.usePubmed = event.target.checked; store.emit(); });
function newChat() {
  if (store.busy) return;
  store.newChat();
  prompt.value = '';
  updateSendButton();
  hideSidebar();
  prompt.focus();
}
$('new-chat').addEventListener('click', newChat);
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    newChat();
  }
  if (event.key === 'Escape') hideSidebar();
});
for (const suggestion of document.querySelectorAll('.suggestion')) {
  suggestion.addEventListener('click', () => {
    if (store.busy) return;
    prompt.value = suggestion.dataset.prompt;
    store.current.usePubmed = suggestion.dataset.pubmed !== 'false';
    store.emit();
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

async function checkServer() {
  try {
    const response = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw Error('offline');
    const health = await response.json();
    const ready = health.ready_for_chat;
    $('server-status').classList.toggle('offline', !ready);
    $('server-status').lastElementChild.textContent = ready ? '서버 연결됨' : '모델 준비 필요';
  } catch {
    $('server-status').classList.add('offline');
    $('server-status').lastElementChild.textContent = '서버 연결 끊김';
  }
}
render();
checkServer();
setInterval(checkServer, 15000);
