const STORAGE_KEY = 'pubmed-chat.v1';
const MODELS = ['qwen3.5-2b', 'qwen3-0.6b'];
const id = () => globalThis.crypto.randomUUID();
const freshChat = () => ({ id: id(), title: '새 대화', model: MODELS[0],
  usePubmed: true, messages: [], updatedAt: Date.now() });

function normalizeTools(calls) {
  return Array.isArray(calls) ? calls.filter(call => call && typeof call.name === 'string')
    .map(call => ({ name: call.name,
      arguments: call.arguments && typeof call.arguments === 'object' ? call.arguments : {},
      ...(typeof call.error === 'string' ? { error: call.error } : {}) })) : [];
}

function restoreChat(chat) {
  if (!chat || typeof chat.id !== 'string' || typeof chat.title !== 'string'
      || !Array.isArray(chat.messages)) return null;
  const messages = chat.messages.filter(message => message && typeof message.id === 'string'
    && ['user', 'assistant'].includes(message.role) && typeof message.content === 'string')
    .map(message => ({ id: message.id, role: message.role, content: message.content,
      toolCalls: normalizeTools(message.toolCalls), truncated: message.truncated === true,
      ...(typeof message.error === 'string' ? { error: message.error } : {}),
      ...(message.role === 'user' && message.pending === true ? { pending: true } : {}) }));
  for (const message of messages) {
    if (message.pending) {
      delete message.pending;
      message.error = '페이지가 닫혀 응답을 받지 못했습니다. 다시 시도해 주세요.';
    }
  }
  return { id: chat.id, title: chat.title, model: MODELS.includes(chat.model) ? chat.model : MODELS[0],
    usePubmed: chat.usePubmed !== false, messages, updatedAt: Number(chat.updatedAt) || Date.now() };
}

function errorMessage(error, status) {
  const hints = {
    context_overflow: '대화가 모델의 길이 한도를 넘었습니다. 새 대화를 시작하거나 질문을 줄여 주세요.',
    model_missing: '모델 파일이 없습니다. 서버에서 모델 준비를 완료해 주세요.',
    device_unavailable: '설정된 실행 장치를 사용할 수 없습니다.',
    pubmed_timeout: 'PubMed 연결 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.',
    tool_limit: '논문 검색 단계가 너무 많아 중단됐습니다. 요청을 더 구체적으로 줄여 주세요.',
  };
  if (hints[error?.code]) return hints[error.code];
  if (status === 413) return hints.context_overflow;
  if (status === 504) return hints.pubmed_timeout;
  if (status === 502) return 'PubMed 검색을 완료하지 못했습니다. 다시 시도하거나 PubMed 검색을 꺼 주세요.';
  return typeof error?.message === 'string' ? error.message : '응답을 받지 못했습니다. 다시 시도해 주세요.';
}

export class ChatStore {
  constructor(storage = globalThis.localStorage, fetcher = globalThis.fetch) {
    this.storage = storage;
    this.fetcher = fetcher;
    this.busy = false;
    this.onChange = () => {};
    this.storageWarning = '';
    this.chats = [];
    try {
      const saved = JSON.parse(storage.getItem(STORAGE_KEY) || 'null');
      if (Array.isArray(saved?.chats)) this.chats = saved.chats.map(restoreChat).filter(Boolean);
      this.activeId = this.chats.some(chat => chat.id === saved?.activeId)
        ? saved.activeId : this.chats[0]?.id;
    } catch {
      this.storageWarning = '대화 기록을 불러오지 못했습니다. 새 대화로 시작합니다.';
    }
    if (!this.chats.length) {
      this.chats = [freshChat()];
      this.activeId = this.chats[0].id;
    }
  }

  get current() { return this.chats.find(chat => chat.id === this.activeId); }

  save() {
    try {
      this.storage.setItem(STORAGE_KEY, JSON.stringify({ chats: this.chats, activeId: this.activeId }));
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

  remove(chatId) {
    if (this.busy) return;
    this.chats = this.chats.filter(chat => chat.id !== chatId);
    if (!this.chats.length) this.chats = [freshChat()];
    if (!this.chats.some(chat => chat.id === this.activeId)) this.activeId = this.chats[0].id;
    this.emit();
  }

  async send(content) {
    if (this.busy || !content.trim()) return;
    const chat = this.current;
    const message = { id: id(), role: 'user', content: content.trim() };
    chat.messages.push(message);
    if (chat.title === '새 대화') chat.title = message.content.replace(/\s+/g, ' ').slice(0, 36);
    return this.request(chat, message);
  }

  async retry(messageId) {
    if (this.busy) return;
    const chat = this.current;
    const message = chat.messages.at(-1);
    if (message?.id === messageId && message.role === 'user' && message.error) {
      return this.request(chat, message);
    }
  }

  async request(chat, message) {
    this.busy = true;
    delete message.error;
    message.pending = true;
    chat.updatedAt = Date.now();
    this.emit();
    const index = chat.messages.indexOf(message);
    const messages = chat.messages.slice(0, index + 1).filter(item => !item.error)
      .map(({ role, content }) => ({ role, content }));
    try {
      const fetcher = this.fetcher;
      const response = await fetcher('/v1/chat/completions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: chat.model, messages, max_tokens: 512,
          temperature: 0.2, use_pubmed: chat.usePubmed }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(errorMessage(data.error, response.status));
      const content = data.choices?.[0]?.message?.content;
      if (typeof content !== 'string' || !content.trim()) throw new Error('빈 응답을 받았습니다. 다시 시도해 주세요.');
      chat.messages.splice(index + 1, 0, { id: id(), role: 'assistant', content,
        toolCalls: normalizeTools(data.tool_calls),
        truncated: data.choices[0].finish_reason === 'length' });
    } catch (error) {
      message.error = error instanceof TypeError
        ? '서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하고 다시 시도해 주세요.'
        : error.message || '응답을 받지 못했습니다. 다시 시도해 주세요.';
    } finally {
      delete message.pending;
      this.busy = false;
      chat.updatedAt = Date.now();
      this.emit();
    }
  }
}

export const escapeHtml = value => String(value).replace(/[&<>"']/g,
  char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

function inlineMarkdown(text) {
  // Escape first; only our own tags and validated HTTP(S) links may enter the DOM.
  return escapeHtml(text).replace(
    /`([^`\n]+)`|\[([^\]\n]+)\]\(([^\s)]+)\)|\*\*([^*\n]+)\*\*|\*(?!\s)([^*\n]+?)(?<!\s)\*/g,
    (match, code, label, target, bold, italic) => {
      if (code !== undefined) return '<code>' + code + '</code>';
      if (bold !== undefined) return '<strong>' + bold + '</strong>';
      if (italic !== undefined) return '<em>' + italic + '</em>';
      if (!/^https?:\/\//i.test(target)) return label;
      return '<a href="' + target + '" target="_blank" rel="noopener noreferrer">' + label + '</a>';
    });
}

export function renderMarkdown(text) {
  return String(text).split(/(```[\s\S]*?(?:```|$))/g).map(part => {
    if (part.startsWith('```')) {
      const code = part.replace(/^```[^\n]*\n?/, '').replace(/```$/, '').replace(/\n$/, '');
      return '<pre><code>' + escapeHtml(code) + '</code></pre>';
    }
    let list = null;
    let output = '';
    const closeList = () => { if (list) output += '</' + list + '>'; list = null; };
    for (const line of part.split('\n')) {
      const bullet = line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)/);
      if (bullet) {
        const type = /^\s*\d+\./.test(line) ? 'ol' : 'ul';
        if (list !== type) { closeList(); output += '<' + type + '>'; list = type; }
        output += '<li>' + inlineMarkdown(bullet[1]) + '</li>';
      } else {
        closeList();
        const heading = line.match(/^#{1,6}\s+(.+)/);
        if (heading) output += '<h3>' + inlineMarkdown(heading[1]) + '</h3>';
        else if (line.trim()) output += '<p>' + inlineMarkdown(line) + '</p>';
      }
    }
    closeList();
    return output;
  }).join('');
}
