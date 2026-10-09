import test from 'node:test';
import assert from 'node:assert/strict';
import { ChatStore, renderMarkdown } from '../app/static/chat.mjs';

const memory = () => {
  const data = new Map();
  return { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value) };
};
const reply = (content, tool_calls = []) => ({ ok: true,
  json: async () => ({ choices: [{ message: { content }, finish_reason: 'stop' }], tool_calls }) });

test('follow-up sends previous turns and selected settings, excluding tool metadata', async () => {
  const requests = [];
  const store = new ChatStore(memory(), async (url, init) => {
    assert.equal(url, '/v1/chat/completions');
    requests.push(JSON.parse(init.body));
    return reply('답변', [{ name: 'pubmed_search_articles', arguments: { query: 'CRISPR' } }]);
  });
  await store.send('첫 질문');
  store.current.model = 'qwen3-0.6b';
  store.current.usePubmed = false;
  await store.send('이어서 설명해줘');
  assert.deepEqual(requests[1].messages, [
    { role: 'user', content: '첫 질문' }, { role: 'assistant', content: '답변' },
    { role: 'user', content: '이어서 설명해줘' },
  ]);
  assert.equal(requests[1].model, 'qwen3-0.6b');
  assert.equal(requests[1].use_pubmed, false);
  assert.equal(store.current.messages[1].toolCalls.length, 1);
});

test('reload restores multiple conversations and the selected conversation', async () => {
  const storage = memory();
  const store = new ChatStore(storage, async () => reply('4'));
  await store.send('2+2');
  const firstId = store.current.id;
  store.newChat();
  await store.send('다른 대화');
  store.select(firstId);
  const restored = new ChatStore(storage);
  assert.equal(restored.chats.length, 2);
  assert.equal(restored.current.id, firstId);
  assert.equal(restored.current.messages[1].content, '4');
});

test('failed requests preserve input and retry sends the question exactly once', async () => {
  let attempts = 0;
  const requests = [];
  const store = new ChatStore(memory(), async (_, init) => {
    requests.push(JSON.parse(init.body));
    if (++attempts === 1) return { ok: false, status: 413,
      json: async () => ({ error: { code: 'context_overflow', message: 'too long' } }) };
    return reply('성공');
  });
  await store.send('질문');
  const question = store.current.messages[0];
  assert.ok(question.error.includes('새 대화'));
  assert.equal(store.busy, false);
  await store.retry(question.id);
  assert.deepEqual(requests[1].messages, [{ role: 'user', content: '질문' }]);
  assert.equal(store.current.messages.length, 2);
  assert.equal(store.current.messages[0].error, undefined);
});

test('duplicate submission is blocked and response stays with its original conversation', async () => {
  let finish;
  let calls = 0;
  const store = new ChatStore(memory(), () => {
    calls++;
    return new Promise(resolve => { finish = resolve; });
  });
  await store.send('');
  const other = store.current;
  other.messages.push({ id: 'seed', role: 'user', content: '다른 대화' });
  store.newChat();
  const original = store.current;
  const pending = store.send('질문');
  await store.send('중복');
  assert.equal(calls, 1);
  assert.equal(store.busy, true);
  store.select(other.id);
  finish(reply('답변'));
  await pending;
  assert.equal(original.messages.length, 2);
  assert.equal(store.current.id, other.id);
  assert.equal(store.current.messages.length, 1);
  assert.equal(store.busy, false);
});

test('reload recovers an interrupted request as a retryable failed question', async () => {
  const saved = { activeId: 'a', chats: [{ id: 'a', title: 'interrupted', messages: [
    { id: 'q', role: 'user', content: 'question', pending: true },
  ] }] };
  const store = new ChatStore({ getItem: () => JSON.stringify(saved), setItem: () => {} }, async () => reply('recovered'));
  assert.equal(store.busy, false);
  assert.ok(store.current.messages[0].error);
  assert.equal(store.current.messages[0].pending, undefined);
  await store.retry('q');
  assert.equal(store.current.messages.at(-1).content, 'recovered');
});

test('corrupted or unavailable storage leaves the chat usable', async () => {
  for (const value of ['{bad', '{"chats":[{"id":"bad","messages":null}]}']) {
    const store = new ChatStore({ getItem: () => value, setItem: () => {} }, async () => reply('ok'));
    await store.send('hello');
    assert.equal(store.current.messages.at(-1).content, 'ok');
  }
  const store = new ChatStore({ getItem: () => { throw Error('blocked'); },
    setItem: () => { throw Error('quota'); } }, async () => reply('ok'));
  await store.send('hello');
  assert.equal(store.current.messages.at(-1).content, 'ok');
  assert.ok(store.storageWarning);
});

test('markdown escapes HTML and unsafe links while keeping PubMed links and code readable', () => {
  const rendered = renderMarkdown('**요약**\n[PMID](https://pubmed.ncbi.nlm.nih.gov/22745249/)\n'
    + '<img src=x onerror=alert(1)>\n[attack](javascript:alert(1))\n```python\nprint("<script>")\n```');
  assert.ok(rendered.includes('<strong>요약</strong>'));
  assert.ok(rendered.includes('href="https://pubmed.ncbi.nlm.nih.gov/22745249/"'));
  assert.ok(rendered.includes('rel="noopener noreferrer"'));
  assert.ok(!rendered.includes('<img'));
  assert.ok(!rendered.includes('href="javascript:'));
  assert.ok(rendered.includes('&lt;script&gt;'));
  assert.ok(rendered.includes('<pre>'));
});

test('native fetch is invoked without a ChatStore receiver', async () => {
  const store = new ChatStore(memory(), function () {
    assert.equal(this, undefined, 'native Window.fetch rejects a custom receiver');
    return Promise.resolve(reply('4'));
  });
  await store.send('2+2');
  assert.equal(store.current.messages.at(-1).content, '4');
});

test('invalid persisted tool metadata is normalized before rendering', () => {
  const saved = { activeId: 'a', chats: [{ id: 'a', title: 'history', messages: [
    { id: 'q', role: 'user', content: 'question' },
    { id: 'r', role: 'assistant', content: 'answer', toolCalls: [null, 42,
      { name: 'pubmed_search_articles', arguments: { query: 'CRISPR' } }], pending: false },
  ] }] };
  const store = new ChatStore({ getItem: () => JSON.stringify(saved), setItem: () => {} });
  assert.deepEqual(store.current.messages[1].toolCalls,
    [{ name: 'pubmed_search_articles', arguments: { query: 'CRISPR' } }]);
});

test('only the latest failed question can be retried after subsequent turns', async () => {
  const requests = [];
  const store = new ChatStore(memory(), async (_, init) => {
    requests.push(JSON.parse(init.body));
    if (requests.length === 1) throw new TypeError('offline');
    return reply('answer B');
  });
  await store.send('A');
  const failedId = store.current.messages[0].id;
  await store.send('B');
  await store.retry(failedId);
  assert.equal(requests.length, 2);
  assert.deepEqual(store.current.messages.map(m => m.content), ['A', 'B', 'answer B']);
});

test('single-asterisk emphasis renders as italics without touching arithmetic', () => {
  const rendered = renderMarkdown('*Nonomuraea gerenzanensis* 균주\n2 * 3 * 4 = 24\n**굵게**');
  assert.ok(rendered.includes('<em>Nonomuraea gerenzanensis</em>'));
  assert.ok(rendered.includes('2 * 3 * 4 = 24'));
  assert.ok(rendered.includes('<strong>굵게</strong>'));
});
