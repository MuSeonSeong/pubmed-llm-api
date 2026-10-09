import test from 'node:test';
import assert from 'node:assert/strict';
import { ATTACHMENT_LIMIT, ChatStore, chatMarkdown, highlight, normalizeSources, readAttachments, readEvents, renderMarkdown,
  withoutCitationList } from '../app/static/chat.mjs';

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
  await tick();  // the first request reaches fetch once it holds the chat's lock
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
  await store.refreshLive();  // no tab holds the chat's lock, so the request died with its page
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


const encoder = new TextEncoder();
const sseText = events => events.map(event => 'data: ' + JSON.stringify(event) + '\n\n').join('');
const done = (content, finish_reason = 'stop', tool_calls = []) =>
  ({ type: 'done', choices: [{ message: { content }, finish_reason }], tool_calls });
const sse = events => new Response(sseText(events), { headers: { 'content-type': 'text/event-stream' } });

/** A stream the test feeds event by event; aborting the request errors it like real fetch. */
function controlledStream(signal) {
  let controller;
  const body = new ReadableStream({ start(c) { controller = c; } });
  signal.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
  return { response: new Response(body, { headers: { 'content-type': 'text/event-stream' } }),
    push: event => controller.enqueue(encoder.encode(sseText([event]))), close: () => controller.close() };
}
const tick = () => new Promise(resolve => setTimeout(resolve, 5));

test('streamed deltas render progressively and the final event carries citations and traces', async () => {
  const bodies = [];
  const snapshots = [];
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([{ type: 'status', stage: 'generating' },
      { type: 'tool', status: 'running', name: 'pubmed_search_articles', arguments: { query: 'CRISPR' } },
      { type: 'tool', status: 'done', name: 'pubmed_search_articles', arguments: { query: 'CRISPR' } },
      { type: 'delta', content: '요약' }, { type: 'delta', content: '입니다' },
      done('요약입니다\n\n출처: PMID 1', 'stop', [{ name: 'pubmed_search_articles', arguments: { query: 'CRISPR' } }])]);
  });
  store.onChange = () => snapshots.push(store.current.messages.at(-1).content);
  await store.send('CRISPR 논문');
  assert.equal(bodies[0].stream, true);
  assert.ok(snapshots.includes('요약'));
  const answer = store.current.messages.at(-1);
  assert.equal(answer.content, '요약입니다\n\n출처: PMID 1');
  assert.deepEqual(answer.toolCalls, [{ name: 'pubmed_search_articles', arguments: { query: 'CRISPR' } }]);
  assert.equal(answer.streaming, undefined);
  assert.equal(store.busy, false);
});

test('stopping mid-answer keeps the partial text as a stopped answer', async () => {
  let stream;
  const store = new ChatStore(memory(), async (_, init) => (stream = controlledStream(init.signal)).response);
  const pending = store.send('긴 설명');
  await tick();
  stream.push({ type: 'delta', content: '부분 답변' });
  await tick();
  assert.equal(store.current.messages.at(-1).content, '부분 답변');
  store.stop();
  await pending;
  const answer = store.current.messages.at(-1);
  assert.equal(answer.role, 'assistant');
  assert.equal(answer.stopped, true);
  assert.equal(answer.content, '부분 답변');
  assert.equal(store.current.messages[0].error, undefined);
  assert.equal(store.busy, false);
});

test('stopping before any text leaves a retryable question', async () => {
  let attempts = 0;
  const store = new ChatStore(memory(), async (_, init) => {
    if (++attempts === 1) return controlledStream(init.signal).response;
    return sse([done('다시 받은 답')]);
  });
  const pending = store.send('질문');
  await tick();
  store.stop();
  await pending;
  const question = store.current.messages.at(-1);
  assert.equal(question.role, 'user');
  assert.ok(question.error.includes('중지'));
  assert.ok(store.canRetry(question.id));
  await store.retry(question.id);
  assert.equal(store.current.messages.at(-1).content, '다시 받은 답');
});

test('regenerate replaces only the last answer and does not resend it', async () => {
  const bodies = [];
  let count = 0;
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([done('답 ' + ++count)]);
  });
  await store.send('질문');
  const first = store.current.messages.at(-1);
  assert.ok(store.canRegenerate(first.id));
  await store.regenerate(first.id);
  assert.deepEqual(store.current.messages.map(m => m.content), ['질문', '답 2']);
  assert.deepEqual(bodies[1].messages, [{ role: 'user', content: '질문' }]);
});

test('editing a question discards later turns and asks again', async () => {
  const bodies = [];
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([done('답 ' + bodies.length)]);
  });
  await store.send('첫 질문');
  await store.send('둘째 질문');
  await store.edit(store.current.messages[0].id, '고친 질문');
  assert.deepEqual(store.current.messages.map(m => m.content), ['고친 질문', '답 3']);
  assert.deepEqual(bodies[2].messages, [{ role: 'user', content: '고친 질문' }]);
});

test('stream error events and truncated streams become retryable failures', async () => {
  for (const events of [[{ type: 'delta', content: '부분' }, { type: 'error', status: 413,
    error: { code: 'context_overflow', message: 'too long' } }], [{ type: 'delta', content: '부분' }]]) {
    const store = new ChatStore(memory(), async () => sse(events));
    await store.send('질문');
    assert.equal(store.current.messages.length, 1);
    assert.ok(store.current.messages[0].error);
    assert.ok(store.canRetry(store.current.messages[0].id));
  }
});

test('reload turns interrupted drafts into stopped answers or retryable questions', async () => {
  const saved = { activeId: 'a', chats: [
    { id: 'a', title: 'with text', messages: [{ id: 'q', role: 'user', content: 'q', pending: true },
      { id: 'd', role: 'assistant', content: '절반', streaming: true }] },
    { id: 'b', title: 'empty', messages: [{ id: 'q2', role: 'user', content: 'q', pending: true },
      { id: 'd2', role: 'assistant', content: '', streaming: true }] }] };
  const store = new ChatStore({ getItem: () => JSON.stringify(saved), setItem: () => {} });
  await store.refreshLive();
  const [withText, empty] = store.chats;
  assert.equal(withText.messages[1].stopped, true);
  assert.equal(withText.messages[0].error, undefined);
  assert.deepEqual(empty.messages.map(m => m.id), ['q2']);
  assert.ok(empty.messages[0].error);
});

test('custom instructions and generation settings are sent and persisted', async () => {
  const storage = memory();
  const bodies = [];
  const store = new ChatStore(storage, async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([done('네')]);
  });
  store.updateSettings({ systemPrompt: '항상 존댓말로 답해줘.', temperature: 0.7, maxTokens: 1024 });
  await store.send('안녕');
  assert.deepEqual(bodies[0].messages[0], { role: 'system', content: '항상 존댓말로 답해줘.' });
  assert.equal(bodies[0].temperature, 0.7);
  assert.equal(bodies[0].max_tokens, 1024);
  store.updateSettings({ temperature: 9, maxTokens: 1.5 });
  const restored = new ChatStore(storage);
  assert.equal(restored.settings.systemPrompt, '항상 존댓말로 답해줘.');
  assert.equal(restored.settings.temperature, 0.2);
  assert.equal(restored.settings.maxTokens, 512);
});

test('rename trims titles and ignores blanks', () => {
  const store = new ChatStore(memory());
  const chatId = store.current.id;
  store.rename(chatId, '   ');
  assert.equal(store.current.title, '새 대화');
  store.rename(chatId, '  당뇨   연구  ');
  assert.equal(store.current.title, '당뇨 연구');
});

test('event reader handles events split across network chunks', async () => {
  const text = sseText([{ type: 'delta', content: '한글' }, done('끝')]);
  const bytes = encoder.encode(text);
  const body = new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
    controller.close();
  } });
  const events = [];
  for await (const event of readEvents(body)) events.push(event);
  assert.equal(events.length, 2);
  assert.equal(events[0].content, '한글');
});

test('markdown renders tables, quotes, rules, labelled code blocks and bare links', () => {
  const rendered = renderMarkdown('| 약물 | 효과 |\n|---|:---:|\n| A | **좋음** |\n\n> 인용\n\n---\n'
    + '자세히는 https://pubmed.ncbi.nlm.nih.gov/1/. 참고\n```js\nconst a = 1;\n```');
  assert.ok(rendered.includes('<table>'));
  assert.ok(rendered.includes('<th>약물</th>'));
  assert.ok(rendered.includes('<td><strong>좋음</strong></td>'));
  assert.ok(rendered.includes('<blockquote>인용</blockquote>'));
  assert.ok(rendered.includes('<hr>'));
  assert.ok(rendered.includes('<a href="https://pubmed.ncbi.nlm.nih.gov/1/" target="_blank" rel="noopener noreferrer">https://pubmed.ncbi.nlm.nih.gov/1/</a>. 참고'));
  assert.ok(rendered.includes('<span>js</span>'));
  assert.ok(rendered.includes('class="code-copy"'));
});

test('bare links cannot break out of the href attribute', () => {
  const rendered = renderMarkdown('https://x.org/"onmouseover="alert(1)');
  assert.ok(!rendered.includes('"onmouseover'));
  assert.ok(!/href="[^"]*"[^>]*onmouseover/.test(rendered));
});


test('requests ask the server to trim old turns and show how many were dropped', async () => {
  const storage = memory();
  let body;
  const store = new ChatStore(storage, async (_, init) => {
    body = JSON.parse(init.body);
    return sse([{ ...done('답'), dropped_messages: 4 }]);
  });
  await store.send('질문');
  assert.equal(body.truncate_history, true);
  assert.equal(store.current.messages.at(-1).dropped, 4);
  assert.equal(new ChatStore(storage).current.messages.at(-1).dropped, 4);
});


test('thinking mode streams reasoning separately and raises the token floor', async () => {
  let body;
  const store = new ChatStore(memory(), async (_, init) => {
    body = JSON.parse(init.body);
    return sse([{ type: 'reasoning', content: '더해 보면' }, { type: 'reasoning', content: ' 4' },
      { type: 'delta', content: '4' },
      { type: 'done', choices: [{ message: { content: '4', reasoning_content: '더해 보면 4' }, finish_reason: 'stop' }] }]);
  });
  store.current.thinking = true;
  await store.send('2+2?');
  assert.equal(body.thinking, true);
  assert.equal(body.max_tokens, 1024);
  const answer = store.current.messages.at(-1);
  assert.equal(answer.reasoning, '더해 보면 4');
  assert.ok(Number.isFinite(answer.thinkMs));
  assert.equal(answer.thinkStart, undefined);
});

test('reasoning-only answers are kept, and plain chats never send the thinking flag', async () => {
  const bodies = [];
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([{ type: 'done', choices: [{ message: { content: '', reasoning_content: '길게 생각' }, finish_reason: 'length' }] }]);
  });
  await store.send('질문');
  assert.equal(bodies[0].thinking, undefined);
  const answer = store.current.messages.at(-1);
  assert.equal(answer.role, 'assistant');
  assert.equal(answer.truncated, true);
  await store.send('다음');
  assert.deepEqual(bodies[1].messages.map(m => m.content), ['질문', '다음']);
});


test('stopping after the final event keeps the answer complete', async () => {
  let stream;
  const store = new ChatStore(memory(), async (_, init) => (stream = controlledStream(init.signal)).response);
  const pending = store.send('질문');
  await tick();
  stream.push(done('완성된 답'));
  await tick();
  store.stop();
  await pending;
  const answer = store.current.messages.at(-1);
  assert.equal(answer.content, '완성된 답');
  assert.equal(answer.stopped, false);
  assert.equal(store.current.messages[0].error, undefined);
});


test('attachments are sent to the model, kept for display and persisted', async () => {
  const storage = memory();
  const bodies = [];
  const store = new ChatStore(storage, async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([done('요약')]);
  });
  await store.send('', [{ name: 'notes.txt', text: '혈당 측정 결과', truncated: false }]);
  const question = store.current.messages[0];
  assert.equal(question.content, '첨부한 파일의 내용을 요약해 줘.');
  assert.ok(bodies[0].messages[0].content.includes('[첨부 파일: notes.txt]'));
  assert.ok(bodies[0].messages[0].content.includes('혈당 측정 결과'));
  assert.equal(store.current.messages[1].model, 'qwen3.5-2b');
  const restored = new ChatStore(storage);
  assert.equal(restored.current.messages[0].attachments[0].name, 'notes.txt');
  assert.ok(chatMarkdown(restored.current).includes('## Qwen3.5 · 2B'));
});

test('reading attachments enforces a shared size budget and rejects binary files', async () => {
  const file = (name, text) => ({ name, text: async () => text });
  const files = await readAttachments([file('a.txt', 'x'.repeat(ATTACHMENT_LIMIT - 10)), file('b.txt', 'y'.repeat(50))]);
  assert.equal(files[1].text.length, 10);
  assert.equal(files[1].truncated, true);
  assert.equal(files.reduce((sum, f) => sum + f.text.length, 0), ATTACHMENT_LIMIT);
  await assert.rejects(readAttachments([file('img.png', 'PNG\u0000\u0001')]), /텍스트 파일이 아닙니다/);
});


test('two tabs sharing storage keep each other\'s chats and deletions', async () => {
  const storage = memory();
  const tabA = new ChatStore(storage, async () => sse([done('A 답')]));
  const tabB = new ChatStore(storage, async () => sse([done('B 답')]));
  await tabA.send('A 질문');
  tabB.newChat();
  await tabB.send('B 질문');
  tabA.save();
  const titles = chats => chats.filter(c => c.messages.length).map(c => c.title).sort();
  assert.deepEqual(titles(new ChatStore(storage).chats), ['A 질문', 'B 질문']);
  const chatA = tabA.chats.find(c => c.title === 'A 질문').id;
  tabB.sync();
  tabB.remove(chatA);
  tabA.save();  // A still holds the chat in memory but must not resurrect it
  assert.deepEqual(titles(new ChatStore(storage).chats), ['B 질문']);
});

test('a chat another tab is answering stays live and locked until that tab finishes', async () => {
  const storage = memory();
  let stream;
  const owner = new ChatStore(storage, async (_, init) => (stream = controlledStream(init.signal)).response);
  const pending = owner.send('질문');
  await tick();
  const watcher = new ChatStore(storage, async () => sse([done('중복')]));
  await watcher.refreshLive();
  const chatId = owner.current.id;
  watcher.select(chatId);
  assert.ok(watcher.liveChats.has(chatId));
  assert.equal(watcher.current.messages[0].error, undefined);
  assert.equal(watcher.canModify(), false);
  await watcher.send('끼어들기');
  watcher.rename(chatId, '바뀐 제목');
  assert.equal(watcher.current.messages.length, 2);  // question and live draft only
  stream.push({ type: 'delta', content: '부분' });
  await tick();
  stream.push(done('완성'));
  stream.close();
  await pending;
  await watcher.sync();
  assert.equal(watcher.liveChats.has(chatId), false);
  assert.deepEqual(watcher.current.messages.map(m => m.content), ['질문', '완성']);
  assert.equal(watcher.canModify(), true);
});

test('a chat whose answering tab vanished is settled by the next check', async () => {
  const storage = memory();
  storage.setItem('pubmed-chat.v1', JSON.stringify({ chats: [] }));
  const tab = new ChatStore(storage);
  storage.setItem('pubmed-chat.v1', JSON.stringify({ chats: [{ id: 'gone', title: 't', rev: Date.now() + 1000,
    messages: [{ id: 'q', role: 'user', content: 'q', pending: true },
      { id: 'd', role: 'assistant', content: '', streaming: true }] }] }));
  await tab.sync();
  const chat = tab.chats.find(c => c.id === 'gone');
  assert.equal(tab.liveChats.has('gone'), false);
  assert.deepEqual(chat.messages.map(m => m.id), ['q']);
  assert.ok(chat.messages[0].error);
});

test('option changes and version switches win over a stale save from another tab', async () => {
  const storage = memory();
  let count = 0;
  const tabA = new ChatStore(storage, async () => sse([done('답 ' + ++count)]));
  await tabA.send('질문');
  await tabA.regenerate(tabA.current.messages.at(-1).id);
  const tabB = new ChatStore(storage);
  await tick();
  tabA.selectVersion(tabA.current.messages.at(-1).id, 0);
  tabA.updateChat({ model: 'qwen3-0.6b' });
  tabB.newChat();  // B saves its older copy of the chat
  const chat = new ChatStore(storage).chats.find(c => c.id === tabA.current.id);
  assert.equal(chat.model, 'qwen3-0.6b');
  assert.equal(chat.messages.at(-1).content, '답 1');
});


test('regenerated answers are kept as versions that can be browsed and persist', async () => {
  const storage = memory();
  let count = 0;
  const store = new ChatStore(storage, async () => sse([done('답 ' + ++count)]));
  await store.send('질문');
  await store.regenerate(store.current.messages.at(-1).id);
  await store.regenerate(store.current.messages.at(-1).id);
  let answer = store.current.messages.at(-1);
  assert.deepEqual(answer.versions.map(v => v.content), ['답 1', '답 2', '답 3']);
  assert.equal(answer.version, 2);
  store.selectVersion(answer.id, 0);
  assert.equal(store.current.messages.at(-1).content, '답 1');
  const restored = new ChatStore(storage);
  answer = restored.current.messages.at(-1);
  assert.equal(answer.content, '답 1');
  assert.equal(answer.version, 0);
  assert.equal(answer.versions.length, 3);
});

test('a failed regeneration restores the previous answer and reports why', async () => {
  let attempts = 0;
  const store = new ChatStore(memory(), async () => {
    if (++attempts === 2) throw new TypeError('offline');
    return sse([done('원래 답')]);
  });
  await store.send('질문');
  await store.regenerate(store.current.messages.at(-1).id);
  assert.deepEqual(store.current.messages.map(m => m.content), ['질문', '원래 답']);
  assert.equal(store.current.messages[0].error, undefined);
  assert.ok(store.notice.includes('서버'));
});


test('sources are normalized, rebuilt from the PMID, persisted and kept per version', async () => {
  const storage = memory();
  const store = new ChatStore(storage, async () => sse([{ ...done('답\n\nPubMed 검색·조회 출처:\n- PMID [1](x)'),
    sources: [{ pmid: '1', url: 'javascript:alert(1)', title: '논문', journal: 'J', year: '2020', abstract_fetched: true },
      { pmid: 'bad' }] }]));
  await store.send('질문');
  const answer = new ChatStore(storage).current.messages.at(-1);
  assert.deepEqual(answer.sources, [{ pmid: '1', url: 'https://pubmed.ncbi.nlm.nih.gov/1/', title: '논문',
    journal: 'J', year: '2020', abstractFetched: true }]);
  assert.equal(withoutCitationList(answer.content), '답');
  assert.equal(normalizeSources('nope').length, 0);
});


test('answers record token usage and elapsed time', async () => {
  const store = new ChatStore(memory(), async () => sse([{ ...done('답'), usage: { completion_tokens: 42 } }]));
  await store.send('질문');
  const answer = store.current.messages.at(-1);
  assert.equal(answer.tokens, 42);
  assert.ok(answer.elapsedMs >= 0);
});


test('lists nest by indentation, keep numbering across blank lines and absorb continuation lines', () => {
  const html = renderMarkdown('1. 첫째\n   설명 줄\n   - 하위 A\n   - 하위 B\n\n2. 둘째\n\n문단 ~~취소~~');
  assert.equal(html, '<ol><li>첫째<br>설명 줄<ul><li>하위 A</li><li>하위 B</li></ul></li><li>둘째</li></ol>'
    + '<p>문단 <del>취소</del></p>');
  assert.equal(renderMarkdown('3. 셋째\n4. 넷째'), '<ol start="3"><li>셋째</li><li>넷째</li></ol>');
  assert.equal(renderMarkdown('- a\n1. b'), '<ul><li>a</li></ul><ol><li>b</li></ol>');
});


test('pinning toggles, persists and survives reload', () => {
  const storage = memory();
  const store = new ChatStore(storage);
  const chatId = store.current.id;
  store.togglePin(chatId);
  assert.equal(new ChatStore(storage).chats.find(c => c.id === chatId).pinned, true);
  store.togglePin(chatId);
  assert.equal(new ChatStore(storage).chats.find(c => c.id === chatId).pinned, false);
});


test('continuing a cut-off answer appends to it and sends the partial answer back', async () => {
  const bodies = [];
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return bodies.length === 1 ? sse([{ ...done('봄에는', 'length'), usage: { completion_tokens: 3 } }])
      : sse([{ type: 'delta', content: ' 꽃이' }, { ...done(' 꽃이 핍니다.'), usage: { completion_tokens: 4 } }]);
  });
  store.current.thinking = true;
  await store.send('사계절');
  const answer = store.current.messages.at(-1);
  assert.ok(store.canContinue(answer.id));
  await store.continueAnswer(answer.id);
  assert.equal(bodies[1].continue_final, true);
  assert.equal(bodies[1].use_pubmed, false);
  assert.equal(bodies[1].thinking, undefined);
  assert.deepEqual(bodies[1].messages.at(-1), { role: 'assistant', content: '봄에는' });
  assert.equal(store.current.messages.length, 2);
  assert.equal(answer.content, '봄에는 꽃이 핍니다.');
  assert.equal(answer.truncated, false);
  assert.equal(answer.tokens, 7);
});

test('a failed continuation leaves the answer untouched', async () => {
  let calls = 0;
  const store = new ChatStore(memory(), async () => {
    if (++calls === 2) throw new TypeError('offline');
    return sse([done('봄에는', 'length')]);
  });
  await store.send('사계절');
  const answer = store.current.messages.at(-1);
  await store.continueAnswer(answer.id);
  assert.equal(answer.content, '봄에는');
  assert.equal(answer.truncated, true);
  assert.equal(answer.streaming, undefined);
  assert.ok(store.notice);
});


test('code highlighting colours tokens and escapes everything', () => {
  const html = highlight('def f(x):  # 주석 <b>\n    return "<script>" + 42', 'python');
  assert.ok(html.includes('<span class="tok-keyword">def</span>'));
  assert.ok(html.includes('<span class="tok-comment"># 주석 &lt;b&gt;</span>'));
  assert.ok(html.includes('<span class="tok-string">&quot;&lt;script&gt;&quot;</span>'));
  assert.ok(html.includes('<span class="tok-number">42</span>'));
  assert.ok(!html.includes('<script>') && !html.includes('<b>'));
  assert.equal(highlight('<x> if', 'unknown'), '&lt;x&gt; if');
  assert.ok(highlight('SELECT * FROM t', 'sql').includes('<span class="tok-keyword">SELECT</span>'));
  assert.ok(renderMarkdown('```js\nconst a = `x`;\n```').includes('<span class="tok-string">`x`</span>'));
});


test('a chat created in one tab picks up a conversation another tab started in it', async () => {
  const storage = memory();
  const tabA = new ChatStore(storage);
  tabA.save();
  const tabB = new ChatStore(storage, async () => sse([done('답')]));
  await tabB.send('hello');
  await tabA.sync();
  assert.equal(tabA.current.messages.length, 2);
  tabA.select(tabA.current.id);  // any save from A must not wipe B's turn
  assert.equal(new ChatStore(storage).current.messages.length, 2);
});

test('a released lock never settles an older copy over the final answer', async () => {
  const storage = memory();
  let stream;
  const owner = new ChatStore(storage, async (_, init) => (stream = controlledStream(init.signal)).response);
  const pending = owner.send('hello');
  await tick();
  const watcher = new ChatStore(storage);
  await watcher.refreshLive();
  assert.ok(watcher.liveChats.has(owner.current.id));
  stream.push(done('full answer'));
  stream.close();
  await pending;
  await watcher.refreshLive();  // before any storage event reached it
  const chat = watcher.chats.find(c => c.id === owner.current.id);
  assert.deepEqual(chat.messages.map(m => m.content), ['hello', 'full answer']);
  assert.equal(chat.messages[0].error, undefined);
  watcher.save();
  assert.deepEqual(new ChatStore(storage).current.messages.map(m => m.content), ['hello', 'full answer']);
});

test('continuing a cited answer extends the text and keeps the citation list last', async () => {
  const bodies = [];
  const cited = '봄에는\n\nPubMed 검색·조회 출처:\n- PMID [1](https://pubmed.ncbi.nlm.nih.gov/1/)';
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return bodies.length === 1 ? sse([{ ...done(cited, 'length'), sources: [{ pmid: '1' }] }]) : sse([done(' 꽃이 핍니다.')]);
  });
  await store.send('사계절');
  const answer = store.current.messages.at(-1);
  await store.continueAnswer(answer.id);
  assert.equal(bodies[1].messages.at(-1).content, '봄에는');
  assert.equal(answer.content, '봄에는 꽃이 핍니다.\n\nPubMed 검색·조회 출처:\n- PMID [1](https://pubmed.ncbi.nlm.nih.gov/1/)');
  assert.equal(withoutCitationList(answer.content), '봄에는 꽃이 핍니다.');
});

test('a question refused because another tab holds the chat is not half-added', async () => {
  const storage = memory();
  const store = new ChatStore(storage, async () => sse([done('답')]));
  let release;
  await new Promise(resolve => navigator.locks.request('pubmed-chat:' + store.current.id,
    () => new Promise(done => { release = done; resolve(); })));
  await store.send('질문');
  assert.equal(store.current.messages.length, 0);
  assert.equal(store.busy, false);
  assert.ok(store.notice);
  release();
});


const PNG = 'data:image/png;base64,iVBORw0KGgo=';
const image = name => ({ name, kind: 'image', dataUrl: PNG });

test('images go to vision models as parts, newest first and at most four per request', async () => {
  const bodies = [];
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([done('보여요')]);
  });
  await store.send('', [image('a.png'), image('b.png'), image('c.png')]);
  assert.equal(store.current.messages[0].content, '이 이미지를 설명해 줘.');
  assert.deepEqual(bodies[0].messages[0].content.map(part => part.type), ['image_url', 'image_url', 'image_url', 'text']);
  await store.send('이것도', [image('d.png'), image('e.png')]);
  const [first, , second] = bodies[1].messages;
  assert.equal(first.content.filter(part => part.type === 'image_url').length, 2);  // only the newest four go
  assert.ok(first.content.at(-1).text.includes('[이미지 생략: a.png]'));
  assert.equal(second.content.filter(part => part.type === 'image_url').length, 2);
  store.updateChat({ model: 'qwen3-0.6b' });
  await store.send('텍스트 모델');
  assert.equal(typeof bodies[2].messages[0].content, 'string');
  assert.ok(bodies[2].messages[0].content.includes('[이미지 생략: c.png]'));
});

test('stored images must be base64 raster data URLs', () => {
  const saved = { activeId: 'x', chats: [{ id: 'x', title: 't', messages: [{ id: 'u', role: 'user', content: 'q', attachments: [
    image('ok.png'), { name: 'svg', kind: 'image', dataUrl: 'data:image/svg+xml;base64,PHN2Zz4=' },
    { name: 'js', kind: 'image', dataUrl: 'javascript:alert(1)' }] }] }] };
  const files = new ChatStore({ getItem: () => JSON.stringify(saved), setItem: () => {} }).current.messages[0].attachments;
  assert.deepEqual(files.map(f => [f.name, Boolean(f.dataUrl), f.removed === true]),
    [['ok.png', true, false], ['svg', false, true], ['js', false, true]]);
});

test('when storage is full, old chats lose their pictures before history is lost', async () => {
  const data = new Map();
  const storage = { getItem: key => data.get(key) ?? null, setItem: (key, value) => {
    if (value.length > 600 && value.includes(PNG)) throw new Error('QuotaExceededError');
    data.set(key, value);
  } };
  const store = new ChatStore(storage, async () => sse([done('답')]));
  store.chats[0].messages.push({ id: 'old', role: 'user', content: '예전', attachments: [image('old.png')] });
  store.chats[0].updatedAt = 1;
  store.newChat();
  store.save();
  const restored = new ChatStore(storage).chats.find(c => c.messages.length);
  assert.equal(restored.messages[0].attachments[0].removed, true);
  assert.ok(store.storageWarning.includes('이미지'));
});


test('regenerating with another model switches the chat and keeps both versions', async () => {
  const bodies = [];
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([done('답 ' + bodies.length)]);
  });
  await store.send('질문');
  await store.regenerate(store.current.messages.at(-1).id, { model: 'qwen3-0.6b' });
  assert.equal(bodies[1].model, 'qwen3-0.6b');
  assert.equal(store.current.model, 'qwen3-0.6b');
  const answer = store.current.messages.at(-1);
  assert.deepEqual(answer.versions.map(v => v.model), ['qwen3.5-2b', 'qwen3-0.6b']);
});


test('code fences named after object properties render as plain code', () => {
  for (const name of ['constructor', '__proto__', 'toString']) {
    assert.ok(renderMarkdown('```' + name + '\nlet x = 1\n```').includes('let x = 1'));
  }
});
