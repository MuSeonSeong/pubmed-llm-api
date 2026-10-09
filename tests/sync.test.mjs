import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ChatStore, readEvents } from '../app/static/chat.mjs';
import { LEGACY_KEY, RemoteSync, forgetCache, importLegacy, legacyChats, skipLegacy, syncKey, uploadImage }
  from '../app/static/sync.mjs';

const memory = (entries = {}) => {
  const data = new Map(Object.entries(entries));
  return { data, getItem: key => data.get(key) ?? null, setItem: (key, value) => { data.set(key, String(value)); },
    removeItem: key => { data.delete(key); } };
};
const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const failure = (status, code, extra = {}) => json(status, { error: { code, message: code, details: {} }, ...extra });
const encoder = new TextEncoder();
const sseText = events => events.map(event => 'data: ' + JSON.stringify(event) + '\n\n').join('');
const sse = events => new Response(sseText(events), { headers: { 'content-type': 'text/event-stream' } });
const done = content => ({ type: 'done', choices: [{ message: { content }, finish_reason: 'stop' }], tool_calls: [] });
const tick = () => new Promise(resolve => setTimeout(resolve, 5));
const PNG = 'data:image/png;base64,iVBORw0KGgo=';

/** A stream the test feeds event by event; aborting the request errors it like real fetch. */
function controlledStream(signal) {
  let controller;
  const body = new ReadableStream({ start(c) { controller = c; } });
  signal.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
  return { response: new Response(body, { headers: { 'content-type': 'text/event-stream' } }),
    push: event => controller.enqueue(encoder.encode(sseText([event]))), close: () => controller.close() };
}

/** One user's data on a server that follows the HTTP contract of the real one, kept in memory. */
function fakeServer({ maxChatBytes = 4 * 1024 * 1024 } = {}) {
  const server = { seq: 0, rows: new Map(), settings: null, files: new Map(), live: new Set(), log: [],
    completions: [], offline: false, signedOut: false, rejectUploads: false, maxChatBytes,
    respond: () => sse([done('답')]) };
  server.seed = chat => server.rows.set(chat.id, { rev: chat.rev, seq: ++server.seq, data: structuredClone(chat) });
  server.tombstone = chatId => server.rows.set(chatId, { rev: 0, seq: ++server.seq, data: null });
  server.chat = chatId => server.rows.get(chatId)?.data;
  server.count = prefix => server.log.filter(line => line.startsWith(prefix)).length;
  server.fetch = async (url, init = {}) => {
    const method = init.method || 'GET';
    const { pathname, search } = new URL(url, 'http://server');
    server.log.push(method + ' ' + pathname + search);
    if (server.offline) throw new TypeError('fetch failed');
    if (server.signedOut) return failure(401, 'unauthorized');
    if (pathname === '/api/chats' && method === 'GET') {
      const since = Number(new URLSearchParams(search).get('since') || 0);
      const changed = [...server.rows].filter(([, row]) => row.seq > since);
      return json(200, { seq: server.seq, chats: changed.filter(([, row]) => row.data).map(([, row]) => row.data),
        deleted: changed.filter(([, row]) => !row.data).map(([chatId]) => chatId), live: [...server.live] });
    }
    const chatId = pathname.match(/^\/api\/chats\/([A-Za-z0-9_-]{1,64})$/)?.[1];
    if (chatId && method === 'PUT') {
      if (encoder.encode(init.body).length > server.maxChatBytes) return failure(413, 'chat_too_large');
      const chat = JSON.parse(init.body);
      const row = server.rows.get(chatId);
      if (server.alwaysStale) {
        await tick();  // a loop then yields to timers, so a test timeout can still end it
        return failure(409, 'stale_chat', { chat: { ...chat, rev: chat.rev + 1 } });
      }
      if (row && !row.data) return failure(410, 'chat_deleted');
      if (row && row.rev > chat.rev) return failure(409, 'stale_chat', { chat: row.data });
      if (row && row.rev === chat.rev) return json(200, { rev: row.rev, seq: row.seq });
      server.rows.set(chatId, { rev: chat.rev, seq: ++server.seq, data: chat });
      return json(200, { rev: chat.rev, seq: server.seq });
    }
    if (chatId && method === 'DELETE') {
      server.tombstone(chatId);
      return new Response(null, { status: 204 });
    }
    if (pathname === '/api/settings' && method === 'GET') return json(200, { settings: server.settings });
    if (pathname === '/api/settings' && method === 'PUT') {
      server.settings = JSON.parse(init.body);
      return new Response(null, { status: 204 });
    }
    if (pathname === '/api/files' && method === 'POST') {
      if (server.rejectUploads) return failure(422, 'invalid_image');
      const bytes = new Uint8Array(await new Response(init.body).arrayBuffer());
      const id = 'file' + String(server.files.size + 1).padStart(6, '0');
      server.files.set(id, { mime: init.headers['Content-Type'], bytes });
      return json(201, { id, url: '/api/files/' + id, mime: init.headers['Content-Type'], size: bytes.length });
    }
    if (pathname === '/v1/chat/completions') {
      const body = JSON.parse(init.body);
      server.completions.push(body);
      return server.respond(body, init);
    }
    return failure(404, 'not_found');
  };
  return server;
}

/** setTimeout for tests: nothing runs until advance() moves the clock past it. */
function clock() {
  let now = 0;
  let timers = [];
  return {
    now: () => now,
    setTimeout: (callback, ms) => {
      const timer = { callback, at: now + ms };
      timers.push(timer);
      return timer;
    },
    clearTimeout: timer => { timers = timers.filter(item => item !== timer); },
    delays: () => timers.map(timer => timer.at - now).sort((a, b) => a - b),
    async advance(ms) {
      const end = now + ms;
      for (let next; (next = timers.filter(timer => timer.at <= end).sort((a, b) => a.at - b.at)[0]);) {
        timers = timers.filter(timer => timer !== next);
        now = next.at;
        next.callback();
        await tick();
      }
      now = end;
    },
  };
}

const KEY = 'pubmed-chat.v2:tester';
/** A browser of the test user: its store and sync. Timers never fire unless a clock is passed. */
function device({ server = fakeServer(), storage = memory(), store: storeOptions = {}, sync: syncOptions = {} } = {}) {
  const store = new ChatStore(storage, server.fetch, { storageKey: KEY, cacheOnly: true, ...storeOptions });
  const statuses = [];
  const notices = [];
  const sync = new RemoteSync(store, { fetcher: server.fetch, storage, document: null,
    setTimeout: () => null, clearTimeout: () => {}, onStatus: status => statuses.push(status),
    onNotice: text => notices.push(text), ...syncOptions });
  return { server, storage, store, sync, statuses, notices };
}
const timed = time => ({ setTimeout: time.setTimeout, clearTimeout: time.clearTimeout, now: time.now });

const chatOf = (chatId, rev, contents = ['질문', '답']) => ({ id: chatId, title: contents[0] || '새 대화',
  model: 'qwen3.5-2b', usePubmed: true, thinking: false, pinned: false, updatedAt: rev, rev,
  messages: contents.map((content, index) => ({ id: chatId + '-' + index, role: index % 2 ? 'assistant' : 'user',
    content, toolCalls: [], truncated: false, stopped: false })) });
const plain = value => JSON.parse(JSON.stringify(value));


test('a store keeps its copy under its own key and reports every save', () => {
  const storage = memory();
  const store = new ChatStore(storage, undefined, { storageKey: 'pubmed-chat.v2:민지' });
  let saves = 0;
  store.onSaved = () => saves++;
  store.togglePin(store.current.id);
  assert.equal(saves, 1);
  assert.equal(storage.getItem('pubmed-chat.v1'), null);
  assert.equal(new ChatStore(storage, undefined, { storageKey: 'pubmed-chat.v2:민지' }).current.pinned, true);
});

test('a save is pushed after a short pause, with saves in between combined into one request', async () => {
  const time = clock();
  const { server, store, sync, statuses } = device({ sync: timed(time) });
  await sync.start();
  assert.equal(sync.status, 'synced');
  assert.equal(server.count('PUT /api/chats'), 0);  // the empty new chat is not uploaded
  await store.send('질문');  // saves when the question is asked and again with the answer
  store.togglePin(store.current.id);
  assert.equal(sync.status, 'syncing');
  assert.equal(server.count('PUT /api/chats'), 0);
  await time.advance(400);
  assert.equal(server.count('PUT /api/chats'), 1);
  assert.deepEqual(server.chat(store.current.id), plain(store.current));
  assert.equal(server.completions[0].chat_id, store.current.id);
  assert.equal(sync.status, 'synced');
  assert.deepEqual(statuses, ['synced', 'syncing', 'synced']);
});

test('pulls ask only for changes since the last one, and the cursor survives a reload', async () => {
  const { server, store, sync, storage } = device();
  server.seed(chatOf('remote-1', 1000, ['다른 기기 질문', '다른 기기 답']));
  await sync.start();
  assert.ok(store.chats.some(chat => chat.id === 'remote-1'));
  server.seed(chatOf('remote-2', 2000, ['새 질문', '새 답']));
  server.seed(chatOf('remote-1', 3000, ['다른 기기 질문', '고친 답']));
  await sync.syncNow();
  assert.equal(store.chats.find(chat => chat.id === 'remote-1').messages[1].content, '고친 답');
  assert.ok(store.chats.some(chat => chat.id === 'remote-2'));
  assert.equal(server.count('PUT /api/chats'), 0);  // what came from the server is not sent back
  const reloaded = device({ server, storage });
  await reloaded.sync.start();
  assert.deepEqual(server.log.filter(line => line.startsWith('GET /api/chats')),
    ['GET /api/chats?since=0', 'GET /api/chats?since=1', 'GET /api/chats?since=3']);
  assert.equal(reloaded.store.chats.find(chat => chat.id === 'remote-1').messages[1].content, '고친 답');
});

test('a push refused as stale adopts the newer copy another device saved', async () => {
  const { server, store, sync } = device();
  await sync.start();
  await store.send('질문');
  const chat = store.current;
  server.seed({ ...plain(chat), title: '다른 기기에서 바꾼 제목', rev: chat.rev + 1000 });
  await sync.flush();
  assert.equal(store.current.title, '다른 기기에서 바꾼 제목');
  assert.equal(store.current.rev, chat.rev + 1000);
  assert.equal(server.count('PUT /api/chats'), 1);
  assert.equal(sync.status, 'synced');
});

test('a chat deleted on another device is deleted here when its push is refused', async () => {
  const { server, store, sync } = device();
  await sync.start();
  await store.send('지울 대화');
  const chatId = store.current.id;
  server.tombstone(chatId);
  await sync.flush();
  assert.ok(!store.chats.some(chat => chat.id === chatId));
  assert.equal(server.count('DELETE'), 0);  // the server has the deletion already
  assert.equal(sync.status, 'synced');
});

/** A server whose answers the test finishes by hand. */
function slowServer() {
  const server = fakeServer();
  server.respond = (body, init) => {
    const stream = controlledStream(init.signal);
    server.finish = text => {
      stream.push(done(text));
      stream.close();
    };
    return stream.response;
  };
  return server;
}

test('a chat deleted elsewhere while this tab answers it is deleted when the answer ends, without a request loop', async () => {
  const { server, store, sync } = device({ server: slowServer() });
  await sync.start();
  const answered = store.send('질문');
  await tick();
  const chatId = store.current.id;
  server.tombstone(chatId);
  await sync.flush();
  assert.equal(server.count('PUT /api/chats'), 1);
  assert.ok(store.chats.some(chat => chat.id === chatId));  // the answer is still being written here
  server.finish('답');
  await answered;
  await sync.flush();
  assert.equal(server.count('PUT /api/chats'), 2);
  assert.ok(!store.chats.some(chat => chat.id === chatId));
  assert.equal(server.count('DELETE'), 0);
});

test('the answering tab\'s copy wins over a newer-looking one saved elsewhere meanwhile', async () => {
  const { server, store, sync } = device({ server: slowServer() });
  await sync.start();
  const answered = store.send('질문');
  await tick();
  const chat = store.current;
  const skewed = chat.rev + 60000;  // e.g. a device whose clock runs fast
  server.seed({ ...plain(chat), title: '다른 기기', rev: skewed });
  await sync.flush();
  assert.equal(server.chat(chat.id).title, '질문');
  assert.ok(server.chat(chat.id).rev > skewed);
  server.finish('완성된 답');
  await answered;
  await sync.flush();
  assert.deepEqual(server.chat(chat.id).messages.map(message => message.content), ['질문', '완성된 답']);
  assert.equal(sync.status, 'synced');
});

test('a server that keeps calling the answering tab\'s copy stale cannot make the sync loop', async () => {
  const { server, store, sync } = device({ server: slowServer() });
  await sync.start();
  const answered = store.send('질문');
  await tick();
  server.alwaysStale = true;
  let timer;
  const bounded = await Promise.race([sync.flush().then(() => true),
    new Promise(resolve => { timer = setTimeout(resolve, 2000, false); })]);
  clearTimeout(timer);
  if (!bounded) sync.stop();  // end a runaway loop, so this test fails instead of never finishing
  assert.equal(bounded, true);
  assert.ok(server.count('PUT /api/chats') > 1 && server.count('PUT /api/chats') <= 15);
  server.alwaysStale = false;
  server.finish('답');
  await answered;
  assert.equal(await sync.flush(), true);
});

test('a deletion is sent once and remembered, also after a reload', async () => {
  const { server, store, sync, storage } = device();
  await sync.start();
  await store.send('지울 대화');
  await sync.flush();
  const chatId = store.current.id;
  store.remove(chatId);
  await sync.flush();
  await sync.syncNow();  // the next pull is past the deletion, so only the remembered confirmation prevents a resend
  assert.deepEqual(server.log.filter(line => line.startsWith('DELETE')), ['DELETE /api/chats/' + chatId]);
  assert.equal(server.chat(chatId), null);
  const reloaded = device({ server, storage });
  await reloaded.sync.start();
  assert.equal(server.count('DELETE'), 1);
  assert.ok(!reloaded.store.chats.some(chat => chat.id === chatId));
});

test('chats deleted on another device disappear here', async () => {
  const { server, store, sync } = device();
  server.seed(chatOf('gone', 1000));
  server.seed(chatOf('kept', 1000));
  await sync.start();
  store.select('gone');
  server.tombstone('gone');
  await sync.syncNow();
  assert.deepEqual(store.chats.filter(chat => chat.messages.length).map(chat => chat.id), ['kept']);
  assert.notEqual(store.activeId, 'gone');
  assert.equal(server.count('DELETE'), 0);
  assert.equal(server.count('PUT /api/chats'), 0);  // not offered back to the server to be refused
});

const answering = () => {
  const chat = chatOf('busy', 1000, ['질문', '부분 답']);
  chat.messages[0].pending = true;
  chat.messages[1].streaming = true;
  return chat;
};

test('a chat another device is answering stays read-only until its final copy arrives', async () => {
  let now = 0;
  const { server, store, sync } = device({ sync: { now: () => now } });
  server.seed(answering());
  server.live.add('busy');
  await sync.start();
  store.select('busy');
  assert.ok(store.liveChats.has('busy'));
  assert.equal(store.canModify(), false);
  // The stream has ended but the final copy is still on its way: not treated as interrupted yet.
  server.live.clear();
  now += 2000;
  await sync.syncNow();
  assert.equal(store.canModify(), false);
  assert.equal(store.current.messages[1].stopped, false);
  server.seed(chatOf('busy', 2000, ['질문', '완성된 답']));
  now += 1500;
  await sync.syncNow();
  assert.equal(store.canModify(), true);
  assert.deepEqual(store.current.messages.map(message => message.content), ['질문', '완성된 답']);
  assert.equal(store.current.messages[1].stopped, false);
});

test('a remote answer whose device vanished is settled here after a grace period, without uploading', async () => {
  let now = 0;
  const { server, store, sync } = device({ sync: { now: () => now } });
  server.seed(answering());
  server.live.add('busy');
  await sync.start();
  store.select('busy');
  server.live.clear();
  now += 2000;
  await sync.syncNow();
  assert.equal(store.canModify(), false);
  now += 10000;
  await sync.syncNow();
  assert.equal(store.canModify(), true);
  const [question, answer] = store.current.messages;
  assert.equal(question.pending, undefined);
  assert.equal(answer.stopped, true);
  assert.equal(answer.streaming, undefined);
  assert.equal(server.count('PUT /api/chats'), 0);  // settling is this browser's view, not a change
});

test('another device follows an answer: read-only while it runs, editable with the final text', async () => {
  const server = fakeServer();
  let finish;
  server.respond = (body, init) => {
    const stream = controlledStream(init.signal);
    server.live.add(body.chat_id);
    finish = text => {
      server.live.delete(body.chat_id);
      stream.push(done(text));
      stream.close();
    };
    return stream.response;
  };
  const phone = device({ server });
  const laptop = device({ server, storage: memory(), store: { locks: null } });  // only the server can tell it
  await phone.sync.start();
  await laptop.sync.start();
  const answered = phone.store.send('질문');
  await tick();
  await phone.sync.flush();  // the question, with the answer still to come, reaches the server
  assert.equal(server.chat(phone.store.current.id).messages[0].pending, true);
  await laptop.sync.syncNow();
  laptop.store.select(phone.store.current.id);
  assert.equal(laptop.store.canModify(), false);
  finish('완성된 답');
  await answered;
  await phone.sync.flush();
  await laptop.sync.syncNow();
  assert.equal(laptop.store.canModify(), true);
  assert.deepEqual(laptop.store.current.messages.map(message => message.content), ['질문', '완성된 답']);
});

test('a chat another tab here is answering is left for that tab to upload', async () => {
  const { server, store, sync } = device();
  await sync.start();
  await store.send('질문');
  store.liveChats.add(store.current.id);
  await sync.flush();
  assert.equal(server.count('PUT /api/chats'), 0);
});

test('settings come from the server when it has them; otherwise this browser\'s are sent', async () => {
  const first = device();
  first.server.settings = { systemPrompt: '항상 한국어로', temperature: 0.5, maxTokens: 1024 };
  await first.sync.start();
  assert.deepEqual(first.store.settings, { systemPrompt: '항상 한국어로', temperature: 0.5, maxTokens: 1024 });
  first.store.updateSettings({ temperature: 0.9 });
  await first.sync.flush();
  assert.deepEqual(first.server.settings, { systemPrompt: '항상 한국어로', temperature: 0.9, maxTokens: 1024 });

  const second = device();
  second.store.updateSettings({ systemPrompt: '내 지시문' });
  await second.sync.start();
  assert.equal(second.server.settings.systemPrompt, '내 지시문');
});

test('settings changed while offline win over the server copy on the next visit', async () => {
  const { server, store, sync, storage } = device();
  server.settings = { systemPrompt: '서버', temperature: 0.2, maxTokens: 512 };
  await sync.start();
  server.offline = true;
  store.updateSettings({ systemPrompt: '오프라인에서 바꿈' });
  await sync.flush();
  assert.equal(sync.status, 'offline');
  server.offline = false;
  const reloaded = device({ server, storage });
  await reloaded.sync.start();
  assert.equal(server.settings.systemPrompt, '오프라인에서 바꿈');
  assert.equal(reloaded.store.settings.systemPrompt, '오프라인에서 바꿈');
});

test('the history from before the server is imported once, with its pictures uploaded', async () => {
  const pictured = chatOf('old-2', 200, ['이 그림은?', '고양이예요']);
  pictured.messages[0].attachments = [{ name: 'cat.png', kind: 'image', dataUrl: PNG }];
  const interrupted = chatOf('old-3', 300, ['끊긴 질문']);
  interrupted.messages[0].pending = true;
  const legacy = { chats: [chatOf('old-1', 100, ['예전 질문', '예전 답']), pictured, interrupted, chatOf('empty', 50, [])],
    activeId: 'old-1', settings: { systemPrompt: '예전 지시문', temperature: 0.4, maxTokens: 1024 } };
  const storage = memory({ [LEGACY_KEY]: JSON.stringify(legacy) });
  const { server, store, sync } = device({ storage });
  await sync.start();
  assert.equal(sync.serverChats, 0);
  assert.equal(legacyChats(storage).length, 3);  // the empty chat is not counted
  const progress = [];
  const result = await importLegacy(store, { fetcher: server.fetch, storage, onProgress: (...step) => progress.push(step) });
  assert.deepEqual(result, { chats: 3, images: 1, uploaded: 1 });
  assert.deepEqual(progress, [[1, 1]]);
  await sync.flush();
  assert.deepEqual(server.chat('old-2').messages[0].attachments,
    [{ name: 'cat.png', kind: 'image', fileId: 'file000001', url: '/api/files/file000001' }]);
  assert.deepEqual([...server.files.get('file000001').bytes], [...Buffer.from('iVBORw0KGgo=', 'base64')]);
  assert.equal(server.files.get('file000001').mime, 'image/png');
  assert.ok(server.chat('old-1').rev > 100);  // touched, so it outranks any older copy
  assert.ok(server.chat('old-3').messages[0].error);  // nothing answers it any more: retryable
  assert.equal(server.chat('empty'), undefined);
  assert.equal(server.settings.systemPrompt, '예전 지시문');
  assert.equal(sync.serverChats, 3);
  assert.equal(storage.getItem(LEGACY_KEY), null);
  assert.equal(storage.getItem(LEGACY_KEY + '.imported'), JSON.stringify(legacy));
  assert.deepEqual(legacyChats(storage), []);
  assert.equal(await importLegacy(store, { fetcher: server.fetch, storage }), null);
});

test('a picture whose upload fails stays inline, and declining an import is remembered', async () => {
  const pictured = chatOf('old', 100, ['이 그림은?', '고양이예요']);
  pictured.messages[0].attachments = [{ name: 'cat.png', kind: 'image', dataUrl: PNG }];
  const storage = memory({ [LEGACY_KEY]: JSON.stringify({ chats: [pictured] }) });
  const { server, store, sync } = device({ storage });
  server.rejectUploads = true;
  await sync.start();
  assert.deepEqual(await importLegacy(store, { fetcher: server.fetch, storage }), { chats: 1, images: 1, uploaded: 0 });
  await sync.flush();
  assert.deepEqual(server.chat('old').messages[0].attachments, [{ name: 'cat.png', kind: 'image', dataUrl: PNG }]);

  const declined = memory({ [LEGACY_KEY]: JSON.stringify({ chats: [pictured] }) });
  assert.equal(legacyChats(declined).length, 1);
  skipLegacy(declined);
  assert.deepEqual(legacyChats(declined), []);
  assert.ok(declined.getItem(LEGACY_KEY));  // kept, just not offered again
});

test('a chat too large for the server is sent again without its inline pictures', async () => {
  const server = fakeServer({ maxChatBytes: 3000 });
  const { store, sync, notices } = device({ server });
  await sync.start();
  const big = 'data:image/png;base64,' + 'A'.repeat(4000);
  await store.send('그림', [{ name: 'big.png', kind: 'image', dataUrl: big }]);
  await sync.flush();
  assert.deepEqual(server.chat(store.current.id).messages[0].attachments, [{ name: 'big.png', kind: 'image', removed: true }]);
  assert.equal(store.current.messages[0].attachments[0].dataUrl, big);  // this browser keeps showing it
  assert.deepEqual(notices, []);
  assert.equal(sync.status, 'synced');

  server.maxChatBytes = 100;  // too large even without pictures: reported once, not retried until it changes
  store.togglePin(store.current.id);
  await sync.flush();
  await sync.flush();
  assert.equal(notices.length, 1);
  assert.match(notices[0], /너무 커서/);
  assert.equal(server.count('PUT /api/chats'), 4);
});

test('uploaded pictures are kept by URL and sent to the model as that URL', async () => {
  const bodies = [];
  const store = new ChatStore(memory(), async (_, init) => {
    bodies.push(JSON.parse(init.body));
    return sse([done('보여요')]);
  });
  await store.send('', [{ name: 'a.jpg', kind: 'image', fileId: 'abcdefgh12', url: '/api/files/abcdefgh12', preview: 'blob:x' }]);
  assert.deepEqual(store.current.messages[0].attachments,
    [{ name: 'a.jpg', kind: 'image', fileId: 'abcdefgh12', url: '/api/files/abcdefgh12' }]);
  assert.deepEqual(bodies[0].messages[0].content, [{ type: 'image_url', image_url: { url: '/api/files/abcdefgh12' } },
    { type: 'text', text: '이 이미지를 설명해 줘.' }]);
  assert.equal(bodies[0].chat_id, store.current.id);

  const saved = { chats: [{ id: 'x', title: 't', messages: [{ id: 'u', role: 'user', content: 'q', attachments: [
    { name: 'ok', kind: 'image', url: '/api/files/abcdefgh12' },
    { name: 'outside', kind: 'image', url: 'https://example.com/a.png' },
    { name: 'escape', kind: 'image', url: '/api/files/../../etc/passwd' },
    { name: 'short', kind: 'image', url: '/api/files/abc' }] }] }] };
  const files = new ChatStore(memory({ 'pubmed-chat.v1': JSON.stringify(saved) })).current.messages[0].attachments;
  assert.deepEqual(files.map(file => [file.name, file.url ?? null, file.removed === true]),
    [['ok', '/api/files/abcdefgh12', false], ['outside', null, true], ['escape', null, true], ['short', null, true]]);
});

test('uploading a picture returns a reference to the server copy', async () => {
  const server = fakeServer();
  const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'image/jpeg' });
  assert.deepEqual(await uploadImage(server.fetch, blob), { fileId: 'file000001', url: '/api/files/file000001' });
  assert.equal(server.files.get('file000001').mime, 'image/jpeg');
  server.rejectUploads = true;
  await assert.rejects(uploadImage(server.fetch, blob), error => error.status === 422 && error.code === 'invalid_image');
});

test('a signed-out session reports unauthorized and stops syncing', async () => {
  const { server, store, sync, statuses } = device();
  await sync.start();
  await store.send('질문');
  server.signedOut = true;
  await sync.flush();
  assert.equal(sync.status, 'unauthorized');
  assert.equal(statuses.at(-1), 'unauthorized');
  const requests = server.log.length;
  store.togglePin(store.current.id);
  await sync.flush();
  await sync.syncNow();
  assert.equal(server.log.length, requests);
});

test('an unreachable server reports offline and is retried after 5, 10, 20 and then 30 seconds', async () => {
  const time = clock();
  const { server, store, sync } = device({ sync: timed(time) });
  server.offline = true;
  await sync.start();
  assert.equal(sync.status, 'offline');
  for (const delay of [5000, 10000, 20000, 30000, 30000]) {
    assert.deepEqual(time.delays(), [delay]);
    await time.advance(delay);
  }
  store.updateSettings({ temperature: 0.5 });  // saves while offline wait for the next retry
  assert.deepEqual(time.delays(), [30000]);
  server.offline = false;
  await time.advance(30000);
  assert.equal(sync.status, 'synced');
  assert.equal(server.settings.temperature, 0.5);
  assert.deepEqual(time.delays(), [5000]);
});

test('pulls run every 5 s while the page is visible, every 30 s while hidden, and at once when shown', async () => {
  const time = clock();
  const page = new EventTarget();
  page.visibilityState = 'visible';
  const { server, sync } = device({ sync: { ...timed(time), document: page } });
  await sync.start();
  assert.deepEqual(time.delays(), [5000]);
  await time.advance(5000);
  assert.equal(server.count('GET /api/chats'), 2);
  page.visibilityState = 'hidden';
  page.dispatchEvent(new Event('visibilitychange'));
  await tick();
  assert.deepEqual(time.delays(), [30000]);
  assert.equal(server.count('GET /api/chats'), 2);
  page.visibilityState = 'visible';
  page.dispatchEvent(new Event('visibilitychange'));
  await tick();
  assert.equal(server.count('GET /api/chats'), 3);
  assert.deepEqual(time.delays(), [5000]);
  sync.stop();
  assert.deepEqual(time.delays(), []);
});

test('if the server lost its history, this browser catches up from the start and uploads its chats', async () => {
  const { server, store, sync } = device();
  server.seed(chatOf('other', 1000));
  await sync.start();
  await store.send('질문');
  await sync.flush();
  server.rows.clear();
  server.seq = 0;
  await sync.syncNow();
  assert.ok(server.chat(store.current.id));
  assert.ok(server.chat('other'));
});

test('a full browser storage trims nothing and makes the next visit catch up from the start', async () => {
  const server = fakeServer();
  const data = new Map();
  let full = false;
  const storage = { getItem: key => data.get(key) ?? null, removeItem: key => data.delete(key),
    setItem: (key, value) => { if (full && key === KEY) throw new Error('QuotaExceededError'); data.set(key, value); } };
  const { store, sync } = device({ server, storage });
  await sync.start();
  await store.send('그림', [{ name: 'a.png', kind: 'image', dataUrl: PNG }]);
  await sync.flush();
  await sync.syncNow();  // the cursor moves past the upload
  assert.ok(JSON.parse(storage.getItem(syncKey(KEY))).seq > 0);
  full = true;
  await store.send('하나 더');
  await sync.flush();
  assert.equal(store.cacheFailed, true);
  assert.equal(store.storageWarning, '');
  assert.equal(store.current.messages[0].attachments[0].dataUrl, PNG);
  assert.equal(server.chat(store.current.id).messages.length, 4);
  assert.equal(JSON.parse(storage.getItem(syncKey(KEY))).seq, 0);
  full = false;
  const reloaded = device({ server, storage });
  await reloaded.sync.start();
  assert.equal(server.log.filter(line => line.startsWith('GET /api/chats')).at(-1), 'GET /api/chats?since=0');
  assert.equal(reloaded.store.chats.find(chat => chat.id === store.current.id).messages.length, 4);
});

test('signing out forgets the user\'s cached chats and sync state', () => {
  const storage = memory({ [KEY]: '{}', [syncKey(KEY)]: '{}', 'pubmed-chat.theme': 'dark' });
  forgetCache(storage, KEY);
  assert.deepEqual([...storage.data.keys()], ['pubmed-chat.theme']);
});

test('keep-alive comment lines in the event stream are ignored', async () => {
  const body = new Response(': ping\n\ndata: {"type":"delta","content":"가"}\n\n: ping\n\n').body;
  const events = [];
  for await (const event of readEvents(body)) events.push(event);
  assert.deepEqual(events, [{ type: 'delta', content: '가' }]);
});

test('pages carry no inline script, style or event handler, so the CSP can forbid them', async () => {
  for (const name of ['index.html', 'login.html']) {
    const html = await readFile(new URL('../app/static/' + name, import.meta.url), 'utf8');
    assert.ok(!/<script(?![^>]*\ssrc=)[^>]*>/i.test(html), name + ': inline script');
    assert.ok(!/<style[\s>]/i.test(html), name + ': style element');
    assert.ok(!/\sstyle\s*=/i.test(html), name + ': style attribute');
    assert.ok(!/\son[a-z]+\s*=/i.test(html), name + ': inline event handler');
    assert.ok(html.includes('<script src="/static/theme.js"></script>'), name + ': theme script');
  }
});
