const form = document.getElementById('login-form');
const error = document.getElementById('login-error');
const submit = document.getElementById('login-submit');
// The page ships the button disabled and the form as POST, so a submit before this script runs (or without it)
// never puts the password in a URL, where the browser history and the access log would keep it.
submit.disabled = false;

function showError(text) {
  error.textContent = text;
  error.hidden = !text;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const name = form.elements.name.value.trim();
  const password = form.elements.password.value;
  if (!name || !password) { showError('이름과 비밀번호를 입력해 주세요.'); return; }
  showError('');
  submit.disabled = true;
  submit.textContent = '로그인하는 중…';
  try {
    const response = await fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, password }) });
    if (response.ok) { location.replace('/'); return; }
    const data = await response.json().catch(() => ({}));
    if (data.error?.code === 'auth_disabled') { location.replace('/'); return; }  // this server needs no sign-in
    if (response.status === 429) showError('시도가 너무 많아요. 잠시 후 다시 시도해 주세요.');
    else if (response.status === 401) {
      showError('이름 또는 비밀번호가 맞지 않아요.');
      form.elements.password.select();
    } else showError('로그인하지 못했어요. 잠시 후 다시 시도해 주세요.');
  } catch { showError('서버에 연결할 수 없어요.'); }
  submit.disabled = false;
  submit.textContent = '로그인';
});
