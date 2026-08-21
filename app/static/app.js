const messages = document.getElementById('messages');
const form = document.getElementById('chat-form');
const input = document.getElementById('message');
const suggestions = document.getElementById('suggestions');
const typeahead = document.getElementById('typeahead');
const resetBtn = document.getElementById('reset');

let sessionId = localStorage.getItem('dismissal_session_id');

// Переключение вкладок «Чат» / «Логи маршрутизации»
const tabButtons = document.querySelectorAll('.tab-btn');
const chatView = document.getElementById('chat-view');
const logsView = document.getElementById('logs-view');

tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    tabButtons.forEach(b => b.classList.toggle('active', b === btn));
    const tab = btn.dataset.tab;
    chatView.hidden = tab !== 'chat';
    logsView.hidden = tab !== 'logs';
    resetBtn.hidden = tab !== 'chat';
    if (tab === 'logs') {
      loadLogs();
      scheduleLogsAutorefresh();
    } else {
      clearInterval(logsTimer);
    }
  });
});

// Живой просмотр логов маршрутизации
const logsFilter = document.getElementById('logs-filter');
const logsAutorefresh = document.getElementById('logs-autorefresh');
const logsRefreshBtn = document.getElementById('logs-refresh');
const logsStatus = document.getElementById('logs-status');
const logsList = document.getElementById('logs-list');

let logsEntries = [];
let logsTimer = null;

function logsBadgeFor(entry) {
  const m = entry.message;
  if (m.includes('AMBIGUOUS') || m.includes('NO_MATCH')) return ['badge-ambiguous', m.includes('NO_MATCH') ? 'NO MATCH' : 'НЕОДНОЗНАЧНО'];
  if (m.startsWith('send:') && m.includes('ok=True')) return ['badge-send-ok', 'ОТПРАВЛЕНО'];
  if (m.startsWith('send:') && m.includes('ok=False')) return ['badge-send-fail', 'ОШИБКА'];
  if (m.startsWith('route:')) return ['badge-route', 'МАРШРУТ'];
  return ['badge-other', entry.logger.replace('hrbot.', '')];
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderLogs() {
  const q = logsFilter.value.trim().toLowerCase();
  const filtered = q ? logsEntries.filter(e => e.message.toLowerCase().includes(q) || e.logger.toLowerCase().includes(q)) : logsEntries;

  if (!filtered.length) {
    logsList.innerHTML = `<div class="empty">${logsEntries.length ? 'Ничего не найдено по фильтру.' : 'Логов пока нет — как только кто-то напишет боту, записи появятся здесь.'}</div>`;
    return;
  }

  logsList.innerHTML = filtered.map(e => {
    const [badgeClass, badgeText] = logsBadgeFor(e);
    const time = e.ts.replace('T', ' ').replace('Z', '').split('.')[0];
    return `<div class="log-row">
      <span class="log-ts">${escapeHtml(time)}</span>
      <span class="log-badge ${badgeClass}">${escapeHtml(badgeText)}</span>
      <span class="log-logger">${escapeHtml(e.logger.replace('hrbot.', ''))}</span>
      <span class="log-msg">${escapeHtml(e.message)}</span>
    </div>`;
  }).join('');
}

async function loadLogs() {
  try {
    const res = await fetch('/api/admin/logs');
    if (!res.ok) {
      let detail = `Ошибка ${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch (e) {}
      logsStatus.textContent = detail;
      logsList.innerHTML = `<div class="empty">${escapeHtml(detail)}</div>`;
      return;
    }
    const data = await res.json();
    logsEntries = data.logs || [];
    logsStatus.textContent = `Записей: ${logsEntries.length} · обновлено ${new Date().toLocaleTimeString('ru-RU')}`;
    renderLogs();
  } catch (err) {
    logsStatus.textContent = 'Ошибка соединения';
  }
}

function scheduleLogsAutorefresh() {
  clearInterval(logsTimer);
  if (logsAutorefresh.checked) {
    logsTimer = setInterval(loadLogs, 4000);
  }
}

logsFilter.addEventListener('input', renderLogs);
logsRefreshBtn.addEventListener('click', loadLogs);
logsAutorefresh.addEventListener('change', scheduleLogsAutorefresh);

// Подсказки при вводе (typeahead)
let typeaheadItems = [];
let typeaheadActive = -1;
let typeaheadTimer = null;
let typeaheadAbort = null;

function hideTypeahead() {
  typeahead.hidden = true;
  typeahead.innerHTML = '';
  typeaheadItems = [];
  typeaheadActive = -1;
}

function renderTypeahead(items) {
  typeaheadItems = items;
  typeaheadActive = -1;
  typeahead.innerHTML = '';
  if (!items.length) {
    hideTypeahead();
    return;
  }
  items.forEach((text, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'typeahead-item';
    btn.textContent = text;
    btn.onmousedown = (e) => {
      e.preventDefault();
      chooseTypeahead(idx);
    };
    typeahead.appendChild(btn);
  });
  typeahead.hidden = false;
}

function chooseTypeahead(idx) {
  const text = typeaheadItems[idx];
  if (!text) return;
  input.value = text;
  hideTypeahead();
  input.focus();
}

function highlightTypeahead(idx) {
  const nodes = typeahead.querySelectorAll('.typeahead-item');
  nodes.forEach((node, i) => node.classList.toggle('active', i === idx));
  typeaheadActive = idx;
}

async function fetchTypeahead(query) {
  if (typeaheadAbort) typeaheadAbort.abort();
  typeaheadAbort = new AbortController();
  try {
    const response = await fetch(`/api/suggest?q=${encodeURIComponent(query)}`, {
      signal: typeaheadAbort.signal
    });
    const data = await response.json();
    if (normalizeForCompare(input.value) === normalizeForCompare(query)) {
      renderTypeahead(data.suggestions || []);
    }
  } catch (err) {
    if (err.name !== 'AbortError') hideTypeahead();
  }
}

function normalizeForCompare(text) {
  return text.trim().toLowerCase();
}

input.addEventListener('input', () => {
  const query = input.value.trim();
  clearTimeout(typeaheadTimer);
  if (query.length < 2) {
    hideTypeahead();
    return;
  }
  typeaheadTimer = setTimeout(() => fetchTypeahead(query), 180);
});

input.addEventListener('keydown', (e) => {
  if (typeahead.hidden || !typeaheadItems.length) return;
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    highlightTypeahead((typeaheadActive + 1) % typeaheadItems.length);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    highlightTypeahead((typeaheadActive - 1 + typeaheadItems.length) % typeaheadItems.length);
  } else if (e.key === 'Enter' && typeaheadActive >= 0) {
    e.preventDefault();
    chooseTypeahead(typeaheadActive);
  } else if (e.key === 'Escape') {
    hideTypeahead();
  }
});

input.addEventListener('blur', () => {
  setTimeout(hideTypeahead, 100);
});

function addMessage(text, who='bot') {
  const el = document.createElement('div');
  el.className = `message ${who}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

function setSuggestions(items=[]) {
  suggestions.innerHTML = '';
  items.forEach(item => {
    const button = document.createElement('button');
    button.className = 'suggestion';
    button.type = 'button';
    button.textContent = item;
    button.onclick = () => send(item);
    suggestions.appendChild(button);
  });
}

async function send(text) {
  const value = (text || input.value).trim();
  if (!value) return;
  input.value = '';
  hideTypeahead();
  addMessage(value, 'user');
  setSuggestions([]);
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sessionId, message: value})
  });
  const data = await response.json();
  sessionId = data.session_id;
  localStorage.setItem('dismissal_session_id', sessionId);
  addMessage(data.text, 'bot');
  setSuggestions(data.options || []);
}

form.addEventListener('submit', (e) => { e.preventDefault(); send(); });

resetBtn.onclick = async () => {
  const response = await fetch('/api/reset', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sessionId})
  });
  const data = await response.json();
  sessionId = data.session_id;
  localStorage.setItem('dismissal_session_id', sessionId);
  messages.innerHTML = '';
  setSuggestions([]);
  hideTypeahead();
  addMessage('Здравствуйте! Я помогу с кадровыми и административными вопросами — увольнение, справки, подотчётные средства, претензии, договоры и другое. Напишите, что вам нужно сделать.');
  input.focus();
};

addMessage('Здравствуйте! Я помогу с кадровыми и административными вопросами — увольнение, справки, подотчётные средства, претензии, договоры и другое. Напишите, что вам нужно сделать.');
setSuggestions(['Как оформить увольнение?', 'Уволить вахтовика', 'Заказать справку 2-НДФЛ', 'Хочу оформить обращение']);
input.focus();
