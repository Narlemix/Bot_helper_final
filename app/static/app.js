const messages = document.getElementById('messages');
const form = document.getElementById('chat-form');
const input = document.getElementById('message');
const suggestions = document.getElementById('suggestions');
const typeahead = document.getElementById('typeahead');
const resetBtn = document.getElementById('reset');

let sessionId = localStorage.getItem('dismissal_session_id');

// ---- Live typing suggestions (typeahead) ----
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
      // mousedown fires before input blur, so the click isn't lost
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
    // Ignore stale responses if the input has since changed/cleared.
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
  // small delay so a click on a suggestion (mousedown) still registers
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
  if (sessionId) {
    await fetch('/api/reset', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId})
    });
  }
  messages.innerHTML = '';
  setSuggestions([]);
  hideTypeahead();
  addMessage('Здравствуйте! Я помогу с вопросами по увольнению. Напишите, что вам нужно сделать.');
  input.focus();
};

addMessage('Здравствуйте! Я помогу с вопросами по увольнению. Напишите, что вам нужно сделать.');
setSuggestions(['Как оформить увольнение?', 'Какие документы нужны?', 'Хочу оформить обращение']);
input.focus();
