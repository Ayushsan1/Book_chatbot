// =========================================================
// CONFIG
// For local testing: use localhost (server must be running)
// For production:    swap to your HF Space URL below
// =========================================================
const API_URL = "https://ayushmishra01-chatbot-intern.hf.space";
// const API_URL = "http://localhost:8000"; // ← uncomment for local testing

// Generate a stable user ID per browser session
const USER_ID = sessionStorage.getItem("thinkhub_uid") || (() => {
  const id = "user_" + Math.random().toString(36).slice(2, 10);
  sessionStorage.setItem("thinkhub_uid", id);
  return id;
})();

// =========================================================
// DOM refs
// =========================================================
const messagesEl = document.getElementById("messages");
const inputEl    = document.getElementById("user-input");
const sendBtn    = document.getElementById("send-btn");
const clearBtn   = document.getElementById("clear-btn");

// Pillar quick-prompt buttons
const PILLAR_PROMPTS = {
  "pillar-infra":     "Tell me about Thinkhub India's Infrastructure solutions.",
  "pillar-cloud":     "What Cloud solutions does Thinkhub India offer? (AWS, Azure, GCP)",
  "pillar-analytics": "Explain Thinkhub India's Business Analytics offerings using SAS and Power BI.",
  "pillar-digital":   "What Digital Solutions does Thinkhub India provide?"
};

document.querySelectorAll(".pillar-item").forEach(el => {
  el.addEventListener("click", () => {
    inputEl.value = PILLAR_PROMPTS[el.id] || "";
    inputEl.focus();
    autoResize();
  });
});

clearBtn.addEventListener("click", () => {
  // Remove all messages except welcome
  const msgs = messagesEl.querySelectorAll(".message:not(#welcome-msg)");
  msgs.forEach(m => m.remove());
  sessionStorage.removeItem("thinkhub_uid");
  location.reload(); // fresh session
});

// =========================================================
// Auto-resize textarea
// =========================================================
function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
}
inputEl.addEventListener("input", autoResize);

// Send on Enter (Shift+Enter = newline)
inputEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// =========================================================
// Core send function
// =========================================================
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  // Render user bubble
  appendMessage("user", text);
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendBtn.disabled = true;

  // Show typing indicator
  const typingEl = showTyping();

  try {
    const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID, question: text })
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    typingEl.remove();
    appendMessage("bot", data.response);
  } catch (err) {
    typingEl.remove();
    appendMessage("bot", "⚠️ Sorry, I couldn't reach the server. Please try again shortly.");
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// =========================================================
// Helpers
// =========================================================
function appendMessage(role, text) {
  const isUser = role === "user";
  const msgEl  = document.createElement("div");
  msgEl.classList.add("message", isUser ? "user-message" : "bot-message");

  const avatarEl = document.createElement("div");
  avatarEl.classList.add("avatar", isUser ? "user-avatar" : "bot-avatar");
  avatarEl.textContent = isUser ? "You" : "TH";

  const bubbleEl = document.createElement("div");
  bubbleEl.classList.add("bubble", isUser ? "user-bubble" : "bot-bubble");
  bubbleEl.innerHTML = formatText(text);

  msgEl.appendChild(avatarEl);
  msgEl.appendChild(bubbleEl);
  messagesEl.appendChild(msgEl);
  scrollToBottom();
}

function showTyping() {
  const msgEl = document.createElement("div");
  msgEl.classList.add("message", "bot-message");

  const avatarEl = document.createElement("div");
  avatarEl.classList.add("avatar", "bot-avatar");
  avatarEl.textContent = "TH";

  const bubbleEl = document.createElement("div");
  bubbleEl.classList.add("bubble", "bot-bubble", "typing-bubble");
  bubbleEl.innerHTML = `<div class="dot"></div><div class="dot"></div><div class="dot"></div>`;

  msgEl.appendChild(avatarEl);
  msgEl.appendChild(bubbleEl);
  messagesEl.appendChild(msgEl);
  scrollToBottom();
  return msgEl;
}

// Simple text formatter: newlines → paragraphs, **bold** → <strong>
function formatText(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .split(/\n+/)
    .filter(p => p.trim())
    .map(p => `<p>${p}</p>`)
    .join("");
}

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}
