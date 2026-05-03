import { useState, useRef, useEffect, useCallback } from "react";

const COMPANIES = ["None", "HackerRank", "Claude", "Visa"];

const STATUS_COLORS = {
  replied: { bg: "bg-emerald-500/15", text: "text-emerald-400", border: "border-emerald-500/30", dot: "bg-emerald-400" },
  escalated: { bg: "bg-amber-500/15", text: "text-amber-400", border: "border-amber-500/30", dot: "bg-amber-400" },
};

const TYPE_COLORS = {
  product_issue: "text-sky-400 bg-sky-500/10 border-sky-500/20",
  bug: "text-red-400 bg-red-500/10 border-red-500/20",
  feature_request: "text-violet-400 bg-violet-500/10 border-violet-500/20",
  invalid: "text-zinc-400 bg-zinc-500/10 border-zinc-500/20",
};

const COMPANY_COLORS = {
  HackerRank: "text-green-400",
  Claude: "text-orange-400",
  Visa: "text-blue-400",
  None: "text-zinc-400",
  Unknown: "text-zinc-400",
};

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map(i => (
        <div key={i} className="w-2 h-2 rounded-full bg-zinc-500 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }} />
      ))}
    </div>
  );
}

function StatusBadge({ status }) {
  const c = STATUS_COLORS[status] || STATUS_COLORS.replied;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${c.bg} ${c.text} ${c.border}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {status}
    </span>
  );
}

function ToolCallBadge({ name, query, company }) {
  const icons = {
    search_company_docs: "🎯",
    search_all_docs: "🔍",
    search_refined: "✨",
    escalate: "⚠️",
    reply_out_of_scope: "🚫",
  };
  return (
    <div className="flex items-start gap-2 py-1.5 px-3 my-1 rounded-lg bg-zinc-800/60 border border-zinc-700/50 text-xs">
      <span>{icons[name] || "🔧"}</span>
      <div className="flex-1 min-w-0">
        <span className="text-zinc-400 font-mono">{name}</span>
        {query && <span className="text-zinc-500"> → </span>}
        {query && <span className="text-zinc-300 italic">"{query}"</span>}
        {company && company !== "null" && company !== "None" && (
          <span className={`ml-2 font-semibold ${COMPANY_COLORS[company] || "text-zinc-400"}`}>
            [{company}]
          </span>
        )}
      </div>
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[75%]">
          <div className="bg-zinc-700 text-zinc-100 rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed">
            {msg.content}
          </div>
          {msg.company && msg.company !== "None" && (
            <div className="flex justify-end mt-1">
              <span className={`text-xs ${COMPANY_COLORS[msg.company]}`}>via {msg.company}</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 mb-6">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0 mt-0.5">
        T
      </div>
      <div className="flex-1 min-w-0">
        {/* Tool calls */}
        {msg.toolCalls && msg.toolCalls.length > 0 && (
          <div className="mb-3">
            <p className="text-xs text-zinc-500 mb-1 font-medium">Searching knowledge base...</p>
            {msg.toolCalls.map((tc, i) => (
              <ToolCallBadge key={i} {...tc} />
            ))}
          </div>
        )}

        {/* Loading state */}
        {msg.streaming && !msg.content && (
          <div className="flex items-center gap-2 text-zinc-500 text-sm">
            <TypingIndicator />
            <span className="text-xs">{msg.statusText || "Thinking..."}</span>
          </div>
        )}

        {/* Status text while streaming */}
        {msg.streaming && msg.content === "" && msg.statusText && !msg.toolCalls?.length && (
          <p className="text-xs text-zinc-500 italic">{msg.statusText}</p>
        )}

        {/* Response content */}
        {msg.content && (
          <div className="bg-zinc-800/50 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-zinc-100 leading-relaxed whitespace-pre-wrap border border-zinc-700/40">
            {msg.content}
            {msg.streaming && (
              <span className="inline-block w-0.5 h-4 bg-orange-400 ml-0.5 animate-pulse align-middle" />
            )}
          </div>
        )}

        {/* Result metadata */}
        {msg.result && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <StatusBadge status={msg.result.status} />
            {msg.result.request_type && (
              <span className={`text-xs px-2.5 py-1 rounded-full border font-medium ${TYPE_COLORS[msg.result.request_type] || TYPE_COLORS.product_issue}`}>
                {msg.result.request_type.replace("_", " ")}
              </span>
            )}
            {msg.result.product_area && (
              <span className="text-xs px-2.5 py-1 rounded-full border border-zinc-700/50 bg-zinc-800/50 text-zinc-400">
                {msg.result.product_area}
              </span>
            )}
          </div>
        )}

        {msg.result?.justification && (
          <p className="mt-2 text-xs text-zinc-500 italic leading-relaxed">
            {msg.result.justification}
          </p>
        )}
      </div>
    </div>
  );
}

function Sidebar({ selectedCompany, onSelect }) {
  const examples = [
    { text: "How do I add extra time for a candidate?", company: "HackerRank" },
    { text: "I forgot to use incognito chat and shared private info", company: "Claude" },
    { text: "My Visa card was declined abroad", company: "Visa" },
    { text: "How do I delete my account?", company: "None" },
    { text: "I found a security vulnerability", company: "Claude" },
  ];

  return (
    <div className="w-64 flex-shrink-0 bg-zinc-900 border-r border-zinc-800 flex flex-col">
      <div className="p-5 border-b border-zinc-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-bold text-sm">T</div>
          <div>
            <h1 className="text-white font-semibold text-sm tracking-tight">TriageAI</h1>
            <p className="text-zinc-500 text-xs">Support Intelligence</p>
          </div>
        </div>
      </div>

      <div className="p-4 border-b border-zinc-800">
        <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider block mb-2">Product</label>
        <div className="flex flex-col gap-1">
          {COMPANIES.map(c => (
            <button key={c} onClick={() => onSelect(c, null)}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-left transition-colors ${selectedCompany === c ? "bg-zinc-700 text-white" : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                }`}>
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${c === "HackerRank" ? "bg-green-400" : c === "Claude" ? "bg-orange-400" : c === "Visa" ? "bg-blue-400" : "bg-zinc-600"
                }`} />
              {c === "None" ? "Auto-detect" : c}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 flex-1 overflow-y-auto">
        <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider block mb-2">Examples</label>
        <div className="flex flex-col gap-1.5">
          {examples.map((ex, i) => (
            <button key={i} onClick={() => onSelect(ex.company, ex.text)}
              className="text-left px-3 py-2.5 rounded-lg text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors leading-relaxed border border-transparent hover:border-zinc-700/50">
              {ex.text}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 border-t border-zinc-800">
        <p className="text-xs text-zinc-600 text-center">Powered by Claude + Qdrant</p>
      </div>
    </div>
  );
}

let msgCounter = 0;
const nextId = () => ++msgCounter;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [company, setCompany] = useState("None");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSelect = useCallback((newCompany, exampleText) => {
    setCompany(newCompany);
    if (exampleText) {
      setInput(exampleText);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, []);

  const updateMsg = useCallback((id, updater) => {
    setMessages(prev => prev.map(m => m.id === id ? updater(m) : m));
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const text = input.trim();
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "46px";
    }
    setIsLoading(true);

    const userId = nextId();
    const botId = nextId();

    setMessages(prev => [
      ...prev,
      { id: userId, role: "user", content: text, company: company !== "None" ? company : null },
      { id: botId, role: "assistant", content: "", streaming: true, statusText: "Connecting...", toolCalls: [], result: null },
    ]);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, company, subject: "" }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        let eventType = "";
        let eventData = "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            eventData = line.slice(6).trim();
          } else if (line === "" && eventType && eventData) {
            try {
              const data = JSON.parse(eventData);

              if (eventType === "status") {
                updateMsg(botId, m => ({ ...m, statusText: data.message }));
              } else if (eventType === "iteration") {
                updateMsg(botId, m => ({ ...m, statusText: `Iteration ${data.current}/${data.max}...` }));
              } else if (eventType === "tool") {
                updateMsg(botId, m => ({ ...m, toolCalls: [...(m.toolCalls || []), data] }));
              } else if (eventType === "token") {
                accText += data.text;
                const captured = accText;
                updateMsg(botId, m => ({ ...m, content: captured, streaming: true }));
              } else if (eventType === "result") {
                updateMsg(botId, m => ({ ...m, result: data, streaming: false }));
              } else if (eventType === "done") {
                updateMsg(botId, m => ({ ...m, streaming: false }));
              } else if (eventType === "error") {
                updateMsg(botId, m => ({ ...m, content: `Error: ${data.message}`, streaming: false }));
              }
            } catch (e) {
              console.warn("SSE parse error:", e);
            }
            eventType = "";
            eventData = "";
          }
        }
      }
    } catch (err) {
      updateMsg(botId, m => ({ ...m, content: `Connection error: ${err.message}`, streaming: false }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      <Sidebar selectedCompany={company} onSelect={handleSelect} />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between flex-shrink-0">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">Support Chat</h2>
            <p className="text-xs text-zinc-500 mt-0.5">
              {company === "None" ? "Auto-detecting product" : `${company} support`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-zinc-500">Live</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {isEmpty && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white text-2xl font-bold mb-4">T</div>
              <h3 className="text-xl font-semibold text-zinc-200 mb-2">Welcome to TriageAI</h3>
              <p className="text-zinc-500 text-sm max-w-sm leading-relaxed">
                Intelligent support triage for HackerRank, Claude, and Visa. Describe your issue and I'll find the right answer or route you to a human.
              </p>
              <div className="grid grid-cols-3 gap-3 mt-8 w-full max-w-lg">
                {[
                  { icon: "🎯", label: "HackerRank", desc: "Tests, interviews, assessments" },
                  { icon: "🤖", label: "Claude", desc: "AI assistant, API, plans" },
                  { icon: "💳", label: "Visa", desc: "Cards, payments, travel" },
                ].map(item => (
                  <div key={item.label} className="p-3 rounded-xl bg-zinc-800/50 border border-zinc-700/50 text-center">
                    <div className="text-2xl mb-1">{item.icon}</div>
                    <div className="text-xs font-semibold text-zinc-300">{item.label}</div>
                    <div className="text-xs text-zinc-500 mt-0.5">{item.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {messages.map(msg => <Message key={msg.id} msg={msg} />)}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-zinc-800 px-6 py-4 flex-shrink-0">
          <div className="flex gap-3 items-end">
            <div className="flex-1 relative">
              <textarea
                ref={(el) => { inputRef.current = el; textareaRef.current = el; }}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe your support issue..."
                rows={1}
                disabled={isLoading}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 resize-none focus:outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500/30 transition-colors leading-relaxed"
                style={{ minHeight: "46px", maxHeight: "120px" }}
                onInput={e => {
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
                }}
              />
            </div>
            <button onClick={sendMessage} disabled={!input.trim() || isLoading}
              className="w-11 h-11 rounded-xl bg-orange-500 hover:bg-orange-400 disabled:bg-zinc-700 disabled:text-zinc-500 text-white flex items-center justify-center transition-colors flex-shrink-0">
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              )}
            </button>
          </div>
          <p className="text-xs text-zinc-600 mt-2 text-center">Press Enter to send · Shift+Enter for new line</p>
        </div>
      </div>
    </div>
  );
}