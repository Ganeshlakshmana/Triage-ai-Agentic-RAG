/**
 * App.tsx — TriageAI Chat Interface (TypeScript)
 *
 * Full-stack streaming chat UI for the multi-provider support triage agent.
 * Features:
 *   - Provider selector (Claude / OpenAI / Gemini / Auto)
 *   - Real-time SSE token streaming with blinking cursor
 *   - Live tool-call badges shown as agent reasons
 *   - Layer 3 confidence score badge
 *   - Status badge: replied / escalated
 *   - Company & subject metadata inputs
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import type {
  Message, ToolCall, Provider, ConfidenceEvent,
  StatusEvent, IterationEvent, ProviderEvent, ToolEvent,
  TokenEvent, ResultEvent, ErrorEvent,
} from "./types";
import { PROVIDER_LABELS, PROVIDER_COLORS } from "./types";
import "./App.css";

// ── Constants ─────────────────────────────────────────────────────────────────

const API_URL = "http://localhost:8000/api/chat";
const PROVIDERS: Provider[] = ["auto", "claude", "openai", "gemini"];
const COMPANIES = ["None", "HackerRank", "Claude", "Visa"];

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface ProviderPillProps {
  provider: Provider;
  selected: boolean;
  onClick: () => void;
}

const ProviderPill: React.FC<ProviderPillProps> = ({ provider, selected, onClick }) => {
  const color = PROVIDER_COLORS[provider];
  return (
    <button
      className={`provider-pill ${selected ? "selected" : ""}`}
      style={selected ? { borderColor: color, color: color, background: color + "18" } : {}}
      onClick={onClick}
      title={`Use ${PROVIDER_LABELS[provider]}`}
    >
      <span className="provider-dot" style={{ background: selected ? color : "#6b7280" }} />
      {PROVIDER_LABELS[provider]}
    </button>
  );
};

interface ToolBadgeProps {
  tool: ToolCall;
  index: number;
}

const ToolBadge: React.FC<ToolBadgeProps> = ({ tool, index }) => (
  <div className="tool-badge">
    <span className="tool-icon">🔍</span>
    <span className="tool-name">{tool.name.replace(/_/g, " ")}</span>
    <span className="tool-query">"{tool.query}"</span>
    {tool.company && tool.company !== "null" && (
      <span className="tool-company">{tool.company}</span>
    )}
  </div>
);

interface ConfidenceBadgeProps {
  confidence: ConfidenceEvent;
}

const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ confidence }) => {
  const levelColor =
    confidence.level === "high" ? "#16a34a" :
    confidence.level === "medium" ? "#d97706" : "#dc2626";
  const providerColor = PROVIDER_COLORS[confidence.provider_routed_to] ?? "#7c3aed";

  return (
    <div className="confidence-badge">
      <span className="conf-label">Layer 3 Confidence</span>
      <span className="conf-score" style={{ color: levelColor }}>
        {(confidence.score * 100).toFixed(0)}% {confidence.level}
      </span>
      <span className="conf-arrow">→</span>
      <span className="conf-provider" style={{ color: providerColor }}>
        {PROVIDER_LABELS[confidence.provider_routed_to] ?? confidence.provider_routed_to}
      </span>
    </div>
  );
};

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isAgent = message.role === "agent";

  return (
    <div className={`message-row ${message.role}`}>
      <div className={`message-bubble ${message.role}`}>
        {/* User message */}
        {!isAgent && <p className="message-text">{message.text}</p>}

        {/* Agent message */}
        {isAgent && (
          <>
            {/* Live tool calls */}
            {message.tool_calls.length > 0 && (
              <div className="tool-calls-container">
                <span className="tool-calls-label">Tool calls</span>
                {message.tool_calls.map((tool, i) => (
                  <ToolBadge key={i} tool={tool} index={i} />
                ))}
              </div>
            )}

            {/* Confidence + provider routing (Layer 3 & 4) */}
            {message.confidence && (
              <ConfidenceBadge confidence={message.confidence} />
            )}

            {/* Response text */}
            <p className="message-text">
              {message.text}
              {message.isStreaming && <span className="cursor">▋</span>}
            </p>

            {/* Error */}
            {message.error && (
              <p className="error-text">⚠ {message.error}</p>
            )}

            {/* Metadata badges */}
            {!message.isStreaming && message.status && (
              <div className="meta-badges">
                <span className={`status-badge ${message.status}`}>
                  {message.status === "replied" ? "✓ Replied" : "⚡ Escalated"}
                </span>
                {message.product_area && (
                  <span className="area-badge">{message.product_area}</span>
                )}
                {message.request_type && (
                  <span className="type-badge">{message.request_type.replace(/_/g, " ")}</span>
                )}
                {message.provider && message.provider !== "auto" && (
                  <span
                    className="provider-badge"
                    style={{ borderColor: PROVIDER_COLORS[message.provider], color: PROVIDER_COLORS[message.provider] }}
                  >
                    {PROVIDER_LABELS[message.provider]}
                  </span>
                )}
              </div>
            )}

            {/* Justification */}
            {!message.isStreaming && message.justification && (
              <p className="justification">{message.justification}</p>
            )}
          </>
        )}
      </div>
    </div>
  );
};

// ── Main App ──────────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [company, setCompany] = useState<string>("None");
  const [subject, setSubject] = useState<string>("");
  const [provider, setProvider] = useState<Provider>("auto");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [statusText, setStatusText] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const updateLastMessage = useCallback((updater: (msg: Message) => Message) => {
    setMessages(prev => {
      const next = [...prev];
      if (next.length > 0 && next[next.length - 1].role === "agent") {
        next[next.length - 1] = updater(next[next.length - 1]);
      }
      return next;
    });
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    setIsLoading(true);
    setStatusText("Connecting...");

    // Add user message
    const userMsg: Message = {
      id: generateId(),
      role: "user",
      text,
      tool_calls: [],
      isStreaming: false,
    };

    // Add empty agent message placeholder
    const agentMsg: Message = {
      id: generateId(),
      role: "agent",
      text: "",
      tool_calls: [],
      isStreaming: true,
    };

    setMessages(prev => [...prev, userMsg, agentMsg]);

    // Set up abort controller
    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, company, subject, provider }),
        signal: abort.signal,
      });

      if (!response.body) throw new Error("No response stream");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            // event type on one line, data on next
          } else if (line.startsWith("data: ")) {
            // find the last "event:" before this data line
            const eventLine = lines[lines.indexOf(line) - 1] ?? "";
            const eventType = eventLine.replace("event: ", "").trim();
            const raw = line.replace("data: ", "").trim();
            if (!raw) continue;

            try {
              const data = JSON.parse(raw);
              handleSSEEvent(eventType, data);
            } catch {
              // ignore malformed lines
            }
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        updateLastMessage(msg => ({
          ...msg,
          error: err.message,
          isStreaming: false,
        }));
      }
    } finally {
      setIsLoading(false);
      setStatusText("");
      updateLastMessage(msg => ({ ...msg, isStreaming: false }));
    }
  }, [input, isLoading, company, subject, provider, updateLastMessage]);

  // SSE parser helper that correctly handles multi-line SSE format
  const parseSSEChunk = useCallback((chunk: string) => {
    const events: Array<{ type: string; data: unknown }> = [];
    const lines = chunk.split("\n");
    let currentEvent = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const raw = line.slice(6).trim();
        if (raw) {
          try {
            events.push({ type: currentEvent, data: JSON.parse(raw) });
          } catch { /* skip */ }
        }
        currentEvent = "";
      }
    }
    return events;
  }, []);

  const handleSSEEvent = useCallback((eventType: string, data: unknown) => {
    switch (eventType) {
      case "status":
        setStatusText((data as StatusEvent).message);
        break;

      case "iteration":
        // Visible in status text
        break;

      case "provider":
        updateLastMessage(msg => ({
          ...msg,
          provider: (data as ProviderEvent).name,
        }));
        break;

      case "confidence":
        updateLastMessage(msg => ({
          ...msg,
          confidence: data as ConfidenceEvent,
        }));
        break;

      case "tool":
        updateLastMessage(msg => ({
          ...msg,
          tool_calls: [
            ...msg.tool_calls,
            { ...(data as ToolEvent), timestamp: Date.now() },
          ],
        }));
        break;

      case "token":
        updateLastMessage(msg => ({
          ...msg,
          text: msg.text + (data as TokenEvent).text,
        }));
        break;

      case "result": {
        const result = data as ResultEvent;
        updateLastMessage(msg => ({
          ...msg,
          text: result.response || msg.text,
          status: result.status,
          product_area: result.product_area,
          request_type: result.request_type,
          justification: result.justification,
          isStreaming: false,
        }));
        break;
      }

      case "error":
        updateLastMessage(msg => ({
          ...msg,
          error: (data as ErrorEvent).message,
          isStreaming: false,
        }));
        setStatusText("");
        break;

      case "done":
        updateLastMessage(msg => ({ ...msg, isStreaming: false }));
        break;
    }
  }, [updateLastMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setIsLoading(false);
    updateLastMessage(msg => ({ ...msg, isStreaming: false }));
  };

  const handleClear = () => {
    setMessages([]);
    setStatusText("");
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">⚡</span>
            <span className="logo-text">TriageAI</span>
          </div>
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(v => !v)}>
            {sidebarOpen ? "◀" : "▶"}
          </button>
        </div>

        {sidebarOpen && (
          <>
            {/* Provider selector */}
            <div className="sidebar-section">
              <label className="section-label">LLM Provider</label>
              <div className="provider-grid">
                {PROVIDERS.map(p => (
                  <ProviderPill
                    key={p}
                    provider={p}
                    selected={provider === p}
                    onClick={() => setProvider(p)}
                  />
                ))}
              </div>
              <p className="section-hint">
                "Auto" tries Claude → OpenAI → Gemini with fallback chain.
                Layer 4 may upgrade provider for low-confidence tickets.
              </p>
            </div>

            {/* Company selector */}
            <div className="sidebar-section">
              <label className="section-label">Company</label>
              <select
                className="select-input"
                value={company}
                onChange={e => setCompany(e.target.value)}
              >
                {COMPANIES.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            {/* Subject */}
            <div className="sidebar-section">
              <label className="section-label">Subject (optional)</label>
              <input
                className="text-input"
                type="text"
                placeholder="Brief subject line..."
                value={subject}
                onChange={e => setSubject(e.target.value)}
              />
            </div>

            {/* Safety stack info */}
            <div className="sidebar-section safety-info">
              <label className="section-label">4-Layer Safety Stack</label>
              <ul className="safety-layers">
                <li><span className="layer-num">1</span> Prompt Injection Blocker</li>
                <li><span className="layer-num">2</span> PII & Toxicity Filter</li>
                <li><span className="layer-num">3</span> Confidence Scorer</li>
                <li><span className="layer-num">4</span> Provider Routing Gate</li>
              </ul>
            </div>

            <div className="sidebar-actions">
              <button className="action-btn danger" onClick={handleClear}>
                Clear Chat
              </button>
            </div>
          </>
        )}
      </aside>

      {/* Main chat area */}
      <main className="chat-main">
        {/* Header */}
        <header className="chat-header">
          <h1 className="chat-title">Support Triage Agent</h1>
          <div className="header-meta">
            <span className="header-company">{company !== "None" ? company : "All companies"}</span>
            <span
              className="header-provider"
              style={{ color: PROVIDER_COLORS[provider] }}
            >
              {PROVIDER_LABELS[provider]}
            </span>
          </div>
        </header>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">💬</div>
              <h2>How can I help you today?</h2>
              <p>Ask anything about HackerRank, Claude, or Visa support.</p>
              <div className="example-prompts">
                {[
                  "How do I add extra time for a candidate?",
                  "My Visa card was declined abroad",
                  "How do I delete a Claude conversation?",
                  "My HackerRank assessment won't load",
                ].map(prompt => (
                  <button
                    key={prompt}
                    className="example-prompt"
                    onClick={() => { setInput(prompt); inputRef.current?.focus(); }}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isLoading && statusText && (
            <div className="status-bar">
              <span className="status-spinner">⟳</span>
              {statusText}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <footer className="input-area">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Describe your support issue..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              disabled={isLoading}
            />
            <div className="input-actions">
              {isLoading ? (
                <button className="send-btn stop" onClick={handleStop}>
                  ⏹ Stop
                </button>
              ) : (
                <button
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={!input.trim()}
                >
                  Send ↑
                </button>
              )}
            </div>
          </div>
          <p className="input-hint">Enter to send · Shift+Enter for new line</p>
        </footer>
      </main>
    </div>
  );
};

export default App;
