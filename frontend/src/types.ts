/**
 * types.ts — TypeScript interfaces for all SSE events and API shapes.
 *
 * Keeps the App component type-safe. Any new event type from the backend
 * should be added here first.
 */

// ── Provider types ────────────────────────────────────────────────────────────

export type Provider = "claude" | "openai" | "gemini" | "auto";

export const PROVIDER_LABELS: Record<Provider, string> = {
  claude: "Claude",
  openai: "OpenAI",
  gemini: "Gemini",
  auto: "Auto",
};

export const PROVIDER_COLORS: Record<Provider, string> = {
  claude: "#d97706",   // amber
  openai: "#16a34a",   // green
  gemini: "#2563eb",   // blue
  auto: "#7c3aed",     // violet
};

// ── SSE Event shapes (from backend) ──────────────────────────────────────────

export interface StatusEvent {
  message: string;
}

export interface IterationEvent {
  current: number;
  max: number;
}

export interface ProviderEvent {
  name: Provider;
}

export interface ConfidenceEvent {
  score: number;
  level: "high" | "medium" | "low";
  provider_routed_to: Provider;
}

export interface ToolEvent {
  name: string;
  query: string;
  company: string;
  reason: string;
}

export interface TokenEvent {
  text: string;
}

export interface ResultEvent {
  status: "replied" | "escalated";
  product_area: string;
  response: string;
  justification: string;
  request_type: "product_issue" | "feature_request" | "bug" | "invalid";
}

export interface ErrorEvent {
  message: string;
}

// ── Chat message shapes ───────────────────────────────────────────────────────

export type MessageRole = "user" | "agent";

export interface ToolCall {
  name: string;
  query: string;
  company: string;
  reason: string;
  timestamp: number;
}

export interface Message {
  id: string;
  role: MessageRole;
  text: string;                       // Streamed response text
  status?: "replied" | "escalated";
  product_area?: string;
  request_type?: string;
  justification?: string;
  provider?: Provider;                // Which provider generated this
  confidence?: ConfidenceEvent;       // Layer 3 result
  tool_calls: ToolCall[];             // Live tool calls during reasoning
  iteration?: number;
  isStreaming: boolean;
  error?: string;
}

// ── API Request shape ─────────────────────────────────────────────────────────

export interface ChatAPIRequest {
  message: string;
  company: string;
  subject: string;
  provider: Provider;
}
