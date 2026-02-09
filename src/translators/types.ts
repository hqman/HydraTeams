// ─── Anthropic Messages API Types ─────────────────────────────

export interface AnthropicRequest {
  model: string;
  system?: string | AnthropicSystemBlock[];
  messages: AnthropicMessage[];
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
  max_tokens: number;
  temperature?: number;
  stream?: boolean;
  metadata?: Record<string, unknown>;
}

export interface AnthropicSystemBlock {
  type: "text";
  text: string;
  cache_control?: { type: string };
}

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

export type AnthropicContentBlock =
  | AnthropicTextBlock
  | AnthropicToolUseBlock
  | AnthropicToolResultBlock;

export interface AnthropicTextBlock {
  type: "text";
  text: string;
}

export interface AnthropicToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface AnthropicToolResultBlock {
  type: "tool_result";
  tool_use_id: string;
  content: string | AnthropicContentBlock[];
  is_error?: boolean;
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

export type AnthropicToolChoice =
  | { type: "auto" }
  | { type: "any" }
  | { type: "tool"; name: string }
  | { type: "none" };

// ─── OpenAI Chat Completions API Types ───────────────────────

export interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  tools?: OpenAITool[];
  tool_choice?: OpenAIToolChoice;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  stream_options?: { include_usage: boolean };
}

export type OpenAIMessage =
  | OpenAISystemMessage
  | OpenAIUserMessage
  | OpenAIAssistantMessage
  | OpenAIToolMessage;

export interface OpenAISystemMessage {
  role: "system";
  content: string;
}

export interface OpenAIUserMessage {
  role: "user";
  content: string;
}

export interface OpenAIAssistantMessage {
  role: "assistant";
  content: string | null;
  tool_calls?: OpenAIToolCall[];
}

export interface OpenAIToolMessage {
  role: "tool";
  tool_call_id: string;
  content: string;
}

export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export type OpenAIToolChoice =
  | "auto"
  | "required"
  | "none"
  | { type: "function"; function: { name: string } };

// ─── OpenAI Streaming Types ──────────────────────────────────

export interface OpenAIStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIStreamChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface OpenAIStreamChoice {
  index: number;
  delta: {
    role?: string;
    content?: string | null;
    tool_calls?: OpenAIStreamToolCall[];
  };
  finish_reason: string | null;
}

export interface OpenAIStreamToolCall {
  index: number;
  id?: string;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
}

// ─── Proxy Config ────────────────────────────────────────────

export interface ProxyConfig {
  port: number;
  targetModel: string;
  targetProvider: "openai" | "chatgpt" | "google" | "ollama";
  targetUrl?: string;
  leadBaseUrl?: string;
  openaiApiKey: string;
  spoofModel: string;
  passthroughModels: string[];
  anthropicApiKey?: string;
  chatgptAccessToken?: string;
  chatgptAccountId?: string;
}

// ─── Stream Translation State ────────────────────────────────

export interface StreamState {
  blockIndex: number;
  activeToolCalls: Map<number, TrackedToolCall>;
  textBlockStarted: boolean;
  messageId: string;
  spoofModel: string;
}

export interface TrackedToolCall {
  id: string;
  name: string;
  anthropicIndex: number;
  started: boolean;
}
