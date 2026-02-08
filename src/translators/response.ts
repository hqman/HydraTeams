import * as http from "node:http";
import * as crypto from "node:crypto";
import type {
  OpenAIStreamChunk,
  StreamState,
  TrackedToolCall,
} from "./types.js";

/**
 * Send a single SSE event to the client in Anthropic format.
 */
function sendSSE(
  res: http.ServerResponse,
  event: string,
  data: unknown
): void {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

/**
 * Map OpenAI finish_reason to Anthropic stop_reason.
 */
function mapStopReason(
  finishReason: string | null
): "end_turn" | "tool_use" | "max_tokens" {
  if (finishReason === "tool_calls") return "tool_use";
  if (finishReason === "length") return "max_tokens";
  return "end_turn";
}

/**
 * Translate an OpenAI SSE streaming response back to Anthropic SSE format.
 *
 * Reads from an upstream ReadableStream (the OpenAI response body), parses
 * SSE chunks, translates them to Anthropic message events, and writes them
 * to the client's ServerResponse.
 */
export async function translateStream(
  upstreamBody: ReadableStream<Uint8Array>,
  response: http.ServerResponse,
  spoofModel: string
): Promise<void> {
  const state: StreamState = {
    blockIndex: 0,
    activeToolCalls: new Map<number, TrackedToolCall>(),
    textBlockStarted: false,
    messageId: `msg_${crypto.randomUUID()}`,
    spoofModel,
  };

  // 1. Send message_start immediately
  sendSSE(response, "message_start", {
    type: "message_start",
    message: {
      id: state.messageId,
      type: "message",
      role: "assistant",
      model: state.spoofModel,
      content: [],
      stop_reason: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  // 2. Parse OpenAI SSE stream
  const reader = upstreamBody.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let outputTokens = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete lines
      const lines = buffer.split("\n");
      // Last element may be incomplete — keep it in the buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();

        // Skip empty lines (SSE separators) and comments
        if (!trimmed || trimmed.startsWith(":")) continue;

        // Only process data lines
        if (!trimmed.startsWith("data: ")) continue;

        const payload = trimmed.slice(6); // Strip "data: "

        // Handle stream end
        if (payload === "[DONE]") {
          sendSSE(response, "message_stop", { type: "message_stop" });
          response.end();
          return;
        }

        // Parse the JSON chunk
        let chunk: OpenAIStreamChunk;
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue; // Skip malformed chunks
        }

        // Track usage if provided
        if (chunk.usage?.completion_tokens) {
          outputTokens = chunk.usage.completion_tokens;
        }

        const choice = chunk.choices?.[0];
        if (!choice) continue;

        const delta = choice.delta;

        // ─── Text content ───
        if (delta.content) {
          if (!state.textBlockStarted) {
            sendSSE(response, "content_block_start", {
              type: "content_block_start",
              index: state.blockIndex,
              content_block: { type: "text", text: "" },
            });
            state.textBlockStarted = true;
          }
          sendSSE(response, "content_block_delta", {
            type: "content_block_delta",
            index: state.blockIndex,
            delta: { type: "text_delta", text: delta.content },
          });
        }

        // ─── Tool calls ───
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const toolIndex = tc.index;

            if (tc.id) {
              // New tool call starting — close text block if open
              if (state.textBlockStarted) {
                sendSSE(response, "content_block_stop", {
                  type: "content_block_stop",
                  index: state.blockIndex,
                });
                state.blockIndex++;
                state.textBlockStarted = false;
              }

              // Register the new tool call
              state.activeToolCalls.set(toolIndex, {
                id: tc.id,
                name: tc.function?.name || "",
                anthropicIndex: state.blockIndex,
                started: true,
              });

              // Send content_block_start for tool_use
              sendSSE(response, "content_block_start", {
                type: "content_block_start",
                index: state.blockIndex,
                content_block: {
                  type: "tool_use",
                  id: tc.id,
                  name: tc.function?.name || "",
                },
              });
              state.blockIndex++;
            }

            // Stream tool call arguments
            if (tc.function?.arguments) {
              const tracked = state.activeToolCalls.get(toolIndex);
              if (tracked) {
                sendSSE(response, "content_block_delta", {
                  type: "content_block_delta",
                  index: tracked.anthropicIndex,
                  delta: {
                    type: "input_json_delta",
                    partial_json: tc.function.arguments,
                  },
                });
              }
            }
          }
        }

        // ─── Finish ───
        if (choice.finish_reason) {
          // Close open text block
          if (state.textBlockStarted) {
            sendSSE(response, "content_block_stop", {
              type: "content_block_stop",
              index: state.blockIndex,
            });
          }

          // Close all tracked tool call blocks
          for (const [, tc] of state.activeToolCalls) {
            sendSSE(response, "content_block_stop", {
              type: "content_block_stop",
              index: tc.anthropicIndex,
            });
          }

          // Send message_delta with stop reason and usage
          sendSSE(response, "message_delta", {
            type: "message_delta",
            delta: { stop_reason: mapStopReason(choice.finish_reason) },
            usage: { input_tokens: 0, output_tokens: outputTokens },
          });
        }
      }
    }

    // Flush any remaining buffer content (edge case: stream ended without final newline)
    if (buffer.trim().startsWith("data: ")) {
      const payload = buffer.trim().slice(6);
      if (payload === "[DONE]") {
        sendSSE(response, "message_stop", { type: "message_stop" });
      }
    }
  } finally {
    reader.releaseLock();
  }

  response.end();
}
