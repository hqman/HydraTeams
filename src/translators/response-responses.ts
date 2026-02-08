import * as http from "node:http";

/**
 * Translates OpenAI Responses API SSE streaming events to Anthropic Messages API SSE format.
 *
 * OpenAI Responses API events:
 * - response.created, response.in_progress
 * - response.output_item.added (message or function_call)
 * - response.content_part.added (output_text)
 * - response.output_text.delta, response.output_text.done
 * - response.function_call_arguments.delta, response.function_call_arguments.done
 * - response.output_item.done
 * - response.completed
 *
 * Anthropic Messages API events:
 * - message_start
 * - content_block_start (text or tool_use)
 * - content_block_delta (text_delta or input_json_delta)
 * - content_block_stop
 * - message_delta (stop_reason + usage)
 * - message_stop
 */

interface ResponsesAPIEvent {
  type: string;
  response?: {
    id: string;
    model: string;
    status: string;
    output?: Array<{
      type: string;
      id?: string;
      role?: string;
      content?: any;
      call_id?: string;
      name?: string;
      arguments?: string;
    }>;
    usage?: {
      input_tokens: number;
      output_tokens: number;
    };
    status_details?: {
      type?: string;
      reason?: string;
    };
  };
  output_index?: number;
  content_index?: number;
  item?: {
    id: string;
    type: string;
    role?: string;
    content?: any;
    call_id?: string;
    name?: string;
    arguments?: string;
    status?: string;
  };
  part?: {
    type: string;
    text?: string;
  };
  delta?: string;
  text?: string;
  arguments?: string;
}

interface TranslatorState {
  messageId: string;
  blockIndex: number;
  currentBlocks: Map<number, {
    type: "text" | "tool_use";
    id?: string;
    name?: string;
  }>;
}

/**
 * Translates OpenAI Responses API SSE stream to Anthropic Messages API SSE stream.
 */
export async function translateResponsesStream(
  upstreamBody: ReadableStream<Uint8Array>,
  response: http.ServerResponse,
  spoofModel: string
): Promise<void> {
  const reader = upstreamBody.getReader();
  const decoder = new TextDecoder();

  const state: TranslatorState = {
    messageId: "",
    blockIndex: -1,
    currentBlocks: new Map(),
  };

  let buffer = "";
  let currentEvent = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          const dataStr = line.slice(5).trim();
          if (!dataStr) continue;

          try {
            const event: ResponsesAPIEvent = JSON.parse(dataStr);
            const anthropicEvents = translateEvent(event, currentEvent, state, spoofModel);

            for (const anthEvent of anthropicEvents) {
              response.write(`event: ${anthEvent.event}\n`);
              response.write(`data: ${JSON.stringify(anthEvent.data)}\n\n`);
            }
          } catch (e) {
            console.error("Failed to parse SSE data:", dataStr, e);
          }

          currentEvent = "";
        } else if (line === "") {
          // Empty line separates events
          currentEvent = "";
        }
      }
    }

    response.end();
  } catch (error) {
    console.error("Error in translateResponsesStream:", error);
    response.end();
  }
}

/**
 * Translates a single OpenAI Responses API event to one or more Anthropic events.
 */
function translateEvent(
  event: ResponsesAPIEvent,
  eventType: string,
  state: TranslatorState,
  spoofModel: string
): Array<{ event: string; data: any }> {
  const results: Array<{ event: string; data: any }> = [];

  switch (event.type) {
    case "response.created": {
      // Send message_start
      state.messageId = event.response?.id || `msg_${Date.now()}`;
      results.push({
        event: "message_start",
        data: {
          type: "message_start",
          message: {
            id: state.messageId,
            type: "message",
            role: "assistant",
            model: spoofModel,
            content: [],
            stop_reason: null,
            usage: {
              input_tokens: 0,
              output_tokens: 0,
            },
          },
        },
      });
      break;
    }

    case "response.output_item.added": {
      const item = event.item;
      if (!item) break;

      if (item.type === "message") {
        // Message item added - we'll handle content parts separately
        // No event needed here
      } else if (item.type === "function_call") {
        // Tool use block starting
        state.blockIndex++;
        state.currentBlocks.set(state.blockIndex, {
          type: "tool_use",
          id: item.call_id,
          name: item.name,
        });

        results.push({
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: state.blockIndex,
            content_block: {
              type: "tool_use",
              id: item.call_id || `toolu_${Date.now()}`,
              name: item.name || "unknown",
              input: {},
            },
          },
        });
      }
      break;
    }

    case "response.content_part.added": {
      const part = event.part;
      if (!part) break;

      if (part.type === "output_text") {
        // Text block starting
        state.blockIndex++;
        state.currentBlocks.set(state.blockIndex, {
          type: "text",
        });

        results.push({
          event: "content_block_start",
          data: {
            type: "content_block_start",
            index: state.blockIndex,
            content_block: {
              type: "text",
              text: "",
            },
          },
        });
      }
      break;
    }

    case "response.output_text.delta": {
      // Text delta
      const blockIdx = event.content_index ?? state.blockIndex;
      results.push({
        event: "content_block_delta",
        data: {
          type: "content_block_delta",
          index: blockIdx,
          delta: {
            type: "text_delta",
            text: event.delta || "",
          },
        },
      });
      break;
    }

    case "response.output_text.done": {
      // Text block complete
      const blockIdx = event.content_index ?? state.blockIndex;
      results.push({
        event: "content_block_stop",
        data: {
          type: "content_block_stop",
          index: blockIdx,
        },
      });
      break;
    }

    case "response.function_call_arguments.delta": {
      // Tool use input delta
      const blockIdx = event.output_index ?? state.blockIndex;
      results.push({
        event: "content_block_delta",
        data: {
          type: "content_block_delta",
          index: blockIdx,
          delta: {
            type: "input_json_delta",
            partial_json: event.delta || "",
          },
        },
      });
      break;
    }

    case "response.function_call_arguments.done": {
      // Tool use block complete
      const blockIdx = event.output_index ?? state.blockIndex;
      results.push({
        event: "content_block_stop",
        data: {
          type: "content_block_stop",
          index: blockIdx,
        },
      });
      break;
    }

    case "response.completed": {
      // Map stop reason
      let stopReason = "end_turn";
      const output = event.response?.output || [];
      const hasToolCall = output.some((item) => item.type === "function_call");

      if (hasToolCall) {
        stopReason = "tool_use";
      } else if (event.response?.status === "incomplete") {
        const reason = event.response?.status_details?.reason;
        if (reason === "max_output_tokens") {
          stopReason = "max_tokens";
        }
      }

      const inputTokens = event.response?.usage?.input_tokens || 0;
      const outputTokens = event.response?.usage?.output_tokens || 0;

      // Send message_delta
      results.push({
        event: "message_delta",
        data: {
          type: "message_delta",
          delta: {
            stop_reason: stopReason,
          },
          usage: {
            input_tokens: inputTokens,
            output_tokens: outputTokens,
          },
        },
      });

      // Send message_stop
      results.push({
        event: "message_stop",
        data: {
          type: "message_stop",
        },
      });
      break;
    }

    // Ignore other event types (response.in_progress, response.output_item.done, etc.)
    default:
      break;
  }

  return results;
}
