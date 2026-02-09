import http from "node:http";
import type { ProxyConfig } from "./translators/types.js";
import { translateRequest } from "./translators/request.js";
import { translateStream } from "./translators/response.js";
import { translateRequestToResponses } from "./translators/request-responses.js";
import { translateResponsesStream } from "./translators/response-responses.js";
import { ProxyLogger } from "./logger.js";

const DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions";
const CHATGPT_API_URL = "https://chatgpt.com/backend-api/codex/responses";
const DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com";

const LEAD_MARKER = "hydra:lead";
const TEAMMATE_MARKER = "the user interacts primarily with the team lead";

function shouldPassthrough(
  model: string,
  passthroughModels: string[],
  searchText?: string,
): boolean {
  if (passthroughModels.length === 0) return false;
  if (passthroughModels.includes("*")) return model.startsWith("claude-");

  if (passthroughModels.includes("lead")) {
    return !!searchText && searchText.includes(LEAD_MARKER);
  }

  return passthroughModels.includes(model);
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

function safeEnd(res: http.ServerResponse, payload?: string): void {
  if (res.writableEnded || res.destroyed) return;
  if (payload === undefined) {
    res.end();
    return;
  }
  res.end(payload);
}

async function handlePassthrough(
  body: string,
  headers: http.IncomingHttpHeaders,
  res: http.ServerResponse,
  upstreamUrl: string,
  onStatus?: (status: number) => void,
): Promise<void> {
  const forwardHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const relayKeys = [
    "x-api-key", "authorization", "anthropic-version", "anthropic-beta",
    "cookie", "x-request-id",
  ];
  for (const key of relayKeys) {
    if (headers[key]) {
      forwardHeaders[key] = headers[key] as string;
    }
  }

  const upstream = await fetch(upstreamUrl, {
    method: "POST",
    headers: forwardHeaders,
    body,
  });
  if (onStatus) onStatus(upstream.status);

  res.writeHead(upstream.status, {
    "Content-Type": upstream.headers.get("content-type") || "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  if (!upstream.body) {
    safeEnd(res);
    return;
  }

  const reader = upstream.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (res.writableEnded || res.destroyed) break;
      res.write(value);
    }
  } finally {
    reader.releaseLock();
    safeEnd(res);
  }
}

function normalizeAnthropicMessagesUrl(baseUrl: string, countTokens: boolean): string {
  const stripped = baseUrl.replace(/\/+$/, "");
  const messagesUrl = stripped.endsWith("/v1/messages")
    ? stripped
    : `${stripped}/v1/messages`;
  return countTokens ? `${messagesUrl}/count_tokens` : messagesUrl;
}

function formatPassthroughTarget(baseUrl: string): string {
  try {
    const parsed = new URL(baseUrl);
    const host = parsed.port ? `${parsed.hostname}:${parsed.port}` : parsed.hostname;
    return `Anthropic@${host}`;
  } catch {
    return `Anthropic@${baseUrl}`;
  }
}

function extractSystemText(system: unknown): string {
  if (typeof system === "string") return system;
  if (Array.isArray(system)) {
    return system
      .map((block: { text?: string }) => block.text || "")
      .join(" ");
  }
  return "";
}

export function createProxyServer(config: ProxyConfig): http.Server {
  const logger = new ProxyLogger();

  logger.banner({
    targetModel: config.targetModel,
    spoofModel: config.spoofModel,
    port: config.port,
    provider: config.targetProvider,
    passthrough: config.passthroughModels,
  });

  return http.createServer(async (req, res) => {
    const pathname = (req.url || "").split("?")[0];

    // Health check
    if (req.method === "GET" && (pathname === "/" || pathname === "/health")) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        status: "ok",
        targetModel: config.targetModel,
        spoofModel: config.spoofModel,
        passthroughModels: config.passthroughModels,
      }));
      return;
    }

    // Handle count_tokens
    if (req.method === "POST" && pathname === "/v1/messages/count_tokens") {
      const body = await readBody(req);
      try {
        const parsed = JSON.parse(body);
        if (parsed.model && shouldPassthrough(parsed.model, config.passthroughModels, parsed.system)) {
          const systemText = extractSystemText(parsed.system);
          const isLeadPassthrough = config.passthroughModels.includes("lead")
            && systemText.includes(LEAD_MARKER);
          const passthroughBaseUrl = isLeadPassthrough && config.leadBaseUrl
            ? config.leadBaseUrl
            : DEFAULT_ANTHROPIC_BASE_URL;
          await handlePassthrough(
            body,
            req.headers,
            res,
            normalizeAnthropicMessagesUrl(passthroughBaseUrl, true),
          );
          return;
        }
        const estimatedTokens = JSON.stringify(parsed.messages || []).length / 4;
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ input_tokens: Math.ceil(estimatedTokens) }));
      } catch {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ input_tokens: 1000 }));
      }
      return;
    }

    // Only handle POST /v1/messages
    if (req.method !== "POST" || pathname !== "/v1/messages") {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { type: "not_found", message: "Not found" } }));
      return;
    }

    let session: ReturnType<ProxyLogger["identify"]> | null = null;
    let requestModel = "unknown";
    let routeLabel = "unknown";
    try {
      const body = await readBody(req);
      const anthropicReq = JSON.parse(body);
      requestModel = anthropicReq.model || "unknown";

      // Extract system text for routing decisions
      const systemText = extractSystemText(anthropicReq.system);

      const msgText = (anthropicReq.messages || []).slice(0, 3).map((m: { content?: string | Array<{ text?: string }> }) => {
        if (typeof m.content === "string") return m.content;
        if (Array.isArray(m.content)) return m.content.map((b: { text?: string }) => b.text || "").join(" ");
        return "";
      }).join(" ");

      const fullText = systemText + " " + msgText;
      const hasMarker = fullText.includes(LEAD_MARKER);
      const isTeammate = systemText.toLowerCase().includes(TEAMMATE_MARKER);
      const isStreaming = anthropicReq.stream !== false;
      const msgCount = anthropicReq.messages?.length || 0;
      const toolCount = anthropicReq.tools?.length || 0;

      // Routing: teammates always translate, lead passthrough if configured
      const isPassthrough = !isTeammate && shouldPassthrough(anthropicReq.model, config.passthroughModels, fullText);
      const isLeadPassthrough = config.passthroughModels.includes("lead") && hasMarker;
      const passthroughBaseUrl = isLeadPassthrough && config.leadBaseUrl
        ? config.leadBaseUrl
        : DEFAULT_ANTHROPIC_BASE_URL;
      const passthroughTarget = formatPassthroughTarget(passthroughBaseUrl);
      routeLabel = isPassthrough ? passthroughTarget : config.targetModel;

      // Identify agent session
      session = logger.identify({ toolCount, msgCount, isTeammate, systemText });

      // Log the request
      logger.logRequest(session, {
        model: anthropicReq.model,
        msgCount,
        toolCount,
        isStreaming,
        isTeammate,
        hasMarker,
        systemLength: systemText.length,
        route: isPassthrough ? "passthrough" : "translate",
        targetModel: isPassthrough ? undefined : config.targetModel,
        passthroughTarget: isPassthrough ? passthroughTarget : undefined,
      });

      // ─── Passthrough to Anthropic-compatible upstream ───
      if (isPassthrough) {
        await handlePassthrough(
          body,
          req.headers,
          res,
          normalizeAnthropicMessagesUrl(passthroughBaseUrl, false),
          (status) => {
            if (status >= 400) {
              logger.logError(
                session!,
                `Passthrough ${status} (model=${anthropicReq.model}, target=${passthroughTarget})`,
              );
            }
          },
        );
        return;
      }

      // ─── Route to target provider ───

      if (config.targetProvider === "chatgpt") {
        // ─── ChatGPT Backend (Responses API) ───
        const responsesReq = translateRequestToResponses(anthropicReq, config.targetModel);

        const MAX_RETRIES = 5;
        let upstream: Response | null = null;
        for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
          upstream = await fetch(CHATGPT_API_URL, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${config.chatgptAccessToken}`,
              "Chatgpt-Account-Id": config.chatgptAccountId || "",
              "User-Agent": "codex-cli/1.0",
            },
            body: JSON.stringify(responsesReq),
          });

          if (upstream.status !== 429) break;
          if (attempt < MAX_RETRIES) {
            const waitMs = Math.min(1000 * Math.pow(2, attempt), 10000);
            logger.logFile(session, `Rate limited (429), retry ${attempt + 1}/${MAX_RETRIES} in ${waitMs}ms`);
            await new Promise(r => setTimeout(r, waitMs));
          }
        }

        if (!upstream || !upstream.ok) {
          const errText = upstream ? await upstream.text() : "No response";
          const status = upstream?.status || 500;
          logger.logError(
            session,
            `ChatGPT ${status} (model=${anthropicReq.model}, target=${config.targetModel}): ${errText.slice(0, 300)}`,
          );
          res.writeHead(status, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ type: "error", error: { type: "api_error", message: errText } }));
          return;
        }

        if (!upstream.body) {
          logger.logError(session, `No response body from ChatGPT (model=${anthropicReq.model}, target=${config.targetModel})`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ type: "error", error: { type: "api_error", message: "No response body" } }));
          return;
        }

        res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
        await translateResponsesStream(upstream.body, res, config.spoofModel);

      } else {
        // ─── OpenAI Chat Completions ───
        const openaiUrl = config.targetUrl || DEFAULT_OPENAI_URL;
        const openaiReq = translateRequest(anthropicReq, config.targetModel);

        if (!isStreaming) {
          openaiReq.stream = false;
          delete openaiReq.stream_options;
        }

        const MAX_RETRIES = 5;
        let upstream: Response | null = null;
        for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
          const headers: Record<string, string> = { "Content-Type": "application/json" };
          if (config.openaiApiKey) headers["Authorization"] = `Bearer ${config.openaiApiKey}`;
          upstream = await fetch(openaiUrl, {
            method: "POST",
            headers,
            body: JSON.stringify(openaiReq),
          });

          if (upstream.status !== 429) break;
          if (attempt < MAX_RETRIES) {
            const waitMs = Math.min(1000 * Math.pow(2, attempt), 10000);
            logger.logFile(session, `Rate limited (429), retry ${attempt + 1}/${MAX_RETRIES} in ${waitMs}ms`);
            await new Promise(r => setTimeout(r, waitMs));
          }
        }

        if (!upstream || !upstream.ok) {
          const errText = upstream ? await upstream.text() : "No response";
          const status = upstream?.status || 500;
          logger.logError(
            session,
            `OpenAI ${status} (model=${anthropicReq.model}, target=${config.targetModel}): ${errText.slice(0, 300)}`,
          );
          const errorType = status === 429 ? "rate_limit_error" : status === 401 ? "authentication_error" : status >= 500 ? "api_error" : "invalid_request_error";
          res.writeHead(status, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ type: "error", error: { type: errorType, message: errText } }));
          return;
        }

        if (!isStreaming) {
          const openaiRes = await upstream.json() as {
            choices?: Array<{ message?: { content?: string; tool_calls?: Array<{ id: string; function: { name: string; arguments: string } }> }; finish_reason?: string }>;
            usage?: { prompt_tokens?: number; completion_tokens?: number };
          };
          const choice = openaiRes.choices?.[0];
          const content: Array<{ type: string; text?: string; id?: string; name?: string; input?: unknown }> = [];
          if (choice?.message?.content) content.push({ type: "text", text: choice.message.content });
          if (choice?.message?.tool_calls) {
            for (const tc of choice.message.tool_calls) {
              content.push({ type: "tool_use", id: tc.id, name: tc.function.name, input: JSON.parse(tc.function.arguments || "{}") });
            }
          }
          const stopReason = choice?.finish_reason === "tool_calls" ? "tool_use" : choice?.finish_reason === "length" ? "max_tokens" : "end_turn";
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ id: `msg_${Date.now()}`, type: "message", role: "assistant", model: config.spoofModel, content, stop_reason: stopReason, usage: { input_tokens: openaiRes.usage?.prompt_tokens || 0, output_tokens: openaiRes.usage?.completion_tokens || 0 } }));
          return;
        }

        if (!upstream.body) {
          logger.logError(session, `No response body from OpenAI (model=${anthropicReq.model}, target=${config.targetModel})`);
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ type: "error", error: { type: "api_error", message: "No response body" } }));
          return;
        }

        res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
        await translateStream(upstream.body, res, config.spoofModel);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Internal proxy error";
      if (session) {
        logger.logError(session, `Proxy error (model=${requestModel}, route=${routeLabel}): ${message}`);
      } else {
        console.error(`[proxy] ERROR (model=${requestModel}, route=${routeLabel}): ${message}`);
      }
      if (res.writableEnded || res.destroyed) return;

      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
        safeEnd(res, JSON.stringify({
          type: "error",
          error: { type: "api_error", message },
        }));
        return;
      }

      // Response already started (e.g., stream mode), so only terminate safely.
      safeEnd(res);
    }
  });
}
