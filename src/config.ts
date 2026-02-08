import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { ProxyConfig } from "./translators/types.js";

interface CodexAuth {
  accessToken: string;
  accountId: string;
}

function loadCodexAuth(): CodexAuth | null {
  try {
    const authPath = join(homedir(), ".codex", "auth.json");
    const auth = JSON.parse(readFileSync(authPath, "utf-8"));
    const token = auth.tokens?.access_token;
    if (!token) return null;

    // Extract chatgpt_account_id from JWT
    const payload = token.split(".")[1];
    const decoded = JSON.parse(Buffer.from(payload, "base64url").toString());
    const authClaim = decoded["https://api.openai.com/auth"] || {};
    const accountId = authClaim.chatgpt_account_id || "";

    console.log(`Using codex auth from ~/.codex/auth.json (plan: ${authClaim.chatgpt_plan_type || "unknown"})`);
    return { accessToken: token, accountId };
  } catch {
    return null;
  }
}

export function loadConfig(args: string[]): ProxyConfig {
  const getArg = (flag: string): string | undefined => {
    const idx = args.indexOf(flag);
    return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : undefined;
  };

  const port = parseInt(getArg("--port") || process.env.HYDRA_PROXY_PORT || "3456", 10);
  const targetModel = getArg("--model") || process.env.HYDRA_TARGET_MODEL || "";
  const targetProvider = (getArg("--provider") || process.env.HYDRA_TARGET_PROVIDER || "openai") as ProxyConfig["targetProvider"];
  const spoofModel = getArg("--spoof") || process.env.HYDRA_SPOOF_MODEL || "claude-sonnet-4-5-20250929";
  const targetUrl = getArg("--target-url") || process.env.HYDRA_TARGET_URL;

  // Passthrough config
  const passthroughArg = getArg("--passthrough");
  let passthroughModels: string[] = [];
  if (passthroughArg) {
    passthroughModels = passthroughArg.split(",").map(m => m.trim());
  } else if (args.includes("--passthrough")) {
    passthroughModels = ["*"];
  } else if (process.env.HYDRA_PASSTHROUGH) {
    const envVal = process.env.HYDRA_PASSTHROUGH;
    passthroughModels = envVal === "true" ? ["*"] : envVal.split(",").map(m => m.trim());
  }

  const anthropicApiKey = process.env.ANTHROPIC_API_KEY;

  // Load auth based on provider
  let openaiApiKey = "";
  let chatgptAccessToken = "";
  let chatgptAccountId = "";

  if (targetProvider === "chatgpt") {
    const codexAuth = loadCodexAuth();
    if (!codexAuth) {
      console.error("Error: No codex auth found. Run: codex --login");
      process.exit(1);
    }
    chatgptAccessToken = codexAuth.accessToken;
    chatgptAccountId = codexAuth.accountId;
  } else if (targetProvider === "openai") {
    openaiApiKey = process.env.OPENAI_API_KEY || "";
    if (!openaiApiKey) {
      const codexAuth = loadCodexAuth();
      openaiApiKey = codexAuth?.accessToken || "";
    }
    if (!openaiApiKey && !targetUrl) {
      console.error("Error: No OpenAI API key found.");
      console.error("  Set OPENAI_API_KEY env var, login with: codex --login, or use --target-url for local servers");
      process.exit(1);
    }
  }

  if (!targetModel) {
    console.error("Error: --model is required (e.g., --model gpt-5.3-codex)");
    process.exit(1);
  }

  if (passthroughModels.length > 0) {
    console.log("Passthrough enabled — Claude Code auth headers will be relayed to Anthropic API.");
  }

  return { port, targetModel, targetProvider, targetUrl, openaiApiKey, spoofModel, passthroughModels, anthropicApiKey, chatgptAccessToken, chatgptAccountId };
}
