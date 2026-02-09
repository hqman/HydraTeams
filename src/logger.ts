import * as fs from "node:fs";
import * as path from "node:path";

// ANSI colors
const C = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  bold: "\x1b[1m",
  cyan: "\x1b[36m",
  yellow: "\x1b[33m",
  green: "\x1b[32m",
  magenta: "\x1b[35m",
  red: "\x1b[31m",
  blue: "\x1b[34m",
};

const TEAMMATE_COLORS = [C.yellow, C.green, C.magenta, C.blue];

interface AgentSession {
  key: string;
  label: string;
  color: string;
  type: "lead" | "teammate" | "warmup";
  lastMsgCount: number;
  requestCount: number;
  logStream: fs.WriteStream;
}

export interface RequestInfo {
  model: string;
  msgCount: number;
  toolCount: number;
  isStreaming: boolean;
  isTeammate: boolean;
  hasMarker: boolean;
  systemLength: number;
  route: "passthrough" | "translate" | "count_tokens";
  targetModel?: string;
  passthroughTarget?: string;
}

export class ProxyLogger {
  private sessions = new Map<string, AgentSession>();
  private teammateIndex = 0;
  private logDir: string;
  private totalRequests = 0;

  constructor(logDir = "./logs") {
    this.logDir = path.resolve(logDir);
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
    // Clear old logs on startup
    try {
      for (const file of fs.readdirSync(this.logDir)) {
        if (file.endsWith(".log")) {
          fs.unlinkSync(path.join(this.logDir, file));
        }
      }
    } catch { /* ignore */ }
  }

  /**
   * Identify which agent sent this request and return its session.
   */
  identify(info: {
    toolCount: number;
    msgCount: number;
    isTeammate: boolean;
    systemText: string;
  }): AgentSession {
    // Warmup (haiku, 0 tools)
    if (info.toolCount === 0) {
      return this.getOrCreate("WARM", "warmup", info.msgCount);
    }

    // Lead
    if (!info.isTeammate) {
      return this.getOrCreate("LEAD", "lead", info.msgCount);
    }

    // Teammate - match by message count proximity
    let bestMatch: AgentSession | null = null;
    let bestDiff = Infinity;

    for (const [, session] of this.sessions) {
      if (session.type !== "teammate") continue;
      const diff = info.msgCount - session.lastMsgCount;
      // Messages increase by ~2 per turn, allow up to 8 for multi-tool loops
      if (diff >= 0 && diff <= 8 && diff < bestDiff) {
        bestMatch = session;
        bestDiff = diff;
      }
    }

    if (bestMatch) {
      bestMatch.lastMsgCount = info.msgCount;
      bestMatch.requestCount++;
      return bestMatch;
    }

    // New teammate
    this.teammateIndex++;
    const key = `T${this.teammateIndex}`;
    const name = this.extractName(info.systemText) || undefined;
    return this.getOrCreate(key, "teammate", info.msgCount, name);
  }

  /**
   * Log a request - compact console line + detailed file entry.
   */
  logRequest(session: AgentSession, info: RequestInfo): void {
    this.totalRequests++;
    const ts = this.ts();
    const num = String(session.requestCount).padStart(3);
    const modelLabel = info.route === "passthrough" ? `model=${info.model}, ` : "";

    const route = info.route === "passthrough"
      ? (info.passthroughTarget || "Anthropic")
      : info.targetModel || "GPT";

    // Console: one compact line
    console.log(
      `${session.color}[${this.pad(session.label, 10)}]${C.reset} ` +
      `${C.dim}${ts}${C.reset} ` +
      `→ ${route} ` +
      `${C.dim}(${modelLabel}${info.msgCount} msgs, ${info.toolCount} tools, #${num})${C.reset}`
    );

    // File: detailed
    session.logStream.write(
      `[${ts}] #${num} → ${route}\n` +
      `  model=${info.model} msgs=${info.msgCount} tools=${info.toolCount} ` +
      `stream=${info.isStreaming} marker=${info.hasMarker} teammate=${info.isTeammate}\n` +
      `  system=${info.systemLength} chars\n\n`
    );

    // Periodic summary every 25 requests
    if (this.totalRequests % 25 === 0) {
      this.summary();
    }
  }

  logError(session: AgentSession, error: string): void {
    const ts = this.ts();
    console.log(`${C.red}[${this.pad(session.label, 10)}] ${ts} ERROR: ${error}${C.reset}`);
    session.logStream.write(`[${ts}] ERROR: ${error}\n\n`);
  }

  logFile(session: AgentSession, message: string): void {
    session.logStream.write(`  ${message}\n`);
  }

  banner(config: {
    targetModel: string;
    spoofModel: string;
    port: number;
    provider: string;
    passthrough: string[];
  }): void {
    console.log(`\n${C.bold}HydraTeams Proxy${C.reset}`);
    console.log(`${C.dim}${"─".repeat(40)}${C.reset}`);
    console.log(`  Port:        ${C.cyan}:${config.port}${C.reset}`);
    console.log(`  Target:      ${C.yellow}${config.targetModel}${C.reset} (${config.provider})`);
    console.log(`  Spoof:       ${config.spoofModel}`);
    console.log(`  Passthrough: ${config.passthrough.join(", ") || "none"}`);
    console.log(`  Logs:        ${this.logDir}/`);
    console.log(`${C.dim}${"─".repeat(40)}${C.reset}\n`);
  }

  summary(): void {
    const parts: string[] = [];
    for (const [, s] of this.sessions) {
      if (s.type === "warmup") continue;
      parts.push(`${s.color}${s.label}${C.reset}:${s.requestCount}`);
    }
    if (parts.length) {
      console.log(`\n${C.dim}── ${parts.join("  ")} ──${C.reset}\n`);
    }
  }

  shutdown(): void {
    for (const [, session] of this.sessions) {
      session.logStream.end();
    }
  }

  // ─── Private ───

  private getOrCreate(
    key: string,
    type: AgentSession["type"],
    msgCount: number,
    label?: string,
  ): AgentSession {
    let session = this.sessions.get(key);
    if (!session) {
      const color =
        type === "lead" ? C.cyan
        : type === "warmup" ? C.dim
        : TEAMMATE_COLORS[(this.teammateIndex - 1) % TEAMMATE_COLORS.length];

      const displayLabel = label || key;
      const logPath = path.join(this.logDir, `${key.toLowerCase()}.log`);

      session = {
        key,
        label: displayLabel,
        color,
        type,
        lastMsgCount: msgCount,
        requestCount: 0,
        logStream: fs.createWriteStream(logPath, { flags: "a" }),
      };
      this.sessions.set(key, session);

      if (type !== "warmup") {
        console.log(
          `\n${color}${C.bold}● ${displayLabel} connected${C.reset} ` +
          `${C.dim}(${type}, → ${key.toLowerCase()}.log)${C.reset}\n`
        );
      }
    }
    session.lastMsgCount = msgCount;
    session.requestCount++;
    return session;
  }

  private extractName(systemText: string): string | null {
    // Try patterns Claude Code uses to identify teammates
    const patterns = [
      /You are a teammate.*?named? "?(\w[\w-]*)"?/i,
      /your (?:agent )?name is "?(\w[\w-]*)"?/i,
      /Name:\s*["']?(\w[\w-]*)/,
    ];
    for (const p of patterns) {
      const m = systemText.match(p);
      if (m) return m[1];
    }
    return null;
  }

  private ts(): string {
    return new Date().toISOString().slice(11, 19);
  }

  private pad(s: string, n: number): string {
    return s.length >= n ? s.slice(0, n) : s + " ".repeat(n - s.length);
  }
}
