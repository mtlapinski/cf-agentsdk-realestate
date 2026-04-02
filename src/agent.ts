import { AIChatAgent } from 'agents/ai-chat-agent';
import {
  convertToModelMessages,
  streamText,
  tool,
  stepCountIs,
  type StreamTextOnFinishCallback,
  type ToolSet,
} from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { z } from 'zod';
import type { Env } from './types';

export type UsageStats = {
  inputTokens: number;
  outputTokens: number;
  requests: number;
};

export type ExecutionMode = 'direct' | 'codemode';

export class RealEstateAgent extends AIChatAgent<Env> {
  private cachedToken: string | null = null;
  private cachedFullName: string | null = null;

  private log(event: string, data?: Record<string, unknown>) {
    console.log(JSON.stringify({ ts: new Date().toISOString(), event, ...data }));
  }

  // ─── Auth ────────────────────────────────────────────────────────────────

  private async getTokenAndName(): Promise<{ token: string; fullName: string }> {
    if (this.cachedToken && this.cachedFullName) {
      this.log('auth.token_cache_hit');
      return { token: this.cachedToken, fullName: this.cachedFullName };
    }

    const [storedToken, storedName] = await Promise.all([
      this.ctx.storage.get<string>('jwt-token'),
      this.ctx.storage.get<string>('seller-full-name'),
    ]);

    if (storedToken && storedName) {
      this.log('auth.token_storage_hit');
      this.cachedToken = storedToken;
      this.cachedFullName = storedName;
      return { token: storedToken, fullName: storedName };
    }

    this.log('auth.login_start', {
      email: this.env.SELLER_EMAIL ?? '(SELLER_EMAIL secret not set)',
      hasPassword: !!this.env.SELLER_PASSWORD,
    });
    const resp = await fetch(`${this.env.LISTINGS_API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: this.env.SELLER_EMAIL,
        password: this.env.SELLER_PASSWORD,
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      this.log('auth.login_failed', { status: resp.status, body: text });
      throw new Error(`Auth failed ${resp.status}: ${text}`);
    }

    const { token, user } = (await resp.json()) as {
      token: string;
      user: { fullName: string };
    };

    this.cachedToken = token;
    this.cachedFullName = user.fullName;
    await Promise.all([
      this.ctx.storage.put('jwt-token', token),
      this.ctx.storage.put('seller-full-name', user.fullName),
    ]);
    this.log('auth.login_success', { fullName: user.fullName });
    return { token, fullName: user.fullName };
  }

  private async apiFetch(
    token: string,
    path: string,
    options: { method?: string; body?: unknown } = {}
  ): Promise<unknown> {
    const method = options.method ?? 'GET';
    this.log('api.request', { method, path });

    const resp = await fetch(`${this.env.LISTINGS_API_URL}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    if (!resp.ok) {
      const text = await resp.text();
      this.log('api.error', { method, path, status: resp.status, body: text });
      throw new Error(`API ${method} ${path} failed ${resp.status}: ${text}`);
    }

    this.log('api.response', { method, path, status: resp.status });
    return resp.json();
  }

  // ─── Mode ─────────────────────────────────────────────────────────────────

  isCodeModeAvailable(): boolean {
    return typeof this.env.CODE_EXECUTOR !== 'undefined' && this.env.CODE_EXECUTOR !== null;
  }

  async getMode(): Promise<ExecutionMode | null> {
    const val = await this.ctx.storage.get<ExecutionMode>('execution-mode');
    return val ?? null;
  }

  async setMode(mode: ExecutionMode): Promise<void> {
    await this.ctx.storage.put('execution-mode', mode);
    this.log('mode.set', { mode });
  }

  // ─── Code Mode execution ──────────────────────────────────────────────────

  private async executeCode(code: string, jwtToken: string): Promise<unknown> {
    if (!this.isCodeModeAvailable()) {
      throw new Error(
        'Code Mode is not available. A Cloudflare Workers paid plan is required. ' +
        'Upgrade at https://dash.cloudflare.com'
      );
    }

    const apiBaseUrl = this.env.LISTINGS_API_URL;
    const executionId = `exec-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;

    this.log('codemode.execute_start', { executionId, path: 'DYNAMIC_WORKER', binding: 'CODE_EXECUTOR' });

    // Build the Dynamic Worker module. JWT and apiBaseUrl are injected as
    // template literals here — the LLM never sees them.
    const workerCode = `
export default {
  async fetch(req, env, ctx) {
    try {
      const apiFetch = async (path, options = {}) => {
        const resp = await fetch(\`${apiBaseUrl}\${path}\`, {
          method: options.method ?? 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ${jwtToken}',
          },
          body: options.body ? JSON.stringify(options.body) : undefined,
        });
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(\`API \${options.method ?? 'GET'} \${path} failed \${resp.status}: \${text}\`);
        }
        return resp.json();
      };

      const userFn = ${code};
      const result = await userFn(apiFetch);
      return new Response(JSON.stringify({ success: true, result }), {
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (err) {
      return new Response(JSON.stringify({ success: false, error: err.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  }
};
`;

    this.log('codemode.dynamic_worker_instantiate', { executionId, generatedCode: workerCode });
    const worker = this.env.CODE_EXECUTOR.get(executionId, async () => ({
      compatibilityDate: '2025-06-01',
      compatibilityFlags: ['nodejs_compat'],
      mainModule: 'executor.js',
      modules: { 'executor.js': workerCode },
    }));

    const response = await worker.getEntrypoint().fetch('http://localhost/run');
    const data = (await response.json()) as { success: boolean; result?: unknown; error?: string };

    if (!data.success) {
      this.log('codemode.execute_error', { executionId, error: data.error });
      throw new Error(data.error ?? 'Code execution failed');
    }

    this.log('codemode.execute_success', { executionId });
    return data.result;
  }

  // ─── Usage ────────────────────────────────────────────────────────────────

  async getUsage(): Promise<UsageStats> {
    return (
      (await this.ctx.storage.get<UsageStats>('usage')) ?? {
        inputTokens: 0,
        outputTokens: 0,
        requests: 0,
      }
    );
  }

  async resetUsage(): Promise<void> {
    await Promise.all([
      this.ctx.storage.delete('usage'),
      this.ctx.storage.delete('jwt-token'),
      this.ctx.storage.delete('seller-full-name'),
      this.ctx.storage.delete('execution-mode'),
    ]);
    this.cachedToken = null;
    this.cachedFullName = null;
    this.log('state.reset');
  }

  // ─── Chat ─────────────────────────────────────────────────────────────────

  async onChatMessage(onFinish: StreamTextOnFinishCallback<ToolSet>) {
    const anthropic = createAnthropic({ apiKey: this.env.ANTHROPIC_API_KEY });
    const model = anthropic('claude-sonnet-4-5');

    const [{ token, fullName }, mode] = await Promise.all([
      this.getTokenAndName(),
      this.getMode(),
    ]);

    this.log('chat.message_received', {
      messageCount: this.messages.length,
      sellerName: fullName,
      mode: mode ?? 'unset',
    });

    const codeModeAvailable = this.isCodeModeAvailable();

    // ── Direct tools ──────────────────────────────────────────────────────

    const directTools = {
      listMyListings: tool({
        description: "Get all listings belonging to the authenticated seller's account only.",
        inputSchema: z.object({}),
        execute: async (_args) => {
          this.log('direct.listMyListings', { path: 'DIRECT_TOOL_CALLS' });
          const data = await this.apiFetch(token, '/listings/mine/all');
          return JSON.stringify(data);
        },
      }),

      listAllListings: tool({
        description: 'Get all active public listings across all sellers.',
        inputSchema: z.object({
          city: z.string().optional().describe('Filter by city'),
          minPrice: z.number().optional().describe('Min price in dollars'),
          maxPrice: z.number().optional().describe('Max price in dollars'),
          minBeds: z.number().int().optional().describe('Minimum number of beds'),
          propertyType: z
            .enum(['single_family', 'condo', 'townhouse', 'multi_family', 'land'])
            .optional(),
        }),
        execute: async (args) => {
          const params = new URLSearchParams();
          if (args.city) params.set('city', args.city);
          if (args.minPrice) params.set('minPrice', String(args.minPrice));
          if (args.maxPrice) params.set('maxPrice', String(args.maxPrice));
          if (args.minBeds) params.set('minBeds', String(args.minBeds));
          if (args.propertyType) params.set('propertyType', args.propertyType);
          const qs = params.toString();
          this.log('direct.listAllListings', { path: 'DIRECT_TOOL_CALLS', filters: args });
          const data = await this.apiFetch(token, `/listings${qs ? `?${qs}` : ''}`);
          return JSON.stringify(data);
        },
      }),

      createListing: tool({
        description: 'Create a new real estate listing.',
        inputSchema: z.object({
          title: z.string().describe('Listing title'),
          description: z.string().optional().describe('Property description'),
          price: z.number().int().positive().describe('Price in cents (e.g. $450,000 = 45000000)'),
          addressLine1: z.string().describe('Street address'),
          addressLine2: z.string().optional(),
          city: z.string(),
          state: z.string().length(2).describe('2-letter state abbreviation'),
          zip: z.string().min(5).describe('ZIP code'),
          beds: z.number().int().min(0),
          baths: z.number().multipleOf(0.5).min(0).describe('Baths in 0.5 increments'),
          sqft: z.number().int().positive().optional(),
          lotSqft: z.number().int().positive().optional(),
          yearBuilt: z.number().int().min(1800).optional(),
          propertyType: z
            .enum(['single_family', 'condo', 'townhouse', 'multi_family', 'land'])
            .default('single_family'),
          garageSpaces: z.number().int().min(0).default(0),
          hasPool: z.boolean().default(false),
          hoaFee: z.number().int().min(0).default(0).describe('HOA fee in cents per month'),
          status: z.enum(['draft', 'active']).default('draft'),
        }),
        execute: async (input: Record<string, unknown>) => {
          this.log('direct.createListing', { path: 'DIRECT_TOOL_CALLS', title: input.title, price: input.price });
          const data = await this.apiFetch(token, '/listings', { method: 'POST', body: input });
          this.log('tool.createListing.success', { result: data });
          return JSON.stringify(data);
        },
      }),
    };

    // ── Code mode tool ────────────────────────────────────────────────────

    const codeTool = {
      execute: tool({
        description:
          'Execute a JavaScript async function to interact with the listings API. ' +
          'Use for any operation: create, read, search, update, delete. ' +
          'Batching multiple operations in one call saves tokens.',
        inputSchema: z.object({
          reasoning: z.string().describe('Brief explanation of what the code does'),
          code: z
            .string()
            .describe(
              'An async arrow function: async (apiFetch) => { ... }. ' +
              'Call apiFetch(path, { method, body }) for all API operations. Return the result.'
            ),
        }),
        execute: async ({ reasoning, code }) => {
          this.log('codemode.tool_called', { path: 'CODE_MODE → DYNAMIC_WORKER', reasoning });
          try {
            const result = await this.executeCode(code, token);
            return JSON.stringify({ success: true, result });
          } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err);
            return JSON.stringify({ success: false, error: msg });
          }
        },
      }),
    };

    // ── setExecutionMode tool (only when mode is unset) ───────────────────

    const setModeTool = {
      setExecutionMode: tool({
        description: 'Set the execution mode for this session.',
        inputSchema: z.object({
          mode: z.enum(['direct', 'codemode']).describe(
            '"direct" = standard tool calls, "codemode" = Dynamic Worker code execution'
          ),
        }),
        execute: async ({ mode: selectedMode }) => {
          if (selectedMode === 'codemode' && !codeModeAvailable) {
            return JSON.stringify({
              success: false,
              error:
                'Code Mode is not available. A Cloudflare Workers paid plan is required. ' +
                'Upgrade at https://dash.cloudflare.com',
            });
          }
          await this.setMode(selectedMode);
          return JSON.stringify({ success: true, mode: selectedMode });
        },
      }),
    };

    // ── Assemble tool set based on current mode ───────────────────────────

    let tools: ToolSet;
    let systemPrompt: string;

    if (mode === null) {
      // Mode not yet chosen — include everything so the LLM can ask then act in one turn
      tools = {
        ...directTools,
        ...(codeModeAvailable ? codeTool : {}),
        ...setModeTool,
      } as ToolSet;
      systemPrompt = buildModeSelectionPrompt(fullName, codeModeAvailable);
      this.log('dispatch.path', { path: 'MODE_SELECTION', availableTools: Object.keys(tools) });
    } else if (mode === 'codemode') {
      // setModeTool always included so the user can switch back to direct at any time
      tools = { ...codeTool, ...setModeTool } as ToolSet;
      systemPrompt = buildCodeModePrompt(fullName);
      this.log('dispatch.path', {
        path: 'CODE_MODE → DYNAMIC_WORKER',
        codeModeAvailable,
        availableTools: Object.keys(tools),
      });
    } else {
      // setModeTool always included so the user can switch to code mode at any time
      tools = { ...directTools, ...setModeTool } as ToolSet;
      systemPrompt = buildDirectPrompt(fullName);
      this.log('dispatch.path', { path: 'DIRECT_TOOL_CALLS', availableTools: Object.keys(tools) });
    }

    const result = streamText({
      model,
      system: systemPrompt,
      messages: convertToModelMessages(this.messages),
      tools,
      stopWhen: stepCountIs(5),
      onFinish: async (event) => {
        const usage = event.usage;
        if (usage) {
          const current = await this.getUsage();
          const updated: UsageStats = {
            inputTokens: current.inputTokens + (usage.inputTokens ?? 0),
            outputTokens: current.outputTokens + (usage.outputTokens ?? 0),
            requests: current.requests + 1,
          };
          await this.ctx.storage.put('usage', updated);
          this.log('chat.finished', {
            mode: mode ?? 'unset',
            inputTokens: usage.inputTokens,
            outputTokens: usage.outputTokens,
            totalInputTokens: updated.inputTokens,
            totalOutputTokens: updated.outputTokens,
            totalRequests: updated.requests,
          });
        }
        (onFinish as unknown as StreamTextOnFinishCallback<typeof tools>)(event as any);
      },
    });

    return result.toUIMessageStreamResponse();
  }
}

// ─── System prompts ─────────────────────────────────────────────────────────

const DATA_RULES = `## Data rules
- price and hoaFee are always in CENTS. $450,000 → 45000000
- baths must be a multiple of 0.5 (e.g. 1, 1.5, 2, 2.5)
- state must be a 2-letter abbreviation (e.g. "TX", "CA")
- zip must be at least 5 characters
- status: "active" = public, "draft" = private`;

const LISTING_CLARIFICATIONS = `## Clarifying question rules
**Viewing listings:** If the user hasn't specified whose listings, ask:
  "Would you like to see all listings across the platform, or just yours?"
**Creating a listing:** If the user hasn't specified public or private, ask:
  "Should this listing be public (active) or private (draft)?"
Skip the question if they already specified. Wait for their answer before calling any tool.`;

function buildModeSelectionPrompt(sellerName: string, codeModeAvailable: boolean): string {
  const modeOptions = codeModeAvailable
    ? `- **Direct tool calls** — standard approach, one tool call per operation
- **Code Mode** — LLM writes code executed in a Dynamic Worker sandbox; fewer tokens for multi-step tasks`
    : `- **Direct tool calls** — the only available option (Code Mode requires a paid Cloudflare Workers plan)`;

  return `You are a helpful real estate assistant for ${sellerName}.

## Execution mode — NOT YET SET
Before performing any listing operation, you MUST ask the user which execution mode to use:
  "How would you like me to execute this? Options:
${modeOptions}"

If the user's message already specifies a mode (e.g. "use code mode", "use direct tools"), call
setExecutionMode immediately with that choice, then proceed with the operation in the same response.
Otherwise ask the question and wait.

Once setExecutionMode is called, use the matching tools for the rest of the response.

${LISTING_CLARIFICATIONS}

${DATA_RULES}

When creating a listing, confirm back with the listing ID. Keep responses concise.`;
}

function buildDirectPrompt(sellerName: string): string {
  return `You are a helpful real estate assistant for ${sellerName}.

## Execution mode: Direct tool calls
Use listMyListings, listAllListings, and createListing to fulfil requests.
To switch to Code Mode, call setExecutionMode({ mode: "codemode" }) then proceed.

${LISTING_CLARIFICATIONS}

${DATA_RULES}

When creating a listing, confirm back with the listing ID. Keep responses concise.`;
}

function buildCodeModePrompt(sellerName: string): string {
  return `You are a helpful real estate assistant for ${sellerName}.

## Execution mode: Code Mode (Dynamic Workers)
Use the execute tool for ALL operations. Write an async arrow function that calls apiFetch.
To switch back to Direct tool calls, call setExecutionMode({ mode: "direct" }) then proceed.

### API reference
- GET  /listings/mine/all          — seller's listings
- GET  /listings?city=&minPrice=&maxPrice=&minBeds=&propertyType=  — public search
- GET  /listings/:id               — single listing
- POST /listings                   — create listing (body below)
- PUT  /listings/:id               — update listing
- DELETE /listings/:id             — delete listing

### Create/update body
title (req), description, price (CENTS), addressLine1 (req), addressLine2,
city (req), state (2-char req), zip (5+ req), beds (int req), baths (0.5 multiples req),
sqft, lotSqft, yearBuilt, propertyType (single_family|condo|townhouse|multi_family|land),
garageSpaces (int, default 0), hasPool (bool), hoaFee (CENTS/mo), status (active|draft)

### Code signature
async (apiFetch) => {
  // apiFetch(path, { method?, body? }) returns parsed JSON
  return await apiFetch('/listings/mine/all');
}

### Key rules
- price and hoaFee in CENTS ($450k = 45000000)
- Batch related operations in ONE execute call to save tokens
- status: "active" = public, "draft" = private
- Always return a useful value from the function

${LISTING_CLARIFICATIONS}

When creating a listing, confirm back with the listing ID. Keep responses concise.`;
}
