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

export class RealEstateAgent extends AIChatAgent<Env> {
  private cachedToken: string | null = null;
  private cachedFullName: string | null = null;

  private log(event: string, data?: Record<string, unknown>) {
    console.log(JSON.stringify({ ts: new Date().toISOString(), event, ...data }));
  }

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

    this.log('auth.login_start', { email: this.env.SELLER_EMAIL });
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
    ]);
    this.cachedToken = null;
    this.cachedFullName = null;
    this.log('state.reset');
  }

  async onChatMessage(onFinish: StreamTextOnFinishCallback<ToolSet>) {
    const anthropic = createAnthropic({ apiKey: this.env.ANTHROPIC_API_KEY });
    const model = anthropic('claude-sonnet-4-5');

    const { token, fullName } = await this.getTokenAndName();
    const msgCount = this.messages.length;
    this.log('chat.message_received', { messageCount: msgCount, sellerName: fullName });

    const tools = {
      listMyListings: tool({
        description: "Get all listings belonging to the authenticated seller's account only.",
        inputSchema: z.object({}),
        execute: async (_args) => {
          this.log('tool.listMyListings');
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
          this.log('tool.listAllListings', { filters: args });
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
          this.log('tool.createListing', { title: input.title, price: input.price });
          const data = await this.apiFetch(token, '/listings', {
            method: 'POST',
            body: input,
          });
          this.log('tool.createListing.success', { result: data });
          return JSON.stringify(data);
        },
      }),
    };

    const result = streamText({
      model,
      system: buildSystemPrompt(fullName),
      messages: convertToModelMessages(this.messages),
      tools,
      stopWhen: stepCountIs(4),
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

function buildSystemPrompt(sellerEmail: string): string {
  return `You are a helpful real estate assistant that manages property listings on behalf of a seller.

## Tools available
- listMyListings: listings for the authenticated seller (${sellerEmail}) only
- listAllListings: all active public listings across every seller, with optional filters
- createListing: create a new listing

## Clarifying question rule
When the user asks to show, list, or view listings — and they have NOT specified whose listings —
you MUST ask this clarifying question before calling any tool:
  "Would you like to see all listings across the platform, or just the listings for ${sellerEmail}?"
Wait for their answer before proceeding. If they have already specified (e.g. "show MY listings"
or "show ALL listings"), skip the question and call the appropriate tool directly.

## Important data rules
- price and hoaFee are always in CENTS. Convert dollars: $450,000 → 45000000
- baths must be a multiple of 0.5 (e.g. 1, 1.5, 2, 2.5)
- state must be a 2-letter abbreviation (e.g. "TX", "CA")
- zip must be at least 5 characters
- status defaults to "draft" unless the user says to publish/activate it

When creating a listing, confirm back with the listing ID returned.
Keep responses concise.`;
}
