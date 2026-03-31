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

export class RealEstateAgent extends AIChatAgent<Env> {
  private cachedToken: string | null = null;

  private async getToken(): Promise<string> {
    if (this.cachedToken) return this.cachedToken;

    const stored = await this.ctx.storage.get<string>('jwt-token');
    if (stored) {
      this.cachedToken = stored;
      return stored;
    }

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
      throw new Error(`Auth failed ${resp.status}: ${text}`);
    }

    const { token } = (await resp.json()) as { token: string };
    this.cachedToken = token;
    await this.ctx.storage.put('jwt-token', token);
    return token;
  }

  private async apiFetch(
    token: string,
    path: string,
    options: { method?: string; body?: unknown } = {}
  ): Promise<unknown> {
    const resp = await fetch(`${this.env.LISTINGS_API_URL}${path}`, {
      method: options.method ?? 'GET',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`API ${options.method ?? 'GET'} ${path} failed ${resp.status}: ${text}`);
    }

    return resp.json();
  }

  async onChatMessage(onFinish: StreamTextOnFinishCallback<ToolSet>) {
    const anthropic = createAnthropic({ apiKey: this.env.ANTHROPIC_API_KEY });
    const model = anthropic('claude-sonnet-4-5');

    const token = await this.getToken();

    const tools = {
      listMyListings: tool({
        description: 'Get all listings belonging to the authenticated seller.',
        inputSchema: z.object({}),
        execute: async (_args) => {
          const data = await this.apiFetch(token, '/listings/mine/all');
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
          const data = await this.apiFetch(token, '/listings', {
            method: 'POST',
            body: input,
          });
          return JSON.stringify(data);
        },
      }),
    };

    const result = streamText({
      model,
      system: SYSTEM_PROMPT,
      messages: convertToModelMessages(this.messages),
      tools,
      stopWhen: stepCountIs(4),
      onFinish: onFinish as unknown as StreamTextOnFinishCallback<typeof tools>,
    });

    return result.toUIMessageStreamResponse();
  }
}

const SYSTEM_PROMPT = `You are a helpful real estate assistant that manages property listings on behalf of a seller.

You have two tools available:
- listMyListings: fetch all of the seller's current listings
- createListing: create a new listing

## Important data rules
- price and hoaFee are always in CENTS. Convert dollars: $450,000 → 45000000
- baths must be a multiple of 0.5 (e.g. 1, 1.5, 2, 2.5)
- state must be a 2-letter abbreviation (e.g. "TX", "CA")
- zip must be at least 5 characters
- status defaults to "draft" unless the user says to publish/activate it

When creating a listing, confirm back with the listing ID returned.
Keep responses concise.`;
