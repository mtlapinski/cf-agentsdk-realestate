export type Env = {
  // Static assets binding
  ASSETS: Fetcher;

  // Durable Object namespace
  REAL_ESTATE_AGENT: DurableObjectNamespace;

  // Dynamic Workers binding (WorkerLoader) — used in M3
  CODE_EXECUTOR: any;

  // cf-playground API base URL (set in wrangler.jsonc vars)
  LISTINGS_API_URL: string;

  // Seller credentials (stored as secrets in .dev.vars)
  SELLER_EMAIL: string;
  SELLER_PASSWORD: string;

  // AI provider
  ANTHROPIC_API_KEY: string;
};
