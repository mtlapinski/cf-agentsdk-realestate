import { Hono } from 'hono';
import { routeAgentRequest } from 'agents';
import { RealEstateAgent } from './agent';
import type { Env } from './types';

const app = new Hono<{ Bindings: Env }>();

app.get('/health', (c) => c.json({ status: 'ok', ts: Date.now() }));

app.post('/api/clear', async (c) => {
  const id = c.env.REAL_ESTATE_AGENT.idFromName('default');
  const stub = c.env.REAL_ESTATE_AGENT.get(id) as unknown as RealEstateAgent;
  await (stub as any).clearState?.();
  return c.json({ success: true });
});

app.all('/agents/*', async (c) => {
  const resp = await routeAgentRequest(c.req.raw, c.env);
  return resp ?? c.notFound();
});

// Fall through to static assets for everything else
app.all('*', (c) => c.env.ASSETS.fetch(c.req.raw));

export default app;
export { RealEstateAgent } from './agent';
