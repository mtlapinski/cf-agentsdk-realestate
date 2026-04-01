import { Hono } from 'hono';
import { routeAgentRequest } from 'agents';
import { RealEstateAgent, type ExecutionMode } from './agent';
import type { Env } from './types';

const app = new Hono<{ Bindings: Env }>();

const SESSION_ID = 'default';

function getAgentStub(env: Env): RealEstateAgent {
  const id = env.REAL_ESTATE_AGENT.idFromName(SESSION_ID);
  return env.REAL_ESTATE_AGENT.get(id) as unknown as RealEstateAgent;
}

app.get('/health', (c) => c.json({ status: 'ok', ts: Date.now() }));

app.get('/api/mode', async (c) => {
  const stub = getAgentStub(c.env);
  const [mode, available] = await Promise.all([
    stub.getMode(),
    Promise.resolve(stub.isCodeModeAvailable()),
  ]);
  return c.json({ mode: mode ?? null, codeModeAvailable: available });
});

app.post('/api/mode', async (c) => {
  const { mode } = await c.req.json<{ mode: ExecutionMode }>();
  if (mode !== 'direct' && mode !== 'codemode') {
    return c.json({ error: 'mode must be "direct" or "codemode"' }, 400);
  }
  const stub = getAgentStub(c.env);
  if (mode === 'codemode' && !stub.isCodeModeAvailable()) {
    return c.json(
      { error: 'Code Mode requires a Cloudflare Workers paid plan. Upgrade at https://dash.cloudflare.com' },
      403
    );
  }
  await stub.setMode(mode);
  return c.json({ success: true, mode });
});

app.get('/api/usage', async (c) => {
  const stub = getAgentStub(c.env);
  const usage = await stub.getUsage();
  return c.json(usage);
});

app.post('/api/clear', async (c) => {
  const stub = getAgentStub(c.env);
  await stub.resetUsage();
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
