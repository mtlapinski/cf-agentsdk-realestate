import { Hono } from 'hono';
import type { Env } from './types';

const app = new Hono<{ Bindings: Env }>();

app.get('/health', (c) => c.json({ status: 'ok', ts: Date.now() }));

export default app;

// Named exports for Durable Object and WorkerEntrypoint classes.
// RealEstateAgent will be added in M2.
