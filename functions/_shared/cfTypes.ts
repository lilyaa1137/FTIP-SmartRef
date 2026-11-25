import { sanitizeDiagnosticDetail } from '../../src/utils/academicCore.ts';

export interface CfPagesEnv {
  GEMINI_API_KEY?: string;
}

export interface CfPagesContext<Env = CfPagesEnv> {
  request: Request;
  env: Env;
  params: Record<string, string | string[]>;
}

export type CfPagesHandler<Env = CfPagesEnv> = (
  context: CfPagesContext<Env>
) => Promise<Response> | Response;

export function resolveApiKey(env: CfPagesEnv | undefined): string | null {
  const raw = typeof env?.GEMINI_API_KEY === 'string' ? env.GEMINI_API_KEY : '';
  const cleaned = raw.trim().replace(/^["']|["']$/g, '').trim();
  if (!cleaned || cleaned === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return cleaned;
}

export function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
    },
  });
}

export async function readJsonBody<T = Record<string, unknown>>(request: Request): Promise<T> {
  const text = await request.text();
  if (!text.trim()) {
    throw new Error('EMPTY_BODY');
  }
  return JSON.parse(text) as T;
}

export interface GeminiRestPart {
  text?: string;
  thoughtSignature?: string;
  functionCall?: {
    id?: string;
    name: string;
    args?: Record<string, unknown>;
  };
  functionResponse?: {
    id?: string;
    name: string;
    response: Record<string, unknown>;
  };
}

export interface GeminiRestContent {
  role: 'user' | 'model';
  parts: GeminiRestPart[];
}

export interface GeminiRestErrorDetail {
  '@type'?: string;
  reason?: string;
  domain?: string;
  metadata?: Record<string, string>;
}

export interface GeminiRestResponse {
  candidates?: Array<{
    content?: GeminiRestContent;
  }>;
  error?: {
    code?: number;
    message?: string;
    status?: string;
    details?: GeminiRestErrorDetail[];
  };
}

export class GeminiUpstreamError extends Error {
  status?: number;
  upstreamHttpStatus?: number;
  upstreamCode?: number;
  grpcStatus?: string;
  reason?: string;
  diagnosticMessage?: string;
  isTimeout?: boolean;
  isNetworkError?: boolean;
  isNonJson?: boolean;

  constructor(
    message: string,
    opts: {
      status?: number;
      upstreamHttpStatus?: number;
      upstreamCode?: number;
      grpcStatus?: string;
      reason?: string;
      diagnosticMessage?: string;
      isTimeout?: boolean;
      isNetworkError?: boolean;
      isNonJson?: boolean;
    } = {}
  ) {
    super(message);
    this.name = 'GeminiUpstreamError';
    this.status = opts.status;
    this.upstreamHttpStatus = opts.upstreamHttpStatus;
    this.upstreamCode = opts.upstreamCode;
    this.grpcStatus = opts.grpcStatus;
    this.reason = opts.reason;
    this.diagnosticMessage = opts.diagnosticMessage;
    this.isTimeout = opts.isTimeout;
    this.isNetworkError = opts.isNetworkError;
    this.isNonJson = opts.isNonJson;
  }
}

function computeRetryDelayMs(attempt: number): number {
  const baseDelayMs = 900 * Math.pow(2, attempt - 1); // attempt 1: 900ms, attempt 2: 1800ms
  const jitterMs = Math.floor(Math.random() * 250);
  return baseDelayMs + jitterMs;
}

export async function callGeminiRest(
  apiKey: string,
  model: string,
  payload: Record<string, unknown>,
  options: { timeoutMs?: number } = {}
): Promise<GeminiRestResponse> {
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
    model
  )}:generateContent`;
  const maxAttempts = 3;
  const perAttemptTimeoutMs = options.timeoutMs ?? 20000;
  const totalBudgetMs = Math.min(Math.max(perAttemptTimeoutMs + 5000, perAttemptTimeoutMs), 25000);
  const deadlineAt = Date.now() + totalBudgetMs;

  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const remainingBudgetMs = deadlineAt - Date.now();
    if (remainingBudgetMs <= 0) {
      break;
    }

    const attemptTimeoutMs = Math.min(perAttemptTimeoutMs, remainingBudgetMs);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), attemptTimeoutMs);

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const ct = (res.headers.get('content-type') || '').toLowerCase();
      if (!ct.includes('application/json')) {
        const rawText = await res.text().catch(() => '');
        const stripped = rawText.replace(/<[^>]+>/g, ' ');
        const detail = sanitizeDiagnosticDetail(stripped || `Non-JSON response (HTTP ${res.status})`, 160);
        throw new GeminiUpstreamError(`Gemini HTTP ${res.status}: ${detail}`, {
          status: res.status,
          upstreamHttpStatus: res.status,
          diagnosticMessage: detail,
          isNonJson: true,
        });
      }

      let data: GeminiRestResponse;
      try {
        data = (await res.json()) as GeminiRestResponse;
      } catch {
        throw new GeminiUpstreamError(`Invalid JSON from Gemini HTTP ${res.status}`, {
          status: res.status,
          upstreamHttpStatus: res.status,
          diagnosticMessage: `Malformed JSON body (HTTP ${res.status})`,
          isNonJson: true,
        });
      }

      if (!res.ok || data.error) {
        const upstreamCode = typeof data.error?.code === 'number' ? data.error.code : undefined;
        const effectiveStatus = upstreamCode || res.status;
        const grpcStatus =
          typeof data.error?.status === 'string' && data.error.status.trim()
            ? data.error.status.trim()
            : undefined;
        const reason = Array.isArray(data.error?.details)
          ? data.error.details.find((d) => typeof d?.reason === 'string' && d.reason.trim())?.reason?.trim()
          : undefined;
        const diagnosticMessage = sanitizeDiagnosticDetail(
          data.error?.message || `Gemini HTTP ${res.status}`,
          180
        );

        throw new GeminiUpstreamError(
          `${grpcStatus ? `${grpcStatus} ` : ''}${diagnosticMessage}`.trim(),
          {
            status: effectiveStatus,
            upstreamHttpStatus: res.status,
            upstreamCode,
            grpcStatus,
            reason,
            diagnosticMessage,
          }
        );
      }

      return data;
    } catch (err) {
      let wrapped: GeminiUpstreamError;
      if (err instanceof GeminiUpstreamError) {
        wrapped = err;
      } else {
        const isAbort =
          controller.signal.aborted ||
          (err instanceof Error && err.name === 'AbortError');
        if (isAbort) {
          wrapped = new GeminiUpstreamError(
            `Gemini API request timed out after ${attemptTimeoutMs}ms`,
            {
              status: 504,
              isTimeout: true,
              diagnosticMessage: `Request timeout (${attemptTimeoutMs}ms)`,
            }
          );
        } else {
          const detail = sanitizeDiagnosticDetail(
            err instanceof Error ? err.message : String(err || 'Network fetch failed'),
            180
          );
          wrapped = new GeminiUpstreamError(detail || 'Network fetch failed', {
            isNetworkError: true,
            diagnosticMessage: detail || 'Network fetch failed',
          });
        }
      }

      lastError = wrapped;
      const grpcUpper = (wrapped.grpcStatus || '').toUpperCase();
      const isTransientStatus =
        wrapped.status === 500 ||
        wrapped.status === 503 ||
        wrapped.upstreamHttpStatus === 500 ||
        wrapped.upstreamHttpStatus === 503 ||
        grpcUpper === 'INTERNAL' ||
        grpcUpper === 'UNAVAILABLE';
      const isRetryable =
        isTransientStatus || wrapped.isTimeout === true || wrapped.isNetworkError === true;

      if (attempt < maxAttempts && isRetryable) {
        const delayMs = computeRetryDelayMs(attempt);
        const remainingAfterFailure = deadlineAt - Date.now();
        const minHeadroomMs = wrapped.isTimeout ? 2500 : 1000;
        if (remainingAfterFailure > delayMs + minHeadroomMs) {
          await new Promise((r) => setTimeout(r, delayMs));
          continue;
        }
      }
      throw wrapped;
    } finally {
      clearTimeout(timer);
    }
  }

  throw lastError;
}
