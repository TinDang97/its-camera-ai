import { SignJWT, jwtVerify } from 'jose';
import { env, securityConfig } from './env';

// Security utilities
export class SecurityManager {
  private static instance: SecurityManager;
  private readonly jwtSecret: Uint8Array;

  constructor() {
    // Use a default secret for development, but require proper configuration in production
    const secret = securityConfig.jwtSecret || 'development-jwt-secret-change-in-production';
    this.jwtSecret = new TextEncoder().encode(secret);
  }

  static getInstance(): SecurityManager {
    if (!SecurityManager.instance) {
      SecurityManager.instance = new SecurityManager();
    }
    return SecurityManager.instance;
  }

  // JWT token management
  async createToken(payload: Record<string, any>, expiresIn: string = '24h'): Promise<string> {
    const token = await new SignJWT(payload)
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime(expiresIn)
      .sign(this.jwtSecret);

    return token;
  }

  async verifyToken(token: string): Promise<any> {
    try {
      const { payload } = await jwtVerify(token, this.jwtSecret);
      return payload;
    } catch (error) {
      throw new Error('Invalid token');
    }
  }

  // CSRF token generation
  generateCSRFToken(): string {
    if (typeof window !== 'undefined' && window.crypto) {
      const array = new Uint32Array(8);
      window.crypto.getRandomValues(array);
      return Array.from(array, dec => dec.toString(16)).join('');
    }
    // Fallback for server-side or unsupported environments
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  // Content Security Policy
  generateNonce(): string {
    if (typeof window !== 'undefined' && window.crypto) {
      const array = new Uint8Array(16);
      window.crypto.getRandomValues(array);
      return btoa(String.fromCharCode(...array));
    }
    // Fallback
    return btoa(Math.random().toString()).substring(0, 16);
  }

  // Rate limiting helper
  generateRateLimitKey(ip: string, endpoint: string): string {
    return `rate_limit:${ip}:${endpoint}`;
  }

  // Input sanitization
  sanitizeInput(input: string): string {
    return input
      .replace(/[<>]/g, '') // Remove potential XSS vectors
      .replace(/javascript:/gi, '') // Remove javascript: protocol
      .trim();
  }

  // Validate URL to prevent open redirects
  isValidRedirectUrl(url: string, allowedDomains: string[] = []): boolean {
    try {
      const urlObj = new URL(url);

      // Allow relative URLs
      if (url.startsWith('/') && !url.startsWith('//')) {
        return true;
      }

      // Check against allowed domains
      if (allowedDomains.length > 0) {
        return allowedDomains.includes(urlObj.hostname);
      }

      // By default, only allow same origin
      if (typeof window !== 'undefined') {
        return urlObj.origin === window.location.origin;
      }

      return false;
    } catch {
      return false;
    }
  }

  // Password strength validation
  validatePasswordStrength(password: string): {
    isValid: boolean;
    score: number;
    feedback: string[];
  } {
    const feedback: string[] = [];
    let score = 0;

    if (password.length < 8) {
      feedback.push('Password must be at least 8 characters long');
    } else {
      score += 1;
    }

    if (!/[a-z]/.test(password)) {
      feedback.push('Password must contain lowercase letters');
    } else {
      score += 1;
    }

    if (!/[A-Z]/.test(password)) {
      feedback.push('Password must contain uppercase letters');
    } else {
      score += 1;
    }

    if (!/\d/.test(password)) {
      feedback.push('Password must contain numbers');
    } else {
      score += 1;
    }

    if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
      feedback.push('Password must contain special characters');
    } else {
      score += 1;
    }

    return {
      isValid: score >= 4,
      score,
      feedback
    };
  }
}

// Rate limiting store (in production, use Redis or similar)
class RateLimitStore {
  private store = new Map<string, { count: number; resetTime: number }>();

  async increment(key: string, windowMs: number, maxRequests: number): Promise<{
    count: number;
    remaining: number;
    resetTime: number;
    exceeded: boolean;
  }> {
    const now = Date.now();
    const record = this.store.get(key);

    if (!record || now > record.resetTime) {
      // Reset window
      const resetTime = now + windowMs;
      this.store.set(key, { count: 1, resetTime });
      return {
        count: 1,
        remaining: maxRequests - 1,
        resetTime,
        exceeded: false,
      };
    }

    record.count++;
    this.store.set(key, record);

    return {
      count: record.count,
      remaining: Math.max(0, maxRequests - record.count),
      resetTime: record.resetTime,
      exceeded: record.count > maxRequests,
    };
  }

  async reset(key: string): Promise<void> {
    this.store.delete(key);
  }

  // Cleanup expired entries
  cleanup(): void {
    const now = Date.now();
    for (const [key, record] of this.store.entries()) {
      if (now > record.resetTime) {
        this.store.delete(key);
      }
    }
  }
}

export const rateLimitStore = new RateLimitStore();

// Cleanup expired rate limit entries every 5 minutes
if (typeof window !== 'undefined') {
  setInterval(() => {
    rateLimitStore.cleanup();
  }, 5 * 60 * 1000);
}

// Security middleware helper for API routes
export function createSecurityMiddleware(options: {
  rateLimit?: { windowMs: number; maxRequests: number };
  csrfProtection?: boolean;
  allowedOrigins?: string[];
} = {}) {
  return async function securityMiddleware(
    req: any,
    res: any,
    next: () => void
  ) {
    const security = SecurityManager.getInstance();

    // CORS handling
    if (options.allowedOrigins && options.allowedOrigins.length > 0) {
      const origin = req.headers.origin;
      if (origin && options.allowedOrigins.includes(origin)) {
        res.setHeader('Access-Control-Allow-Origin', origin);
      }
    }

    // Rate limiting
    if (options.rateLimit) {
      const clientIP = req.headers['x-forwarded-for'] || req.connection?.remoteAddress || 'unknown';
      const rateLimitKey = security.generateRateLimitKey(clientIP, req.url);
      const rateLimit = await rateLimitStore.increment(
        rateLimitKey,
        options.rateLimit.windowMs,
        options.rateLimit.maxRequests
      );

      if (rateLimit.exceeded) {
        res.status(429).json({
          error: 'Too Many Requests',
          retryAfter: Math.ceil((rateLimit.resetTime - Date.now()) / 1000),
        });
        return;
      }

      // Add rate limit headers
      res.setHeader('X-RateLimit-Limit', options.rateLimit.maxRequests.toString());
      res.setHeader('X-RateLimit-Remaining', rateLimit.remaining.toString());
      res.setHeader('X-RateLimit-Reset', rateLimit.resetTime.toString());
    }

    // CSRF protection for state-changing methods
    if (options.csrfProtection && ['POST', 'PUT', 'DELETE', 'PATCH'].includes(req.method)) {
      const csrfToken = req.headers['x-csrf-token'];
      if (!csrfToken) {
        res.status(403).json({ error: 'CSRF token required' });
        return;
      }
      // TODO: Implement CSRF token validation logic
    }

    // Add security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

    next();
  };
}

// React hook for security utilities
export function useSecurity() {
  const security = SecurityManager.getInstance();

  return {
    generateCSRFToken: () => security.generateCSRFToken(),
    sanitizeInput: (input: string) => security.sanitizeInput(input),
    validatePassword: (password: string) => security.validatePasswordStrength(password),
    isValidRedirect: (url: string, allowedDomains?: string[]) =>
      security.isValidRedirectUrl(url, allowedDomains),
  };
}

// Content Security Policy helpers
export function generateCSP(nonce?: string): string {
  const csp = [
    "default-src 'self'",
    `script-src 'self' 'unsafe-eval' ${nonce ? `'nonce-${nonce}'` : "'unsafe-inline'"}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self'",
    "connect-src 'self'",
    "media-src 'self'",
    "object-src 'none'",
    "child-src 'self'",
    "worker-src 'self'",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "upgrade-insecure-requests",
  ];

  return csp.join('; ');
}

export const security = SecurityManager.getInstance();
