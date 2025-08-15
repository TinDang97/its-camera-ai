import { z } from 'zod';

// Environment schema validation
const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  NEXT_PUBLIC_APP_ENV: z.enum(['development', 'staging', 'production']).default('development'),
  NEXT_PUBLIC_APP_NAME: z.string().default('ITS Camera AI Dashboard'),
  NEXT_PUBLIC_APP_VERSION: z.string().default('1.0.0'),

  // API Configuration
  NEXT_PUBLIC_API_BASE_URL: z.string().default('http://localhost:8000'),
  NEXT_PUBLIC_WS_URL: z.string().default('ws://localhost:8000/ws'),
  API_SECRET_KEY: z.string().optional(),

  // Database
  DATABASE_URL: z.string().optional(),
  REDIS_URL: z.string().optional(),

  // Authentication
  NEXTAUTH_URL: z.string().default('http://localhost:3000'),
  NEXTAUTH_SECRET: z.string().optional(),

  // Security
  ENCRYPTION_KEY: z.string().optional(),
  JWT_SECRET: z.string().optional(),

  // Monitoring
  NEXT_PUBLIC_ANALYTICS_ID: z.string().optional(),
  NEXT_PUBLIC_PERFORMANCE_MONITORING: z.string().transform((val) => val === 'true').default('true'),
  SENTRY_DSN: z.string().optional(),
});

// Parse and validate environment variables
const parseEnv = () => {
  try {
    return envSchema.parse(process.env);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const missingVars = error.errors.map(err => `${err.path.join('.')}: ${err.message}`);
      console.warn(`⚠️ Environment validation warnings:\n${missingVars.join('\n')}`);
      // Return parsed values with defaults for development
      return envSchema.parse({
        ...process.env,
        // Provide development defaults
        NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
        NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
        NEXTAUTH_URL: process.env.NEXTAUTH_URL || 'http://localhost:3000',
      });
    }
    throw error;
  }
};

export const env = parseEnv();

// Environment checks
export const isDevelopment = env.NODE_ENV === 'development';
export const isProduction = env.NODE_ENV === 'production';
export const isTest = env.NODE_ENV === 'test';

// API configuration
export const apiConfig = {
  baseUrl: env.NEXT_PUBLIC_API_BASE_URL,
  wsUrl: env.NEXT_PUBLIC_WS_URL,
  secretKey: env.API_SECRET_KEY,
  timeout: 30000, // 30 seconds
  retries: 3,
};

// Security configuration
export const securityConfig = {
  encryptionKey: env.ENCRYPTION_KEY,
  jwtSecret: env.JWT_SECRET,
  sessionTimeout: 24 * 60 * 60 * 1000, // 24 hours
  maxLoginAttempts: 5,
};

// Monitoring configuration
export const monitoringConfig = {
  analyticsId: env.NEXT_PUBLIC_ANALYTICS_ID,
  performanceMonitoring: env.NEXT_PUBLIC_PERFORMANCE_MONITORING,
  sentryDsn: env.SENTRY_DSN,
};
