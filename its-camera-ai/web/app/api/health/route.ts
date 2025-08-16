import { NextRequest, NextResponse } from 'next/server';

/**
 * Health check endpoint for the ITS Camera AI frontend
 * Used by Docker health checks and load balancers
 */

interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version: string;
  environment: string;
  uptime: number;
  checks: {
    server: 'ok' | 'error';
    memory: 'ok' | 'warning' | 'error';
    api: 'ok' | 'unreachable' | 'error';
  };
  details?: {
    memoryUsage?: NodeJS.MemoryUsage;
    processId?: number;
    nodeVersion?: string;
    platform?: string;
    error?: string;
  };
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const startTime = Date.now();

  try {
    // Get basic system info
    const memoryUsage = process.memoryUsage();
    const uptime = process.uptime();

    // Check memory usage (warn if heap used > 80%, error if > 90%)
    const heapUsedMB = memoryUsage.heapUsed / 1024 / 1024;
    const heapTotalMB = memoryUsage.heapTotal / 1024 / 1024;
    const memoryUsagePercent = (heapUsedMB / heapTotalMB) * 100;

    let memoryStatus: 'ok' | 'warning' | 'error' = 'ok';
    if (memoryUsagePercent > 90) {
      memoryStatus = 'error';
    } else if (memoryUsagePercent > 80) {
      memoryStatus = 'warning';
    }

    // Check API connectivity (optional - can be disabled in production)
    let apiStatus: 'ok' | 'unreachable' | 'error' = 'ok';
    const checkApiConnection = process.env.HEALTH_CHECK_API !== 'false';

    if (checkApiConnection) {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_API_URL;
        if (apiUrl) {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout

          const response = await fetch(`${apiUrl}/health`, {
            signal: controller.signal,
            headers: {
              'User-Agent': 'ITS-Camera-AI-Frontend-HealthCheck',
            },
          });

          clearTimeout(timeoutId);

          if (!response.ok) {
            apiStatus = 'error';
          }
        }
      } catch (error) {
        apiStatus = 'unreachable';
      }
    }

    // Determine overall status
    const hasErrors = memoryStatus === 'error' || apiStatus === 'error';
    const hasWarnings = memoryStatus === 'warning';

    const status: HealthStatus = {
      status: hasErrors ? 'unhealthy' : 'healthy',
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      uptime: Math.round(uptime),
      checks: {
        server: 'ok',
        memory: memoryStatus,
        api: apiStatus,
      },
      details: {
        memoryUsage: {
          rss: Math.round(memoryUsage.rss / 1024 / 1024), // MB
          heapTotal: Math.round(heapTotalMB), // MB
          heapUsed: Math.round(heapUsedMB), // MB
          external: Math.round(memoryUsage.external / 1024 / 1024), // MB
        },
        processId: process.pid,
        nodeVersion: process.version,
        platform: process.platform,
      },
    };

    // Set appropriate HTTP status code
    const httpStatus = status.status === 'healthy' ? 200 : 503;

    // Add response time header
    const responseTime = Date.now() - startTime;

    const response = NextResponse.json(status, { status: httpStatus });
    response.headers.set('X-Response-Time', `${responseTime}ms`);
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');
    response.headers.set('Pragma', 'no-cache');
    response.headers.set('Expires', '0');

    return response;

  } catch (error) {
    console.error('Health check error:', error);

    const errorStatus: HealthStatus = {
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      uptime: 0,
      checks: {
        server: 'error',
        memory: 'error',
        api: 'error',
      },
      details: {
        error: error instanceof Error ? error.message : 'Unknown error',
      },
    };

    const response = NextResponse.json(errorStatus, { status: 503 });
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');

    return response;
  }
}

// HEAD request for simple health checks
export async function HEAD(): Promise<NextResponse> {
  try {
    const response = new NextResponse(null, { status: 200 });
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');
    return response;
  } catch {
    return new NextResponse(null, { status: 503 });
  }
}