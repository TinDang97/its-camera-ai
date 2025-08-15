'use client';

import { env, monitoringConfig } from './env';

// Performance metrics collection
export class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();
  private observers: PerformanceObserver[] = [];

  init() {
    if (typeof window === 'undefined') return;

    // Web Vitals monitoring
    this.initWebVitals();

    // Performance observer for navigation timing
    this.initNavigationTiming();

    // Resource timing observer
    this.initResourceTiming();

    // Long tasks observer
    this.initLongTasksObserver();

    console.log('📊 Performance monitoring initialized');
  }

  private initWebVitals() {
    // Core Web Vitals
    if (typeof window !== 'undefined' && 'web-vitals' in window) {
      import('web-vitals').then(({ onCLS, onFID, onFCP, onLCP, onTTFB }) => {
        onCLS(this.sendMetric.bind(this));
        onFID(this.sendMetric.bind(this));
        onFCP(this.sendMetric.bind(this));
        onLCP(this.sendMetric.bind(this));
        onTTFB(this.sendMetric.bind(this));
      }).catch(() => {
        // web-vitals not available
      });
    }
  }

  private initNavigationTiming() {
    if (!('PerformanceObserver' in window)) return;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            this.recordMetric('navigation.domContentLoaded', navEntry.domContentLoadedEventEnd - navEntry.domContentLoadedEventStart);
            this.recordMetric('navigation.load', navEntry.loadEventEnd - navEntry.loadEventStart);
          }
        }
      });

      observer.observe({ entryTypes: ['navigation'] });
      this.observers.push(observer);
    } catch (error) {
      console.warn('Navigation timing observer not supported');
    }
  }

  private initResourceTiming() {
    if (!('PerformanceObserver' in window)) return;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const resourceEntry = entry as PerformanceResourceTiming;

          // Track resource load times by type
          const resourceType = this.getResourceType(resourceEntry.name);
          this.recordMetric(`resource.${resourceType}`, resourceEntry.duration);

          // Track failed resources
          if (resourceEntry.transferSize === 0 && resourceEntry.decodedBodySize > 0) {
            this.recordMetric('resource.cached', 1);
          }
        }
      });

      observer.observe({ entryTypes: ['resource'] });
      this.observers.push(observer);
    } catch (error) {
      console.warn('Resource timing observer not supported');
    }
  }

  private initLongTasksObserver() {
    if (!('PerformanceObserver' in window)) return;

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.recordMetric('longtask.duration', entry.duration);
          this.sendAlert('Long task detected', { duration: entry.duration });
        }
      });

      observer.observe({ entryTypes: ['longtask'] });
      this.observers.push(observer);
    } catch (error) {
      // Long tasks observer not supported in all browsers
    }
  }

  private getResourceType(url: string): string {
    if (url.includes('.css')) return 'css';
    if (url.includes('.js')) return 'js';
    if (/\.(png|jpg|jpeg|gif|webp|avif|svg)/.test(url)) return 'image';
    if (/\.(woff|woff2|ttf|otf)/.test(url)) return 'font';
    return 'other';
  }

  private recordMetric(name: string, value: number) {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(value);
  }

  private sendMetric(metric: any) {
    if (monitoringConfig.analyticsId) {
      // Send to analytics service
      this.sendToAnalytics({
        name: metric.name,
        value: metric.value,
        id: metric.id,
        timestamp: Date.now(),
      });
    }

    // Log performance issues
    if (this.isPerformanceIssue(metric)) {
      console.warn(`⚠️ Performance issue detected:`, metric);
      this.sendAlert('Performance threshold exceeded', metric);
    }
  }

  private isPerformanceIssue(metric: any): boolean {
    const thresholds = {
      CLS: 0.1,
      FID: 100,
      LCP: 2500,
      FCP: 1800,
      TTFB: 600,
    };

    return metric.value > (thresholds[metric.name as keyof typeof thresholds] || Infinity);
  }

  private async sendToAnalytics(data: any) {
    try {
      await fetch('/api/analytics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
    } catch (error) {
      console.error('Failed to send analytics data:', error);
    }
  }

  private sendAlert(message: string, data: any) {
    // Send alerts for critical performance issues
    if (env.NODE_ENV === 'production') {
      console.error(`🚨 ${message}:`, data);
      // TODO: Implement alerting logic (email, Slack, etc.)
    }
  }

  // Public methods for manual tracking
  startTimer(name: string): () => void {
    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      this.recordMetric(name, duration);
    };
  }

  trackUserAction(action: string, data?: any) {
    this.sendToAnalytics({
      type: 'user_action',
      action,
      data,
      timestamp: Date.now(),
    });
  }

  getMetrics(): Record<string, number[]> {
    return Object.fromEntries(this.metrics);
  }

  destroy() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
    this.metrics.clear();
  }
}

// Error boundary reporting
export class ErrorReporter {
  static init() {
    if (typeof window === 'undefined') return;

    // Global error handler
    window.addEventListener('error', (event) => {
      ErrorReporter.reportError(event.error, {
        type: 'javascript_error',
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      });
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      ErrorReporter.reportError(event.reason, {
        type: 'unhandled_promise_rejection',
      });
    });

    console.log('🛡️ Error reporting initialized');
  }

  static reportError(error: Error | any, context?: any) {
    const errorReport = {
      message: error?.message || 'Unknown error',
      stack: error?.stack,
      context,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
    };

    // Send to error tracking service
    if (monitoringConfig.sentryDsn) {
      // Sentry will handle this automatically if configured
    } else {
      // Send to custom error endpoint
      fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorReport),
      }).catch(() => {
        // Fallback: log to console if error reporting fails
        console.error('Failed to report error:', errorReport);
      });
    }
  }
}

// React hook for performance monitoring
export function usePerformanceMonitoring() {
  if (typeof window === 'undefined') return { trackEvent: () => {}, trackPerformance: () => {} };

  const monitoring = performanceMonitor;

  const trackEvent = (name: string, value?: number, metadata?: Record<string, any>) => {
    monitoring.trackUserAction(name, { value, ...metadata });
  };

  const trackPerformance = (metric: string, value: number) => {
    monitoring.recordMetric(metric, value);
  };

  return { trackEvent, trackPerformance };
}

// Singleton instances
export const performanceMonitor = new PerformanceMonitor();

// Auto-initialize in browser environment
if (typeof window !== 'undefined') {
  // Initialize error reporting immediately
  ErrorReporter.init();

  // Initialize performance monitoring after page load
  if (document.readyState === 'complete') {
    performanceMonitor.init();
  } else {
    window.addEventListener('load', () => {
      performanceMonitor.init();
    });
  }
}
