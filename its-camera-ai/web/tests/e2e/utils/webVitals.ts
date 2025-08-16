import { Page } from 'puppeteer';

export interface WebVitalsMetrics {
  // Core Web Vitals
  lcp: number | null; // Largest Contentful Paint
  fid: number | null; // First Input Delay (measured on interaction)
  cls: number | null; // Cumulative Layout Shift

  // Additional Performance Metrics
  fcp: number | null; // First Contentful Paint
  ttfb: number | null; // Time to First Byte
  tbt: number | null; // Total Blocking Time
  si: number | null; // Speed Index

  // Navigation Timing
  domContentLoaded: number | null;
  loadEvent: number | null;
  navigationStart: number | null;

  // Resource Timing
  resourceCount: number;
  transferSize: number;
  encodedBodySize: number;
  decodedBodySize: number;

  // Memory (if available)
  usedJSHeapSize?: number;
  totalJSHeapSize?: number;
  jsHeapSizeLimit?: number;
}

export interface WebVitalsThresholds {
  lcp: { good: number; needsImprovement: number };
  fid: { good: number; needsImprovement: number };
  cls: { good: number; needsImprovement: number };
  fcp: { good: number; needsImprovement: number };
  ttfb: { good: number; needsImprovement: number };
  tbt: { good: number; needsImprovement: number };
}

export interface WebVitalsReport {
  metrics: WebVitalsMetrics;
  scores: Record<keyof WebVitalsThresholds, 'good' | 'needs-improvement' | 'poor'>;
  overallScore: 'good' | 'needs-improvement' | 'poor';
  timestamp: string;
  url: string;
  userAgent: string;
}

export class WebVitalsTester {
  private page: Page;
  private thresholds: WebVitalsThresholds;

  constructor(page: Page, customThresholds?: Partial<WebVitalsThresholds>) {
    this.page = page;
    this.thresholds = {
      // Core Web Vitals thresholds (Google recommendations)
      lcp: { good: 2500, needsImprovement: 4000 },
      fid: { good: 100, needsImprovement: 300 },
      cls: { good: 0.1, needsImprovement: 0.25 },

      // Additional metrics thresholds
      fcp: { good: 1800, needsImprovement: 3000 },
      ttfb: { good: 800, needsImprovement: 1800 },
      tbt: { good: 200, needsImprovement: 600 },

      ...customThresholds,
    };
  }

  /**
   * Initialize Web Vitals collection
   */
  async initializeWebVitals(): Promise<void> {
    // Inject Web Vitals library and collection script
    await this.page.addScriptTag({
      url: 'https://unpkg.com/web-vitals@3/dist/web-vitals.iife.js',
    });

    // Initialize metrics collection
    await this.page.evaluate(() => {
      // @ts-ignore - web-vitals is loaded via script tag
      if (typeof webVitals !== 'undefined') {
        // @ts-ignore
        window.webVitalsMetrics = {
          lcp: null,
          fid: null,
          cls: null,
          fcp: null,
          ttfb: null,
          tbt: null,
        };

        // Collect Core Web Vitals
        // @ts-ignore
        webVitals.onLCP((metric) => {
          // @ts-ignore
          window.webVitalsMetrics.lcp = metric.value;
        });

        // @ts-ignore
        webVitals.onFID((metric) => {
          // @ts-ignore
          window.webVitalsMetrics.fid = metric.value;
        });

        // @ts-ignore
        webVitals.onCLS((metric) => {
          // @ts-ignore
          window.webVitalsMetrics.cls = metric.value;
        });

        // @ts-ignore
        webVitals.onFCP((metric) => {
          // @ts-ignore
          window.webVitalsMetrics.fcp = metric.value;
        });

        // @ts-ignore
        webVitals.onTTFB((metric) => {
          // @ts-ignore
          window.webVitalsMetrics.ttfb = metric.value;
        });

        // Calculate TBT manually (simplified)
        const observer = new PerformanceObserver((list) => {
          let tbt = 0;
          list.getEntries().forEach((entry) => {
            if (entry.duration > 50) {
              tbt += entry.duration - 50;
            }
          });
          // @ts-ignore
          window.webVitalsMetrics.tbt = tbt;
        });
        observer.observe({ type: 'longtask', buffered: true });
      }
    });
  }

  /**
   * Collect comprehensive Web Vitals metrics
   */
  async collectWebVitals(): Promise<WebVitalsMetrics> {
    // Wait for page to stabilize
    await this.page.waitForLoadState('networkidle');
    await this.page.waitForTimeout(2000);

    // Get Web Vitals metrics
    const webVitalsMetrics = await this.page.evaluate(() => {
      // @ts-ignore
      return window.webVitalsMetrics || {};
    });

    // Get navigation timing metrics
    const navigationTiming = await this.page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const paint = performance.getEntriesByType('paint');

      return {
        navigationStart: navigation.navigationStart,
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.navigationStart,
        loadEvent: navigation.loadEventEnd - navigation.navigationStart,
        ttfb: navigation.responseStart - navigation.navigationStart,
        fcp: paint.find(p => p.name === 'first-contentful-paint')?.startTime || null,
      };
    });

    // Get resource timing metrics
    const resourceTiming = await this.page.evaluate(() => {
      const resources = performance.getEntriesByType('resource');

      const stats = resources.reduce(
        (acc, resource) => {
          acc.count++;
          acc.transferSize += (resource as PerformanceResourceTiming).transferSize || 0;
          acc.encodedBodySize += (resource as PerformanceResourceTiming).encodedBodySize || 0;
          acc.decodedBodySize += (resource as PerformanceResourceTiming).decodedBodySize || 0;
          return acc;
        },
        { count: 0, transferSize: 0, encodedBodySize: 0, decodedBodySize: 0 }
      );

      return {
        resourceCount: stats.count,
        transferSize: stats.transferSize,
        encodedBodySize: stats.encodedBodySize,
        decodedBodySize: stats.decodedBodySize,
      };
    });

    // Get memory metrics if available
    const memoryMetrics = await this.page.evaluate(() => {
      // @ts-ignore
      if (performance.memory) {
        // @ts-ignore
        return {
          usedJSHeapSize: performance.memory.usedJSHeapSize,
          totalJSHeapSize: performance.memory.totalJSHeapSize,
          jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
        };
      }
      return {};
    });

    // Calculate Speed Index (simplified approximation)
    const speedIndex = await this.calculateSpeedIndex();

    return {
      // Core Web Vitals
      lcp: webVitalsMetrics.lcp,
      fid: webVitalsMetrics.fid,
      cls: webVitalsMetrics.cls,

      // Additional Performance Metrics
      fcp: navigationTiming.fcp || webVitalsMetrics.fcp,
      ttfb: navigationTiming.ttfb || webVitalsMetrics.ttfb,
      tbt: webVitalsMetrics.tbt,
      si: speedIndex,

      // Navigation Timing
      domContentLoaded: navigationTiming.domContentLoaded,
      loadEvent: navigationTiming.loadEvent,
      navigationStart: navigationTiming.navigationStart,

      // Resource Timing
      resourceCount: resourceTiming.resourceCount,
      transferSize: resourceTiming.transferSize,
      encodedBodySize: resourceTiming.encodedBodySize,
      decodedBodySize: resourceTiming.decodedBodySize,

      // Memory
      ...memoryMetrics,
    };
  }

  /**
   * Generate Web Vitals report with scores
   */
  async generateReport(): Promise<WebVitalsReport> {
    const metrics = await this.collectWebVitals();
    const scores = this.calculateScores(metrics);
    const overallScore = this.calculateOverallScore(scores);

    return {
      metrics,
      scores,
      overallScore,
      timestamp: new Date().toISOString(),
      url: this.page.url(),
      userAgent: await this.page.evaluate(() => navigator.userAgent),
    };
  }

  /**
   * Trigger First Input Delay measurement
   */
  async triggerFIDMeasurement(): Promise<void> {
    // Click on the page to trigger FID measurement
    await this.page.click('body');
    await this.page.waitForTimeout(100);
  }

  /**
   * Monitor real-time Web Vitals
   */
  async monitorRealTimeWebVitals(duration: number = 30000): Promise<WebVitalsMetrics[]> {
    const metrics: WebVitalsMetrics[] = [];
    const interval = 1000; // Collect metrics every second
    const iterations = duration / interval;

    for (let i = 0; i < iterations; i++) {
      const currentMetrics = await this.collectWebVitals();
      metrics.push(currentMetrics);
      await this.page.waitForTimeout(interval);
    }

    return metrics;
  }

  /**
   * Calculate scores based on thresholds
   */
  private calculateScores(metrics: WebVitalsMetrics): Record<keyof WebVitalsThresholds, 'good' | 'needs-improvement' | 'poor'> {
    const scores: Record<string, 'good' | 'needs-improvement' | 'poor'> = {};

    Object.keys(this.thresholds).forEach((metric) => {
      const value = metrics[metric as keyof WebVitalsMetrics] as number;
      const threshold = this.thresholds[metric as keyof WebVitalsThresholds];

      if (value === null || value === undefined) {
        scores[metric] = 'poor'; // Treat missing metrics as poor
      } else if (value <= threshold.good) {
        scores[metric] = 'good';
      } else if (value <= threshold.needsImprovement) {
        scores[metric] = 'needs-improvement';
      } else {
        scores[metric] = 'poor';
      }
    });

    return scores as Record<keyof WebVitalsThresholds, 'good' | 'needs-improvement' | 'poor'>;
  }

  /**
   * Calculate overall score
   */
  private calculateOverallScore(scores: Record<string, 'good' | 'needs-improvement' | 'poor'>): 'good' | 'needs-improvement' | 'poor' {
    const scoreValues = Object.values(scores);
    const goodCount = scoreValues.filter(s => s === 'good').length;
    const needsImprovementCount = scoreValues.filter(s => s === 'needs-improvement').length;
    const poorCount = scoreValues.filter(s => s === 'poor').length;

    // If majority are good, overall is good
    if (goodCount >= scoreValues.length / 2) {
      return 'good';
    }

    // If any core web vitals are poor, overall is poor
    if (poorCount > 0) {
      return 'poor';
    }

    return 'needs-improvement';
  }

  /**
   * Calculate Speed Index (simplified)
   */
  private async calculateSpeedIndex(): Promise<number | null> {
    try {
      return await this.page.evaluate(() => {
        // Simplified Speed Index calculation
        // This is an approximation - real Speed Index requires visual completeness data
        const paint = performance.getEntriesByType('paint');
        const fcp = paint.find(p => p.name === 'first-contentful-paint');
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;

        if (fcp && navigation) {
          // Rough approximation: FCP + some additional time based on load event
          const loadTime = navigation.loadEventEnd - navigation.navigationStart;
          return fcp.startTime + (loadTime * 0.3);
        }

        return null;
      });
    } catch {
      return null;
    }
  }
}

/**
 * Web Vitals testing utilities
 */
export const webVitalsUtils = {
  /**
   * Format metrics for display
   */
  formatMetrics(metrics: WebVitalsMetrics): Record<string, string> {
    const formatValue = (value: number | null | undefined, unit: string = 'ms'): string => {
      if (value === null || value === undefined) return 'N/A';
      return `${Math.round(value)}${unit}`;
    };

    return {
      LCP: formatValue(metrics.lcp),
      FID: formatValue(metrics.fid),
      CLS: formatValue(metrics.cls, ''),
      FCP: formatValue(metrics.fcp),
      TTFB: formatValue(metrics.ttfb),
      TBT: formatValue(metrics.tbt),
      SI: formatValue(metrics.si),
      'DOM Content Loaded': formatValue(metrics.domContentLoaded),
      'Load Event': formatValue(metrics.loadEvent),
      'Resource Count': metrics.resourceCount.toString(),
      'Transfer Size': formatValue(metrics.transferSize / 1024, 'KB'),
    };
  },

  /**
   * Check if metrics meet performance budgets
   */
  checkPerformanceBudget(
    metrics: WebVitalsMetrics,
    budget: Partial<WebVitalsMetrics>
  ): { passed: boolean; violations: string[] } {
    const violations: string[] = [];

    Object.keys(budget).forEach((key) => {
      const metricKey = key as keyof WebVitalsMetrics;
      const budgetValue = budget[metricKey] as number;
      const actualValue = metrics[metricKey] as number;

      if (actualValue !== null && actualValue > budgetValue) {
        violations.push(`${key}: ${actualValue} exceeds budget of ${budgetValue}`);
      }
    });

    return {
      passed: violations.length === 0,
      violations,
    };
  },

  /**
   * Generate performance recommendations
   */
  generateRecommendations(report: WebVitalsReport): string[] {
    const recommendations: string[] = [];

    if (report.scores.lcp !== 'good') {
      recommendations.push(
        'Improve Largest Contentful Paint by optimizing images, using CDN, and reducing server response times'
      );
    }

    if (report.scores.fid !== 'good') {
      recommendations.push(
        'Improve First Input Delay by reducing JavaScript execution time and using web workers'
      );
    }

    if (report.scores.cls !== 'good') {
      recommendations.push(
        'Improve Cumulative Layout Shift by setting image dimensions and avoiding dynamic content insertion'
      );
    }

    if (report.scores.fcp !== 'good') {
      recommendations.push(
        'Improve First Contentful Paint by optimizing critical rendering path and reducing render-blocking resources'
      );
    }

    if (report.scores.ttfb !== 'good') {
      recommendations.push(
        'Improve Time to First Byte by optimizing server response time and using faster hosting'
      );
    }

    if (report.scores.tbt !== 'good') {
      recommendations.push(
        'Reduce Total Blocking Time by breaking up long-running JavaScript tasks and optimizing third-party scripts'
      );
    }

    if (report.metrics.resourceCount > 100) {
      recommendations.push(
        'Reduce the number of resources by bundling, removing unused dependencies, and using resource hints'
      );
    }

    if (report.metrics.transferSize > 2 * 1024 * 1024) { // 2MB
      recommendations.push(
        'Reduce transfer size by enabling compression, optimizing images, and removing unused code'
      );
    }

    return recommendations;
  },
};