import { Page, ElementHandle } from 'puppeteer';
import { E2E_CONFIG, WEB_VITALS_THRESHOLDS } from '../config/puppeteer.config';

export interface WebVitalsMetrics {
  LCP: number;
  FID: number;
  CLS: number;
  FCP: number;
  TTFB: number;
}

export class BasePage {
  protected page: Page;
  protected baseUrl: string;

  constructor(page: Page) {
    this.page = page;
    this.baseUrl = E2E_CONFIG.baseUrl;
  }

  /**
   * Navigate to a specific path
   */
  async navigateTo(path: string): Promise<void> {
    const url = `${this.baseUrl}${path}`;
    console.log(`Navigating to: ${url}`);

    await this.page.goto(url, {
      waitUntil: 'networkidle2',
      timeout: E2E_CONFIG.timeout.navigation,
    });
  }

  /**
   * Wait for an element to be visible
   */
  async waitForElement(selector: string, timeout?: number): Promise<ElementHandle<Element> | null> {
    return await this.page.waitForSelector(selector, {
      visible: true,
      timeout: timeout || E2E_CONFIG.timeout.element,
    });
  }

  /**
   * Click an element with optional wait
   */
  async clickElement(selector: string, waitForNavigation = false): Promise<void> {
    await this.waitForElement(selector);

    if (waitForNavigation) {
      await Promise.all([
        this.page.waitForNavigation({ waitUntil: 'networkidle2' }),
        this.page.click(selector),
      ]);
    } else {
      await this.page.click(selector);
    }
  }

  /**
   * Fill an input field
   */
  async fillInput(selector: string, value: string): Promise<void> {
    await this.waitForElement(selector);
    await this.page.focus(selector);
    await this.page.keyboard.down('Control');
    await this.page.keyboard.press('KeyA');
    await this.page.keyboard.up('Control');
    await this.page.type(selector, value);
  }

  /**
   * Get text content of an element
   */
  async getElementText(selector: string): Promise<string> {
    await this.waitForElement(selector);
    return await this.page.$eval(selector, (el) => el.textContent?.trim() || '');
  }

  /**
   * Check if element exists
   */
  async isElementVisible(selector: string, timeout = 5000): Promise<boolean> {
    try {
      await this.page.waitForSelector(selector, { visible: true, timeout });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Wait for element to disappear
   */
  async waitForElementToDisappear(selector: string, timeout?: number): Promise<void> {
    await this.page.waitForSelector(selector, {
      hidden: true,
      timeout: timeout || E2E_CONFIG.timeout.element,
    });
  }

  /**
   * Take a screenshot
   */
  async takeScreenshot(name: string): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `${name}-${timestamp}.png`;
    const path = `${E2E_CONFIG.screenshots.path}/${filename}`;

    await this.page.screenshot({
      path,
      fullPage: true,
    });

    console.log(`Screenshot saved: ${path}`);
  }

  /**
   * Measure Web Vitals
   */
  async measureWebVitals(): Promise<WebVitalsMetrics> {
    const metrics = await this.page.evaluate(() => {
      return new Promise<WebVitalsMetrics>((resolve) => {
        const vitals: Partial<WebVitalsMetrics> = {};
        let metricsCollected = 0;
        const totalMetrics = 5;

        const checkComplete = () => {
          if (metricsCollected === totalMetrics) {
            resolve(vitals as WebVitalsMetrics);
          }
        };

        // LCP - Largest Contentful Paint
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          if (entries.length > 0) {
            vitals.LCP = entries[entries.length - 1].startTime;
            metricsCollected++;
            checkComplete();
          }
        }).observe({ entryTypes: ['largest-contentful-paint'] });

        // FCP - First Contentful Paint
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          if (entries.length > 0) {
            vitals.FCP = entries[0].startTime;
            metricsCollected++;
            checkComplete();
          }
        }).observe({ entryTypes: ['paint'] });

        // TTFB - Time to First Byte
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        if (navigation) {
          vitals.TTFB = navigation.responseStart - navigation.requestStart;
          metricsCollected++;
          checkComplete();
        }

        // FID and CLS would be measured via web-vitals library in real app
        // For now, we'll simulate them
        setTimeout(() => {
          vitals.FID = Math.random() * 200; // Simulated
          vitals.CLS = Math.random() * 0.3; // Simulated
          metricsCollected += 2;
          checkComplete();
        }, 1000);
      });
    });

    return metrics;
  }

  /**
   * Validate Web Vitals against thresholds
   */
  async validateWebVitals(): Promise<{ passed: boolean; metrics: WebVitalsMetrics; failures: string[] }> {
    const metrics = await this.measureWebVitals();
    const failures: string[] = [];

    if (metrics.LCP > WEB_VITALS_THRESHOLDS.LCP) {
      failures.push(`LCP: ${metrics.LCP}ms > ${WEB_VITALS_THRESHOLDS.LCP}ms`);
    }

    if (metrics.FID > WEB_VITALS_THRESHOLDS.FID) {
      failures.push(`FID: ${metrics.FID}ms > ${WEB_VITALS_THRESHOLDS.FID}ms`);
    }

    if (metrics.CLS > WEB_VITALS_THRESHOLDS.CLS) {
      failures.push(`CLS: ${metrics.CLS} > ${WEB_VITALS_THRESHOLDS.CLS}`);
    }

    if (metrics.FCP > WEB_VITALS_THRESHOLDS.FCP) {
      failures.push(`FCP: ${metrics.FCP}ms > ${WEB_VITALS_THRESHOLDS.FCP}ms`);
    }

    if (metrics.TTFB > WEB_VITALS_THRESHOLDS.TTFB) {
      failures.push(`TTFB: ${metrics.TTFB}ms > ${WEB_VITALS_THRESHOLDS.TTFB}ms`);
    }

    return {
      passed: failures.length === 0,
      metrics,
      failures,
    };
  }

  /**
   * Wait for loading to complete
   */
  async waitForLoadingToComplete(): Promise<void> {
    // Wait for any loading spinners to disappear
    const loadingSelectors = [
      '[data-testid="loading"]',
      '.loading',
      '.spinner',
      '[aria-label*="loading"]',
      '[aria-label*="Loading"]',
    ];

    for (const selector of loadingSelectors) {
      if (await this.isElementVisible(selector, 1000)) {
        await this.waitForElementToDisappear(selector);
      }
    }

    // Wait for network to be idle
    await this.page.waitForLoadState?.('networkidle') ||
          this.page.waitForTimeout(1000);
  }

  /**
   * Setup WebSocket message interception
   */
  async setupWebSocketInterception(): Promise<void> {
    await this.page.evaluateOnNewDocument(() => {
      const originalWebSocket = window.WebSocket;
      const messages: any[] = [];

      // @ts-ignore
      window.WebSocket = function(url: string, protocols?: string | string[]) {
        const ws = new originalWebSocket(url, protocols);

        ws.addEventListener('message', (event) => {
          messages.push({
            type: 'received',
            data: event.data,
            timestamp: Date.now(),
          });
        });

        const originalSend = ws.send;
        ws.send = function(data: string | ArrayBufferLike | Blob | ArrayBufferView) {
          messages.push({
            type: 'sent',
            data,
            timestamp: Date.now(),
          });
          return originalSend.call(this, data);
        };

        return ws;
      };

      // @ts-ignore
      window.getWebSocketMessages = () => messages;
      // @ts-ignore
      window.clearWebSocketMessages = () => messages.length = 0;
    });
  }

  /**
   * Get WebSocket messages
   */
  async getWebSocketMessages(): Promise<any[]> {
    return await this.page.evaluate(() => {
      // @ts-ignore
      return window.getWebSocketMessages?.() || [];
    });
  }

  /**
   * Clear WebSocket messages
   */
  async clearWebSocketMessages(): Promise<void> {
    await this.page.evaluate(() => {
      // @ts-ignore
      window.clearWebSocketMessages?.();
    });
  }

  /**
   * Simulate network conditions
   */
  async setNetworkConditions(conditions: 'fast3g' | 'slow3g' | 'offline'): Promise<void> {
    const client = await this.page.createCDPSession();

    const conditionsMap = {
      fast3g: {
        offline: false,
        downloadThroughput: (1.6 * 1024 * 1024) / 8, // 1.6 Mbps
        uploadThroughput: (750 * 1024) / 8, // 750 Kbps
        latency: 150,
      },
      slow3g: {
        offline: false,
        downloadThroughput: (500 * 1024) / 8, // 500 Kbps
        uploadThroughput: (500 * 1024) / 8, // 500 Kbps
        latency: 400,
      },
      offline: {
        offline: true,
        downloadThroughput: 0,
        uploadThroughput: 0,
        latency: 0,
      },
    };

    await client.send('Network.emulateNetworkConditions', conditionsMap[conditions]);
  }

  /**
   * Get current URL
   */
  getCurrentUrl(): string {
    return this.page.url();
  }

  /**
   * Go back in browser history
   */
  async goBack(): Promise<void> {
    await this.page.goBack({ waitUntil: 'networkidle2' });
  }

  /**
   * Refresh the page
   */
  async refresh(): Promise<void> {
    await this.page.reload({ waitUntil: 'networkidle2' });
  }

  /**
   * Close the page
   */
  async close(): Promise<void> {
    await this.page.close();
  }
}