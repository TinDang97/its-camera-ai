import { LaunchOptions, BrowserLaunchArgumentOptions, Browser } from 'puppeteer';
import puppeteer from 'puppeteer';

export interface E2EConfig {
  browser: LaunchOptions & BrowserLaunchArgumentOptions;
  viewport: {
    width: number;
    height: number;
  };
  network: {
    throttling: string;
    cacheEnabled: boolean;
  };
  screenshots: {
    onFailure: boolean;
    path: string;
  };
  videos: {
    enabled: boolean;
    path: string;
  };
  timeout: {
    default: number;
    navigation: number;
    element: number;
  };
  baseUrl: string;
  headless: boolean;
}

export const E2E_CONFIG: E2EConfig = {
  browser: {
    headless: process.env.CI === 'true' || process.env.HEADLESS === 'true',
    slowMo: process.env.CI === 'true' ? 0 : 50,
    devtools: process.env.DEBUG === 'true',
    timeout: 30000,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--disable-gpu',
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      '--disable-features=TranslateUI',
      '--disable-ipc-flooding-protection',
      '--window-size=1920,1080',
    ],
  },
  viewport: {
    width: 1920,
    height: 1080
  },
  network: {
    throttling: 'Fast 3G',
    cacheEnabled: false,
  },
  screenshots: {
    onFailure: true,
    path: './tests/e2e/screenshots',
  },
  videos: {
    enabled: process.env.RECORD_VIDEO === 'true',
    path: './tests/e2e/videos',
  },
  timeout: {
    default: 30000,
    navigation: 60000,
    element: 10000,
  },
  baseUrl: process.env.E2E_BASE_URL || 'http://localhost:3002',
  headless: process.env.CI === 'true' || process.env.HEADLESS === 'true',
};

export const WEB_VITALS_THRESHOLDS = {
  LCP: 2500, // Largest Contentful Paint
  FID: 100,  // First Input Delay
  CLS: 0.1,  // Cumulative Layout Shift
  FCP: 1800, // First Contentful Paint
  TTFB: 800, // Time to First Byte
};

export const PERFORMANCE_BUDGETS = {
  mainBundle: 500 * 1024, // 500KB
  totalPageSize: 2 * 1024 * 1024, // 2MB
  firstLoad: 3000, // 3 seconds
  routeChange: 1000, // 1 second
};

/**
 * Get configured test browser instance
 */
export async function getTestBrowser(): Promise<Browser> {
  return await puppeteer.launch(E2E_CONFIG.browser);
}