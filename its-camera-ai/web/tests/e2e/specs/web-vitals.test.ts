import { Browser, Page } from 'puppeteer';
import { AnalyticsDashboardPage } from '../pages/AnalyticsDashboardPage';
import { LoginPage } from '../pages/LoginPage';
import { CameraManagementPage } from '../pages/CameraManagementPage';
import { getTestBrowser } from '../config/puppeteer.config';
import { WebVitalsTester, webVitalsUtils, WebVitalsReport } from '../utils/webVitals';
import { TEST_USERS, MOCK_API_RESPONSES } from '../fixtures/test-data';

describe('Web Vitals Performance Tests', () => {
  let browser: Browser;
  let page: Page;
  let analyticsPage: AnalyticsDashboardPage;
  let loginPage: LoginPage;
  let cameraPage: CameraManagementPage;
  let webVitalsTester: WebVitalsTester;

  // Performance budgets (in milliseconds unless specified)
  const performanceBudgets = {
    lcp: 2500, // Largest Contentful Paint
    fid: 100,  // First Input Delay
    cls: 0.1,  // Cumulative Layout Shift (unitless)
    fcp: 1800, // First Contentful Paint
    ttfb: 800, // Time to First Byte
    tbt: 200,  // Total Blocking Time
  };

  beforeAll(async () => {
    browser = await getTestBrowser({
      // Performance-focused browser settings
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-web-security',
        '--disable-features=TranslateUI',
        '--disable-ipc-flooding-protection',
      ],
    });
  });

  afterAll(async () => {
    if (browser) {
      await browser.close();
    }
  });

  beforeEach(async () => {
    page = await browser.newPage();

    // Set cache policy for consistent testing
    await page.setCacheEnabled(false);

    // Set network conditions for consistent testing
    const client = await page.target().createCDPSession();
    await client.send('Network.emulateNetworkConditions', {
      offline: false,
      downloadThroughput: 1.6 * 1024 * 1024 / 8, // 1.6 Mbps
      uploadThroughput: 750 * 1024 / 8, // 750 Kbps
      latency: 40, // 40ms RTT
    });

    analyticsPage = new AnalyticsDashboardPage(page);
    loginPage = new LoginPage(page);
    cameraPage = new CameraManagementPage(page);
    webVitalsTester = new WebVitalsTester(page, {
      // Custom thresholds for our application
      lcp: { good: 2000, needsImprovement: 3500 },
      fid: { good: 80, needsImprovement: 200 },
      cls: { good: 0.05, needsImprovement: 0.15 },
      fcp: { good: 1500, needsImprovement: 2500 },
      ttfb: { good: 600, needsImprovement: 1200 },
      tbt: { good: 150, needsImprovement: 400 },
    });

    // Initialize Web Vitals collection
    await webVitalsTester.initializeWebVitals();

    // Mock API responses for consistent performance testing
    await page.setRequestInterception(true);
    page.on('request', (request) => {
      const url = request.url();

      if (url.includes('/api/analytics/current')) {
        request.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(MOCK_API_RESPONSES.ANALYTICS_DATA.metrics),
        });
      } else if (url.includes('/api/cameras')) {
        request.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(MOCK_API_RESPONSES.CAMERAS_LIST),
        });
      } else if (url.includes('/api/incidents')) {
        request.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(MOCK_API_RESPONSES.INCIDENTS_LIST),
        });
      } else {
        request.continue();
      }
    });
  });

  afterEach(async () => {
    if (page) {
      await page.close();
    }
  });

  describe('Authentication Performance', () => {
    test('should meet performance budgets for login page', async () => {
      await loginPage.navigateToLogin();

      // Trigger FID measurement
      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      // Check performance budgets
      const budgetCheck = webVitalsUtils.checkPerformanceBudget(
        report.metrics,
        performanceBudgets
      );

      expect(budgetCheck.passed).toBe(true);
      if (!budgetCheck.passed) {
        console.log('Performance budget violations:', budgetCheck.violations);
        console.log('Performance metrics:', webVitalsUtils.formatMetrics(report.metrics));
      }

      // Core Web Vitals should be good or needs improvement
      expect(['good', 'needs-improvement']).toContain(report.scores.lcp);
      expect(['good', 'needs-improvement']).toContain(report.scores.fid);
      expect(['good', 'needs-improvement']).toContain(report.scores.cls);
    });

    test('should measure login flow performance', async () => {
      await loginPage.navigateToLogin();

      const startTime = Date.now();
      await loginPage.loginWithUser(TEST_USERS.admin);
      const loginTime = Date.now() - startTime;

      // Login should complete within 3 seconds
      expect(loginTime).toBeLessThan(3000);

      const report = await webVitalsTester.generateReport();

      // Verify post-login performance
      expect(report.metrics.cls).toBeLessThan(0.25); // CLS should be acceptable
      expect(report.overallScore).not.toBe('poor');
    });
  });

  describe('Dashboard Performance', () => {
    beforeEach(async () => {
      await loginPage.navigateToLogin();
      await loginPage.loginWithUser(TEST_USERS.admin);
    });

    test('should meet performance budgets for overview dashboard', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      // Log performance metrics for debugging
      console.log('Overview Dashboard Performance:', webVitalsUtils.formatMetrics(report.metrics));

      // Check Core Web Vitals
      expect(report.metrics.lcp).toBeLessThan(4000); // Should be less than 4s
      expect(report.metrics.cls).toBeLessThan(0.25);  // Should have low layout shift

      if (report.metrics.fid !== null) {
        expect(report.metrics.fid).toBeLessThan(300); // Should be responsive
      }

      // Overall score should not be poor
      expect(report.overallScore).not.toBe('poor');
    });

    test('should meet performance budgets for traffic dashboard', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('traffic');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      console.log('Traffic Dashboard Performance:', webVitalsUtils.formatMetrics(report.metrics));

      // Traffic dashboard with charts should still perform well
      expect(report.metrics.lcp).toBeLessThan(5000); // Slightly higher for charts
      expect(report.metrics.cls).toBeLessThan(0.3);   // Charts might cause some shift

      // Check resource usage
      expect(report.metrics.resourceCount).toBeLessThan(150);
      expect(report.metrics.transferSize).toBeLessThan(3 * 1024 * 1024); // 3MB max
    });

    test('should meet performance budgets for incidents dashboard', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('incidents');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      console.log('Incidents Dashboard Performance:', webVitalsUtils.formatMetrics(report.metrics));

      expect(report.metrics.lcp).toBeLessThan(4000);
      expect(report.metrics.cls).toBeLessThan(0.25);
      expect(report.overallScore).not.toBe('poor');
    });

    test('should meet performance budgets for cameras dashboard', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('cameras');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      console.log('Cameras Dashboard Performance:', webVitalsUtils.formatMetrics(report.metrics));

      expect(report.metrics.lcp).toBeLessThan(4000);
      expect(report.metrics.cls).toBeLessThan(0.25);
      expect(report.overallScore).not.toBe('poor');
    });

    test('should meet performance budgets for maps dashboard', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('maps');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      console.log('Maps Dashboard Performance:', webVitalsUtils.formatMetrics(report.metrics));

      // Maps might have higher LCP due to map tiles
      expect(report.metrics.lcp).toBeLessThan(6000);
      expect(report.metrics.cls).toBeLessThan(0.4); // Maps might cause layout shift
      expect(report.overallScore).not.toBe('poor');
    });

    test('should meet performance budgets for reports dashboard', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('reports');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      console.log('Reports Dashboard Performance:', webVitalsUtils.formatMetrics(report.metrics));

      expect(report.metrics.lcp).toBeLessThan(4000);
      expect(report.metrics.cls).toBeLessThan(0.25);
      expect(report.overallScore).not.toBe('poor');
    });
  });

  describe('Camera Management Performance', () => {
    beforeEach(async () => {
      await loginPage.navigateToLogin();
      await loginPage.loginWithUser(TEST_USERS.admin);
    });

    test('should meet performance budgets for camera management page', async () => {
      await cameraPage.navigateToPage();
      await cameraPage.waitForPageLoad();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      console.log('Camera Management Performance:', webVitalsUtils.formatMetrics(report.metrics));

      expect(report.metrics.lcp).toBeLessThan(4000);
      expect(report.metrics.cls).toBeLessThan(0.25);
      expect(report.overallScore).not.toBe('poor');
    });
  });

  describe('Navigation Performance', () => {
    beforeEach(async () => {
      await loginPage.navigateToLogin();
      await loginPage.loginWithUser(TEST_USERS.admin);
    });

    test('should maintain performance during dashboard navigation', async () => {
      const dashboards = ['overview', 'traffic', 'incidents', 'cameras'] as const;
      const navigationReports: WebVitalsReport[] = [];

      for (const dashboard of dashboards) {
        const startTime = Date.now();
        await analyticsPage.navigateToAnalyticsDashboard(dashboard);
        await analyticsPage.waitForLoadingToComplete();
        const navigationTime = Date.now() - startTime;

        // Navigation should be fast
        expect(navigationTime).toBeLessThan(2000);

        await webVitalsTester.triggerFIDMeasurement();
        const report = await webVitalsTester.generateReport();
        navigationReports.push(report);

        // Each navigation should maintain good performance
        expect(report.metrics.cls).toBeLessThan(0.3); // Some layout shift during navigation is acceptable
        expect(report.overallScore).not.toBe('poor');
      }

      // Log navigation performance summary
      console.log('Navigation Performance Summary:');
      navigationReports.forEach((report, index) => {
        console.log(`${dashboards[index]}: ${webVitalsUtils.formatMetrics(report.metrics).LCP} LCP, ${webVitalsUtils.formatMetrics(report.metrics).CLS} CLS`);
      });
    });
  });

  describe('Real-time Performance', () => {
    beforeEach(async () => {
      await loginPage.navigateToLogin();
      await loginPage.loginWithUser(TEST_USERS.admin);
    });

    test('should maintain performance with real-time updates', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      // Monitor performance over time with simulated real-time updates
      const metrics = await webVitalsTester.monitorRealTimeWebVitals(10000); // 10 seconds

      // Performance should remain stable over time
      const clsValues = metrics
        .map(m => m.cls)
        .filter(cls => cls !== null) as number[];

      if (clsValues.length > 0) {
        const maxCls = Math.max(...clsValues);
        expect(maxCls).toBeLessThan(0.5); // CLS shouldn't grow too much over time
      }

      // Memory usage shouldn't grow excessively
      const memoryMetrics = metrics.filter(m => m.usedJSHeapSize !== undefined);
      if (memoryMetrics.length > 1) {
        const initialMemory = memoryMetrics[0].usedJSHeapSize!;
        const finalMemory = memoryMetrics[memoryMetrics.length - 1].usedJSHeapSize!;
        const memoryGrowth = finalMemory - initialMemory;

        // Memory growth should be reasonable (less than 50MB over 10 seconds)
        expect(memoryGrowth).toBeLessThan(50 * 1024 * 1024);
      }
    });
  });

  describe('Mobile Performance', () => {
    beforeEach(async () => {
      // Set mobile viewport and network conditions
      await page.setViewport({ width: 375, height: 667 });

      const client = await page.target().createCDPSession();
      await client.send('Network.emulateNetworkConditions', {
        offline: false,
        downloadThroughput: 0.75 * 1024 * 1024 / 8, // 0.75 Mbps (3G)
        uploadThroughput: 0.25 * 1024 * 1024 / 8, // 0.25 Mbps
        latency: 100, // 100ms RTT
      });

      await loginPage.navigateToLogin();
      await loginPage.loginWithUser(TEST_USERS.admin);
    });

    test('should meet mobile performance budgets', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();

      console.log('Mobile Performance:', webVitalsUtils.formatMetrics(report.metrics));

      // Mobile performance budgets (more lenient)
      expect(report.metrics.lcp).toBeLessThan(6000); // 6s for mobile
      expect(report.metrics.cls).toBeLessThan(0.3);

      if (report.metrics.fid !== null) {
        expect(report.metrics.fid).toBeLessThan(500); // 500ms for mobile
      }

      // Transfer size should be reasonable for mobile
      expect(report.metrics.transferSize).toBeLessThan(2 * 1024 * 1024); // 2MB max for mobile
    });
  });

  describe('Performance Recommendations', () => {
    beforeEach(async () => {
      await loginPage.navigateToLogin();
      await loginPage.loginWithUser(TEST_USERS.admin);
    });

    test('should generate performance recommendations', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await webVitalsTester.triggerFIDMeasurement();

      const report = await webVitalsTester.generateReport();
      const recommendations = webVitalsUtils.generateRecommendations(report);

      console.log('Performance Report:', {
        scores: report.scores,
        overallScore: report.overallScore,
        recommendations: recommendations,
      });

      // Should generate recommendations if performance is not optimal
      if (report.overallScore === 'poor' || report.overallScore === 'needs-improvement') {
        expect(recommendations.length).toBeGreaterThan(0);
      }

      // Log recommendations for manual review
      if (recommendations.length > 0) {
        console.log('Performance Recommendations:');
        recommendations.forEach((rec, index) => {
          console.log(`${index + 1}. ${rec}`);
        });
      }
    });
  });

  describe('Performance Regression Detection', () => {
    beforeEach(async () => {
      await loginPage.navigateToLogin();
      await loginPage.loginWithUser(TEST_USERS.admin);
    });

    test('should detect performance regressions', async () => {
      // This test would typically compare against baseline metrics
      // For now, we'll just ensure metrics are collected consistently

      const runs = [];

      // Run test multiple times to check consistency
      for (let i = 0; i < 3; i++) {
        await analyticsPage.navigateToAnalyticsDashboard('overview');
        await analyticsPage.waitForLoadingToComplete();

        await webVitalsTester.triggerFIDMeasurement();
        const report = await webVitalsTester.generateReport();
        runs.push(report);

        // Small delay between runs
        await page.waitForTimeout(1000);
      }

      // Check for consistency in performance metrics
      const lcpValues = runs.map(r => r.metrics.lcp).filter(lcp => lcp !== null) as number[];
      const clsValues = runs.map(r => r.metrics.cls).filter(cls => cls !== null) as number[];

      if (lcpValues.length > 1) {
        const lcpVariance = Math.max(...lcpValues) - Math.min(...lcpValues);
        // LCP shouldn't vary too much between runs (within 50% of average)
        const avgLcp = lcpValues.reduce((a, b) => a + b, 0) / lcpValues.length;
        expect(lcpVariance).toBeLessThan(avgLcp * 0.5);
      }

      if (clsValues.length > 1) {
        const clsVariance = Math.max(...clsValues) - Math.min(...clsValues);
        // CLS should be consistent between runs
        expect(clsVariance).toBeLessThan(0.2);
      }

      console.log('Performance Consistency Check:', {
        lcpValues,
        clsValues,
        avgLcp: lcpValues.length > 0 ? lcpValues.reduce((a, b) => a + b, 0) / lcpValues.length : 'N/A',
        avgCls: clsValues.length > 0 ? clsValues.reduce((a, b) => a + b, 0) / clsValues.length : 'N/A',
      });
    });
  });
});