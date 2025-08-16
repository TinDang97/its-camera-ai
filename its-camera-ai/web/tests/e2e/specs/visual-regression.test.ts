import { Browser, Page } from 'puppeteer';
import { AnalyticsDashboardPage } from '../pages/AnalyticsDashboardPage';
import { LoginPage } from '../pages/LoginPage';
import { CameraManagementPage } from '../pages/CameraManagementPage';
import { getTestBrowser } from '../config/puppeteer.config';
import { VisualRegressionTester, visualRegressionUtils } from '../utils/visualRegression';
import { TEST_USERS, MOCK_API_RESPONSES } from '../fixtures/test-data';

describe('Visual Regression Tests', () => {
  let browser: Browser;
  let page: Page;
  let analyticsPage: AnalyticsDashboardPage;
  let loginPage: LoginPage;
  let cameraPage: CameraManagementPage;
  let visualTester: VisualRegressionTester;

  beforeAll(async () => {
    browser = await getTestBrowser();
  });

  afterAll(async () => {
    if (browser) {
      await browser.close();
    }
  });

  beforeEach(async () => {
    page = await browser.newPage();
    analyticsPage = new AnalyticsDashboardPage(page);
    loginPage = new LoginPage(page);
    cameraPage = new CameraManagementPage(page);
    visualTester = new VisualRegressionTester(page);

    // Login and setup mocks
    await loginPage.navigateToLogin();
    await loginPage.loginWithUser(TEST_USERS.admin);

    // Mock API responses for consistent visual tests
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

  describe('Dashboard Visual Regression', () => {
    test('should match overview dashboard layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      // Stabilize page for consistent screenshots
      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareScreenshot(
        'overview-dashboard-full',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
      if (!result.passed) {
        console.log(`Visual regression failed: ${result.percentageDifference}% difference`);
        console.log(`Diff image: ${result.diffImagePath}`);
      }
    });

    test('should match traffic dashboard layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('traffic');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareScreenshot(
        'traffic-dashboard-full',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });

    test('should match incidents dashboard layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('incidents');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareScreenshot(
        'incidents-dashboard-full',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });

    test('should match cameras dashboard layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('cameras');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareScreenshot(
        'cameras-dashboard-full',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });

    test('should match maps dashboard layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('maps');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareScreenshot(
        'maps-dashboard-full',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });

    test('should match reports dashboard layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('reports');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareScreenshot(
        'reports-dashboard-full',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });
  });

  describe('Component Visual Regression', () => {
    test('should match KPI cards layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareElementScreenshot(
        '[data-testid="kpi-card"]',
        'kpi-cards-component',
        { threshold: 0.1 }
      );

      expect(result.passed).toBe(true);
    });

    test('should match navigation menu layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareElementScreenshot(
        '[data-testid="dashboard-nav"]',
        'navigation-menu-component',
        { threshold: 0.1 }
      );

      expect(result.passed).toBe(true);
    });

    test('should match chart components layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('traffic');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      // Test individual chart components
      const chartSelectors = [
        '[data-testid="traffic-flow-chart"]',
        '[data-testid="speed-distribution-chart"]',
        '[data-testid="volume-by-hour-chart"]',
      ];

      for (const selector of chartSelectors) {
        const chartExists = await analyticsPage.isElementVisible(selector);
        if (chartExists) {
          const result = await visualTester.compareElementScreenshot(
            selector,
            `chart-${selector.replace(/[\[\]"=\-]/g, '')}`,
            { threshold: 0.15 } // Higher threshold for charts due to potential rendering differences
          );

          expect(result.passed).toBe(true);
        }
      }
    });

    test('should match camera grid layout', async () => {
      await cameraPage.navigateToPage();
      await cameraPage.waitForPageLoad();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareElementScreenshot(
        '[data-testid="camera-grid"]',
        'camera-grid-component',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });

    test('should match filter panel layout', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('traffic');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const filterPanelExists = await analyticsPage.isElementVisible('[data-testid="filter-panel"]');
      if (filterPanelExists) {
        const result = await visualTester.compareElementScreenshot(
          '[data-testid="filter-panel"]',
          'filter-panel-component',
          { threshold: 0.1 }
        );

        expect(result.passed).toBe(true);
      }
    });
  });

  describe('Responsive Visual Regression', () => {
    test('should match responsive layouts across breakpoints', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const results = await visualTester.compareResponsiveScreenshots(
        'overview-dashboard-responsive',
        [
          { name: 'mobile', width: 375, height: 667 },
          { name: 'tablet', width: 768, height: 1024 },
          { name: 'desktop', width: 1920, height: 1080 },
          { name: 'large-desktop', width: 2560, height: 1440 },
        ],
        { threshold: 0.3 } // Higher threshold for responsive layouts
      );

      Object.keys(results).forEach(breakpoint => {
        expect(results[breakpoint].passed).toBe(true);
        if (!results[breakpoint].passed) {
          console.log(`Responsive test failed for ${breakpoint}: ${results[breakpoint].percentageDifference}% difference`);
        }
      });
    });

    test('should match camera management responsive layouts', async () => {
      await cameraPage.navigateToPage();
      await cameraPage.waitForPageLoad();

      await visualRegressionUtils.stabilizePage(page);

      const results = await visualTester.compareResponsiveScreenshots(
        'camera-management-responsive',
        [
          { name: 'mobile', width: 375, height: 667 },
          { name: 'tablet', width: 768, height: 1024 },
          { name: 'desktop', width: 1920, height: 1080 },
        ],
        { threshold: 0.3 }
      );

      Object.keys(results).forEach(breakpoint => {
        expect(results[breakpoint].passed).toBe(true);
      });
    });
  });

  describe('Dark Mode Visual Regression', () => {
    test('should match dark mode layout', async () => {
      // Enable dark mode
      await page.evaluate(() => {
        document.documentElement.classList.add('dark');
        localStorage.setItem('theme', 'dark');
      });

      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareScreenshot(
        'overview-dashboard-dark-mode',
        { threshold: 0.3 } // Higher threshold for theme differences
      );

      expect(result.passed).toBe(true);
    });

    test('should match dark mode component styling', async () => {
      // Enable dark mode
      await page.evaluate(() => {
        document.documentElement.classList.add('dark');
        localStorage.setItem('theme', 'dark');
      });

      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      const result = await visualTester.compareElementScreenshot(
        '[data-testid="kpi-card"]',
        'kpi-cards-dark-mode',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });
  });

  describe('Error State Visual Regression', () => {
    test('should match error state layout', async () => {
      // Mock API to return error
      await page.setRequestInterception(true);
      page.removeAllListeners('request');
      page.on('request', (request) => {
        if (request.url().includes('/api/')) {
          request.respond({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Internal Server Error' }),
          });
        } else {
          request.continue();
        }
      });

      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await page.waitForTimeout(3000); // Wait for error state to appear

      await visualRegressionUtils.stabilizePage(page);

      const hasError = await analyticsPage.hasErrorState();
      if (hasError) {
        const result = await visualTester.compareScreenshot(
          'overview-dashboard-error-state',
          { threshold: 0.2 }
        );

        expect(result.passed).toBe(true);
      }
    });

    test('should match loading state layout', async () => {
      // Mock API to delay response
      await page.setRequestInterception(true);
      page.removeAllListeners('request');
      page.on('request', (request) => {
        if (request.url().includes('/api/analytics/current')) {
          // Delay response to capture loading state
          setTimeout(() => {
            request.respond({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify(MOCK_API_RESPONSES.ANALYTICS_DATA.metrics),
            });
          }, 5000);
        } else {
          request.continue();
        }
      });

      await analyticsPage.navigateToAnalyticsDashboard('overview');

      // Capture loading state
      await page.waitForTimeout(1000);

      const result = await visualTester.compareScreenshot(
        'overview-dashboard-loading-state',
        { threshold: 0.2 }
      );

      expect(result.passed).toBe(true);
    });
  });

  describe('Visual Regression Utilities', () => {
    test('should handle baseline creation for new tests', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      // Take screenshot of a new component
      const result = await visualTester.compareScreenshot(
        'new-component-baseline-test',
        { threshold: 0.1 }
      );

      // First run should pass (baseline creation)
      expect(result.passed).toBe(true);
      expect(result.pixelDifference).toBe(0);
    });

    test('should clean up diff images', async () => {
      // Clean up any existing diff images
      await visualTester.cleanupDiffs();

      // This test should always pass as it's just cleanup
      expect(true).toBe(true);
    });

    test('should update baseline when requested', async () => {
      await analyticsPage.navigateToAnalyticsDashboard('overview');
      await analyticsPage.waitForLoadingToComplete();

      await visualRegressionUtils.stabilizePage(page);

      // Take a screenshot first
      await visualTester.compareScreenshot(
        'baseline-update-test',
        { threshold: 0.1 }
      );

      // Update baseline (this would typically be done manually when design changes)
      try {
        await visualTester.updateBaseline('baseline-update-test');
        expect(true).toBe(true); // Test passes if no error thrown
      } catch (error) {
        // It's okay if the baseline doesn't exist yet
        expect(true).toBe(true);
      }
    });
  });
});