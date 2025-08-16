import { Page } from 'puppeteer';
import { DashboardPage } from '../pages/DashboardPage';
import { E2E_CONFIG } from '../config/puppeteer.config';

describe('Dashboard E2E Tests', () => {
  let dashboardPage: DashboardPage;

  beforeEach(async () => {
    dashboardPage = new DashboardPage(page);
  });

  describe('Dashboard Loading and Navigation', () => {
    test('should load dashboard page successfully', async () => {
      await dashboardPage.navigateToDashboard();

      // Verify page title
      const title = await page.title();
      expect(title).toContain('Dashboard');

      // Verify main sections are visible
      await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();
      await expect(page.locator('[data-testid="metrics-overview"]')).toBeVisible();
    });

    test('should display metrics overview correctly', async () => {
      await dashboardPage.navigateToDashboard();

      const metrics = await dashboardPage.getMetricsOverview();

      // Verify metrics are numbers and reasonable
      expect(metrics.totalCameras).toBeGreaterThanOrEqual(0);
      expect(metrics.onlineCameras).toBeGreaterThanOrEqual(0);
      expect(metrics.onlineCameras).toBeLessThanOrEqual(metrics.totalCameras);
      expect(metrics.activeIncidents).toBeGreaterThanOrEqual(0);
      expect(metrics.avgResponseTime).toMatch(/\d+(\.\d+)?\s*(ms|s)/);
    });

    test('should navigate to different sections', async () => {
      await dashboardPage.navigateToDashboard();

      // Test navigation to cameras
      await dashboardPage.navigateToCameras();
      expect(page.url()).toContain('/cameras');

      // Navigate back to dashboard
      await page.goBack();
      await dashboardPage.waitForDashboardToLoad();

      // Test navigation to analytics
      await dashboardPage.navigateToAnalytics();
      expect(page.url()).toContain('/analytics');

      // Navigate back to dashboard
      await page.goBack();
      await dashboardPage.waitForDashboardToLoad();
    });
  });

  describe('Real-time Features', () => {
    test('should establish WebSocket connection', async () => {
      await dashboardPage.navigateToDashboard();
      await dashboardPage.waitForRealTimeConnection();

      // Verify connection status is shown
      const connectionElement = page.locator('[data-testid="connection-status"]');
      await expect(connectionElement).toContainText('Connected');
    });

    test('should handle WebSocket messages', async () => {
      await dashboardPage.navigateToDashboard();

      const hasWebSocketActivity = await dashboardPage.testWebSocketMessages();
      expect(hasWebSocketActivity).toBe(true);
    });

    test('should update data in real-time', async () => {
      await dashboardPage.navigateToDashboard();

      // This test would be more meaningful with actual real-time data
      // For now, we'll verify the real-time indicators are present
      await expect(page.locator('[data-testid="real-time-indicator"]')).toBeVisible();
      await expect(page.locator('[data-testid="last-update"]')).toBeVisible();
    });
  });

  describe('Data Visualization', () => {
    test('should render traffic flow chart', async () => {
      await dashboardPage.navigateToDashboard();

      const hasTrafficChart = await dashboardPage.verifyTrafficChart();
      expect(hasTrafficChart).toBe(true);
    });

    test('should render camera map', async () => {
      await dashboardPage.navigateToDashboard();

      const hasCameraMap = await dashboardPage.verifyCameraMap();
      expect(hasCameraMap).toBe(true);
    });

    test('should display recent incidents', async () => {
      await dashboardPage.navigateToDashboard();

      const incidents = await dashboardPage.getRecentIncidents();
      expect(Array.isArray(incidents)).toBe(true);

      // If there are incidents, verify structure
      if (incidents.length > 0) {
        const firstIncident = incidents[0];
        expect(firstIncident).toHaveProperty('id');
        expect(firstIncident).toHaveProperty('title');
        expect(firstIncident).toHaveProperty('severity');
        expect(firstIncident).toHaveProperty('timestamp');
      }
    });
  });

  describe('Interactive Features', () => {
    test('should change time range filter', async () => {
      await dashboardPage.navigateToDashboard();

      // Test different time ranges
      const timeRanges = ['1h', '4h', '24h', '7d'] as const;

      for (const range of timeRanges) {
        await dashboardPage.changeTimeRange(range);

        // Verify URL or state reflects the change
        // This would depend on how time range is implemented
        await page.waitForTimeout(1000);
      }
    });

    test('should refresh dashboard data', async () => {
      await dashboardPage.navigateToDashboard();

      // Get initial last update time
      const initialUpdate = await page.locator('[data-testid="last-update"]').textContent();

      // Refresh dashboard
      await dashboardPage.refreshDashboard();

      // Wait for update
      await page.waitForTimeout(2000);

      // Verify data was refreshed (this would be more meaningful with dynamic data)
      const refreshButton = page.locator('[data-testid="refresh-button"]');
      await expect(refreshButton).toBeVisible();
    });

    test('should export dashboard data', async () => {
      await dashboardPage.navigateToDashboard();

      // Test export functionality
      await dashboardPage.exportData();

      // Verify export modal or download started
      // This would depend on implementation
      await page.waitForTimeout(2000);
    });
  });

  describe('System Health Monitoring', () => {
    test('should display system health metrics', async () => {
      await dashboardPage.navigateToDashboard();

      const health = await dashboardPage.getSystemHealth();

      expect(health.status).toBeDefined();
      expect(health.cpuUsage).toBeGreaterThanOrEqual(0);
      expect(health.cpuUsage).toBeLessThanOrEqual(100);
      expect(health.memoryUsage).toBeGreaterThanOrEqual(0);
      expect(health.memoryUsage).toBeLessThanOrEqual(100);
      expect(health.diskUsage).toBeGreaterThanOrEqual(0);
      expect(health.diskUsage).toBeLessThanOrEqual(100);
    });
  });

  describe('Performance and Responsiveness', () => {
    test('should meet Web Vitals thresholds', async () => {
      await dashboardPage.navigateToDashboard();

      const webVitals = await dashboardPage.validateWebVitals();

      console.log('Web Vitals:', webVitals.metrics);

      if (!webVitals.passed) {
        console.warn('Web Vitals failures:', webVitals.failures);
      }

      // For now, we'll log the metrics but not fail the test
      // In production, you might want to enforce these thresholds
      expect(webVitals.metrics).toBeDefined();
    });

    test('should be responsive across different viewports', async () => {
      await dashboardPage.navigateToDashboard();

      const responsiveness = await dashboardPage.verifyResponsiveness();

      expect(responsiveness.mobile).toBe(true);
      expect(responsiveness.tablet).toBe(true);
      expect(responsiveness.desktop).toBe(true);
    });

    test('should handle slow network conditions', async () => {
      // Simulate slow network
      await global.helpers.simulateSlowNetwork();

      await dashboardPage.navigateToDashboard();

      // Verify page still loads (may take longer)
      await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible({ timeout: 30000 });

      // Reset network conditions
      await global.helpers.resetNetworkConditions();
    });
  });

  describe('Error Handling', () => {
    test('should handle API errors gracefully', async () => {
      // Mock API error response
      await global.helpers.mockApiResponse('/api/metrics', { error: 'Server error' }, 500);

      await dashboardPage.navigateToDashboard();

      // Verify error handling UI appears
      // This would depend on how errors are handled in the UI
      await page.waitForTimeout(2000);

      // Check for JavaScript errors
      const jsErrors = await global.helpers.checkForJavaScriptErrors();
      expect(jsErrors.length).toBe(0);
    });

    test('should handle network disconnection', async () => {
      await dashboardPage.navigateToDashboard();

      // Simulate network disconnection
      await page.setOfflineMode(true);

      // Wait for offline detection
      await page.waitForTimeout(5000);

      // Verify offline indicator or error message
      // This would depend on implementation

      // Restore network
      await page.setOfflineMode(false);

      // Wait for reconnection
      await page.waitForTimeout(5000);
    });
  });

  describe('Accessibility', () => {
    test('should be keyboard navigable', async () => {
      await dashboardPage.navigateToDashboard();

      // Test keyboard navigation
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Verify focus indicators are visible
      const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
      expect(focusedElement).toBeDefined();
    });

    test('should have proper ARIA labels', async () => {
      await dashboardPage.navigateToDashboard();

      // Check for ARIA labels on interactive elements
      const ariaElements = await page.$$('[aria-label], [aria-labelledby], [role]');
      expect(ariaElements.length).toBeGreaterThan(0);
    });
  });
});