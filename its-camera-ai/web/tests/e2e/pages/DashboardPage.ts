import { Page } from 'puppeteer';
import { BasePage } from './BasePage';

export class DashboardPage extends BasePage {
  // Page selectors
  private selectors = {
    // Navigation
    header: '[data-testid="dashboard-header"]',
    navigation: '[data-testid="main-navigation"]',
    userMenu: '[data-testid="user-menu"]',

    // Dashboard sections
    metricsOverview: '[data-testid="metrics-overview"]',
    cameraStatus: '[data-testid="camera-status"]',
    recentIncidents: '[data-testid="recent-incidents"]',
    trafficFlow: '[data-testid="traffic-flow"]',
    systemHealth: '[data-testid="system-health"]',

    // Metrics cards
    totalCameras: '[data-testid="metric-total-cameras"]',
    onlineCameras: '[data-testid="metric-online-cameras"]',
    activeIncidents: '[data-testid="metric-active-incidents"]',
    avgResponseTime: '[data-testid="metric-avg-response-time"]',

    // Charts and visualizations
    trafficChart: '[data-testid="traffic-chart"]',
    cameraMap: '[data-testid="camera-map"]',
    incidentTimeline: '[data-testid="incident-timeline"]',

    // Navigation links
    camerasLink: '[data-testid="nav-cameras"]',
    analyticsLink: '[data-testid="nav-analytics"]',
    incidentsLink: '[data-testid="nav-incidents"]',
    settingsLink: '[data-testid="nav-settings"]',

    // Real-time indicators
    connectionStatus: '[data-testid="connection-status"]',
    lastUpdate: '[data-testid="last-update"]',
    realTimeIndicator: '[data-testid="real-time-indicator"]',

    // Filters and controls
    timeRangeSelector: '[data-testid="time-range-selector"]',
    refreshButton: '[data-testid="refresh-button"]',
    exportButton: '[data-testid="export-button"]',

    // Loading states
    loadingSpinner: '[data-testid="loading-spinner"]',
    skeletonLoader: '[data-testid="skeleton-loader"]',
  } as const;

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to the dashboard
   */
  async navigateToDashboard(): Promise<void> {
    await this.navigateTo('/en/dashboard');
    await this.waitForDashboardToLoad();
  }

  /**
   * Wait for dashboard to fully load
   */
  async waitForDashboardToLoad(): Promise<void> {
    await this.waitForElement(this.selectors.header);
    await this.waitForElement(this.selectors.metricsOverview);
    await this.waitForLoadingToComplete();

    // Wait for charts to render
    await this.waitForElement(this.selectors.trafficChart);

    // Wait for real-time connection
    await this.waitForRealTimeConnection();
  }

  /**
   * Wait for real-time WebSocket connection
   */
  async waitForRealTimeConnection(): Promise<void> {
    // Wait for connection status to show connected
    await this.page.waitForFunction(
      (selector) => {
        const element = document.querySelector(selector);
        return element && element.textContent?.includes('Connected');
      },
      { timeout: 10000 },
      this.selectors.connectionStatus
    );
  }

  /**
   * Get metrics overview data
   */
  async getMetricsOverview(): Promise<{
    totalCameras: number;
    onlineCameras: number;
    activeIncidents: number;
    avgResponseTime: string;
  }> {
    await this.waitForElement(this.selectors.metricsOverview);

    const totalCameras = parseInt(await this.getElementText(this.selectors.totalCameras));
    const onlineCameras = parseInt(await this.getElementText(this.selectors.onlineCameras));
    const activeIncidents = parseInt(await this.getElementText(this.selectors.activeIncidents));
    const avgResponseTime = await this.getElementText(this.selectors.avgResponseTime);

    return {
      totalCameras,
      onlineCameras,
      activeIncidents,
      avgResponseTime,
    };
  }

  /**
   * Check if real-time updates are working
   */
  async verifyRealTimeUpdates(): Promise<boolean> {
    // Get initial last update time
    const initialUpdate = await this.getElementText(this.selectors.lastUpdate);

    // Wait for update (real-time updates should happen every 5-30 seconds)
    await this.page.waitForTimeout(35000);

    // Get new last update time
    const newUpdate = await this.getElementText(this.selectors.lastUpdate);

    return initialUpdate !== newUpdate;
  }

  /**
   * Navigate to cameras section
   */
  async navigateToCameras(): Promise<void> {
    await this.clickElement(this.selectors.camerasLink, true);
  }

  /**
   * Navigate to analytics section
   */
  async navigateToAnalytics(): Promise<void> {
    await this.clickElement(this.selectors.analyticsLink, true);
  }

  /**
   * Navigate to incidents section
   */
  async navigateToIncidents(): Promise<void> {
    await this.clickElement(this.selectors.incidentsLink, true);
  }

  /**
   * Change time range filter
   */
  async changeTimeRange(range: '1h' | '4h' | '24h' | '7d'): Promise<void> {
    await this.clickElement(this.selectors.timeRangeSelector);

    // Wait for dropdown to appear and select option
    const optionSelector = `[data-testid="time-range-${range}"]`;
    await this.waitForElement(optionSelector);
    await this.clickElement(optionSelector);

    // Wait for data to refresh
    await this.waitForLoadingToComplete();
  }

  /**
   * Refresh dashboard data
   */
  async refreshDashboard(): Promise<void> {
    await this.clickElement(this.selectors.refreshButton);
    await this.waitForLoadingToComplete();
  }

  /**
   * Export dashboard data
   */
  async exportData(): Promise<void> {
    await this.clickElement(this.selectors.exportButton);

    // Wait for export modal or download to start
    await this.page.waitForTimeout(2000);
  }

  /**
   * Check traffic chart data
   */
  async verifyTrafficChart(): Promise<boolean> {
    await this.waitForElement(this.selectors.trafficChart);

    // Check if chart has data points
    const hasData = await this.page.evaluate((selector) => {
      const chart = document.querySelector(selector);
      if (!chart) return false;

      // Look for chart elements (depends on chart library)
      const dataElements = chart.querySelectorAll('[data-testid*="chart-data"], .recharts-line, .recharts-bar');
      return dataElements.length > 0;
    }, this.selectors.trafficChart);

    return hasData;
  }

  /**
   * Check camera map functionality
   */
  async verifyCameraMap(): Promise<boolean> {
    await this.waitForElement(this.selectors.cameraMap);

    // Check if map has camera markers
    const hasMarkers = await this.page.evaluate((selector) => {
      const map = document.querySelector(selector);
      if (!map) return false;

      // Look for camera markers
      const markers = map.querySelectorAll('[data-testid*="camera-marker"], .camera-marker');
      return markers.length > 0;
    }, this.selectors.cameraMap);

    return hasMarkers;
  }

  /**
   * Get recent incidents list
   */
  async getRecentIncidents(): Promise<Array<{
    id: string;
    title: string;
    severity: string;
    timestamp: string;
  }>> {
    await this.waitForElement(this.selectors.recentIncidents);

    const incidents = await this.page.evaluate((selector) => {
      const container = document.querySelector(selector);
      if (!container) return [];

      const incidentElements = container.querySelectorAll('[data-testid*="incident-item"]');
      return Array.from(incidentElements).map((element) => {
        const id = element.getAttribute('data-incident-id') || '';
        const title = element.querySelector('[data-testid="incident-title"]')?.textContent || '';
        const severity = element.querySelector('[data-testid="incident-severity"]')?.textContent || '';
        const timestamp = element.querySelector('[data-testid="incident-timestamp"]')?.textContent || '';

        return { id, title, severity, timestamp };
      });
    }, this.selectors.recentIncidents);

    return incidents;
  }

  /**
   * Check system health status
   */
  async getSystemHealth(): Promise<{
    status: string;
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
  }> {
    await this.waitForElement(this.selectors.systemHealth);

    const health = await this.page.evaluate((selector) => {
      const container = document.querySelector(selector);
      if (!container) return null;

      const status = container.querySelector('[data-testid="system-status"]')?.textContent || '';
      const cpuText = container.querySelector('[data-testid="cpu-usage"]')?.textContent || '0%';
      const memoryText = container.querySelector('[data-testid="memory-usage"]')?.textContent || '0%';
      const diskText = container.querySelector('[data-testid="disk-usage"]')?.textContent || '0%';

      return {
        status,
        cpuUsage: parseInt(cpuText.replace('%', '')),
        memoryUsage: parseInt(memoryText.replace('%', '')),
        diskUsage: parseInt(diskText.replace('%', '')),
      };
    }, this.selectors.systemHealth);

    return health || { status: '', cpuUsage: 0, memoryUsage: 0, diskUsage: 0 };
  }

  /**
   * Verify dashboard responsiveness
   */
  async verifyResponsiveness(): Promise<{ mobile: boolean; tablet: boolean; desktop: boolean }> {
    const results = { mobile: false, tablet: false, desktop: false };

    // Test mobile viewport
    await this.page.setViewport({ width: 375, height: 667 });
    await this.page.waitForTimeout(1000);
    results.mobile = await this.isElementVisible(this.selectors.header);

    // Test tablet viewport
    await this.page.setViewport({ width: 768, height: 1024 });
    await this.page.waitForTimeout(1000);
    results.tablet = await this.isElementVisible(this.selectors.header);

    // Test desktop viewport
    await this.page.setViewport({ width: 1920, height: 1080 });
    await this.page.waitForTimeout(1000);
    results.desktop = await this.isElementVisible(this.selectors.header);

    return results;
  }

  /**
   * Test WebSocket message handling
   */
  async testWebSocketMessages(): Promise<boolean> {
    await this.setupWebSocketInterception();
    await this.clearWebSocketMessages();

    // Wait for some WebSocket activity
    await this.page.waitForTimeout(10000);

    const messages = await this.getWebSocketMessages();
    return messages.length > 0;
  }

  /**
   * Logout from dashboard
   */
  async logout(): Promise<void> {
    await this.clickElement(this.selectors.userMenu);

    // Wait for dropdown and click logout
    const logoutSelector = '[data-testid="logout-button"]';
    await this.waitForElement(logoutSelector);
    await this.clickElement(logoutSelector, true);
  }
}