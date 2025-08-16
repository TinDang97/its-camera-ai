import { Page } from 'puppeteer';
import { BasePage } from './BasePage';

export interface AnalyticsMetrics {
  vehicleCount: number;
  averageSpeed: number;
  congestionLevel: 'low' | 'medium' | 'high';
  incidentCount: number;
  timestamp: string;
}

export interface ChartDataPoint {
  time: string;
  value: number;
  label?: string;
}

export class AnalyticsDashboardPage extends BasePage {
  private selectors = {
    // Main dashboard navigation
    dashboardNav: '[data-testid="dashboard-nav"]',
    overviewTab: '[data-testid="overview-tab"], [href*="/dashboard/overview"]',
    trafficTab: '[data-testid="traffic-tab"], [href*="/dashboard/traffic"]',
    incidentsTab: '[data-testid="incidents-tab"], [href*="/dashboard/incidents"]',
    camerasTab: '[data-testid="cameras-tab"], [href*="/dashboard/cameras"]',
    mapsTab: '[data-testid="maps-tab"], [href*="/dashboard/maps"]',
    reportsTab: '[data-testid="reports-tab"], [href*="/dashboard/reports"]',

    // Overview Dashboard Elements
    overviewContainer: '[data-testid="overview-dashboard"]',
    kpiCards: '[data-testid="kpi-card"]',
    totalVehiclesCard: '[data-testid="total-vehicles-kpi"]',
    averageSpeedCard: '[data-testid="average-speed-kpi"]',
    activeIncidentsCard: '[data-testid="active-incidents-kpi"]',
    systemHealthCard: '[data-testid="system-health-kpi"]',

    // Traffic Dashboard Elements
    trafficContainer: '[data-testid="traffic-dashboard"]',
    trafficFlowChart: '[data-testid="traffic-flow-chart"]',
    speedDistributionChart: '[data-testid="speed-distribution-chart"]',
    volumeByHourChart: '[data-testid="volume-by-hour-chart"]',
    congestionHeatmap: '[data-testid="congestion-heatmap"]',

    // Incidents Dashboard Elements
    incidentsContainer: '[data-testid="incidents-dashboard"]',
    incidentsList: '[data-testid="incidents-list"]',
    incidentItem: '[data-testid="incident-item"]',
    incidentSeverityChart: '[data-testid="incident-severity-chart"]',
    incidentTrendsChart: '[data-testid="incident-trends-chart"]',
    resolvedIncidentsChart: '[data-testid="resolved-incidents-chart"]',

    // Cameras Dashboard Elements
    camerasContainer: '[data-testid="cameras-dashboard"]',
    cameraStatusOverview: '[data-testid="camera-status-overview"]',
    cameraGrid: '[data-testid="camera-grid"]',
    cameraHealthChart: '[data-testid="camera-health-chart"]',
    uptimeChart: '[data-testid="uptime-chart"]',

    // Maps Dashboard Elements
    mapsContainer: '[data-testid="maps-dashboard"]',
    mapVisualization: '[data-testid="map-visualization"]',
    mapControls: '[data-testid="map-controls"]',
    layerControls: '[data-testid="layer-controls"]',
    trafficLayer: '[data-testid="traffic-layer-toggle"]',
    incidentsLayer: '[data-testid="incidents-layer-toggle"]',
    camerasLayer: '[data-testid="cameras-layer-toggle"]',

    // Reports Dashboard Elements
    reportsContainer: '[data-testid="reports-dashboard"]',
    reportsList: '[data-testid="reports-list"]',
    generateReportBtn: '[data-testid="generate-report-btn"]',
    reportFilters: '[data-testid="report-filters"]',
    downloadReportBtn: '[data-testid="download-report-btn"]',

    // Charts and Visualizations
    chart: '[data-testid*="chart"]',
    chartContainer: '.chart-container, [data-testid="chart-container"]',
    chartLegend: '[data-testid="chart-legend"]',
    chartTooltip: '[data-testid="chart-tooltip"]',
    chartTitle: '[data-testid="chart-title"]',

    // Time Controls
    timeRangeSelector: '[data-testid="time-range-selector"]',
    timePicker: '[data-testid="time-picker"]',
    refreshButton: '[data-testid="refresh-data"]',
    autoRefreshToggle: '[data-testid="auto-refresh-toggle"]',
    lastUpdated: '[data-testid="last-updated"]',

    // Filters and Controls
    filterPanel: '[data-testid="filter-panel"]',
    cameraFilter: '[data-testid="camera-filter"]',
    locationFilter: '[data-testid="location-filter"]',
    severityFilter: '[data-testid="severity-filter"]',
    statusFilter: '[data-testid="status-filter"]',
    clearFiltersBtn: '[data-testid="clear-filters"]',

    // Export and Actions
    exportButton: '[data-testid="export-data"]',
    exportOptions: '[data-testid="export-options"]',
    shareButton: '[data-testid="share-dashboard"]',
    fullscreenButton: '[data-testid="fullscreen-view"]',

    // Loading and Error States
    loadingSpinner: '[data-testid="loading"], .loading, .spinner',
    errorMessage: '[data-testid="error-message"]',
    emptyState: '[data-testid="empty-state"]',
    noDataMessage: '[data-testid="no-data-message"]',

    // Real-time Indicators
    liveIndicator: '[data-testid="live-indicator"]',
    dataUpdateIndicator: '[data-testid="data-updating"]',
    connectionStatus: '[data-testid="connection-status"]',

    // Tooltips and Help
    helpButton: '[data-testid="help-button"]',
    tooltip: '[data-testid="tooltip"]',
    infoPanel: '[data-testid="info-panel"]',
  } as const;

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to specific analytics dashboard
   */
  async navigateToAnalyticsDashboard(dashboard: 'overview' | 'traffic' | 'incidents' | 'cameras' | 'maps' | 'reports'): Promise<void> {
    const tabSelectors = {
      overview: this.selectors.overviewTab,
      traffic: this.selectors.trafficTab,
      incidents: this.selectors.incidentsTab,
      cameras: this.selectors.camerasTab,
      maps: this.selectors.mapsTab,
      reports: this.selectors.reportsTab,
    };

    await this.clickElement(tabSelectors[dashboard]);
    await this.waitForDashboardToLoad(dashboard);
  }

  /**
   * Wait for specific dashboard to load
   */
  async waitForDashboardToLoad(dashboard: string): Promise<void> {
    const containerSelectors = {
      overview: this.selectors.overviewContainer,
      traffic: this.selectors.trafficContainer,
      incidents: this.selectors.incidentsContainer,
      cameras: this.selectors.camerasContainer,
      maps: this.selectors.mapsContainer,
      reports: this.selectors.reportsContainer,
    };

    const selector = containerSelectors[dashboard as keyof typeof containerSelectors];
    if (selector) {
      await this.waitForElement(selector);
    }

    // Wait for loading to complete
    await this.waitForLoadingToComplete();
  }

  /**
   * Wait for loading to complete
   */
  async waitForLoadingToComplete(): Promise<void> {
    try {
      await this.page.waitForFunction(
        (loadingSelector) => !document.querySelector(loadingSelector),
        { timeout: 15000 },
        this.selectors.loadingSpinner
      );
    } catch {
      // Loading might not appear or might be very fast
    }
  }

  /**
   * Get KPI metrics from overview dashboard
   */
  async getKPIMetrics(): Promise<{
    totalVehicles: number;
    averageSpeed: number;
    activeIncidents: number;
    systemHealth: string;
  }> {
    const totalVehicles = parseInt(await this.getElementText(this.selectors.totalVehiclesCard) || '0');
    const averageSpeed = parseFloat(await this.getElementText(this.selectors.averageSpeedCard) || '0');
    const activeIncidents = parseInt(await this.getElementText(this.selectors.activeIncidentsCard) || '0');
    const systemHealth = await this.getElementText(this.selectors.systemHealthCard) || 'unknown';

    return {
      totalVehicles,
      averageSpeed,
      activeIncidents,
      systemHealth,
    };
  }

  /**
   * Check if charts are loaded and displaying data
   */
  async areChartsLoaded(): Promise<boolean> {
    const charts = await this.page.$$(this.selectors.chart);

    if (charts.length === 0) {
      return false;
    }

    // Check if charts have data (look for chart elements like bars, lines, etc.)
    for (const chart of charts) {
      const hasData = await chart.evaluate(el => {
        // Look for common chart elements
        const svgElements = el.querySelectorAll('svg, canvas');
        const chartElements = el.querySelectorAll('.recharts-surface, .chart-data, .highcharts-container');

        return svgElements.length > 0 || chartElements.length > 0;
      });

      if (!hasData) {
        return false;
      }
    }

    return true;
  }

  /**
   * Get chart data points (if accessible)
   */
  async getChartData(chartSelector: string): Promise<ChartDataPoint[]> {
    const chartElement = await this.page.$(chartSelector);

    if (!chartElement) {
      return [];
    }

    return await chartElement.evaluate(el => {
      // Try to extract data from common chart libraries
      const dataPoints: ChartDataPoint[] = [];

      // For Recharts or similar libraries
      const rechartData = el.querySelectorAll('.recharts-bar, .recharts-line-dot, .recharts-area-dot');
      rechartData.forEach((point, index) => {
        const value = point.getAttribute('value') || point.getAttribute('payload');
        if (value) {
          dataPoints.push({
            time: `Point ${index}`,
            value: parseFloat(value),
          });
        }
      });

      // For Chart.js or canvas-based charts
      const canvas = el.querySelector('canvas');
      if (canvas && (canvas as any).chart) {
        try {
          const chartData = (canvas as any).chart.data;
          if (chartData && chartData.datasets) {
            chartData.datasets.forEach((dataset: any) => {
              dataset.data.forEach((value: number, index: number) => {
                dataPoints.push({
                  time: chartData.labels?.[index] || `Point ${index}`,
                  value: value,
                  label: dataset.label,
                });
              });
            });
          }
        } catch (error) {
          console.warn('Error extracting Chart.js data:', error);
        }
      }

      return dataPoints;
    });
  }

  /**
   * Set time range for analytics
   */
  async setTimeRange(range: '1h' | '24h' | '7d' | '30d' | 'custom'): Promise<void> {
    const timeRangeSelector = await this.page.$(this.selectors.timeRangeSelector);

    if (timeRangeSelector) {
      await timeRangeSelector.click();

      // Select time range option
      const optionSelector = `[data-value="${range}"], option[value="${range}"]`;
      await this.page.waitForSelector(optionSelector, { timeout: 5000 });
      await this.clickElement(optionSelector);

      // Wait for data to reload
      await this.waitForLoadingToComplete();
    }
  }

  /**
   * Refresh dashboard data
   */
  async refreshData(): Promise<void> {
    await this.clickElement(this.selectors.refreshButton);
    await this.waitForLoadingToComplete();
  }

  /**
   * Toggle auto-refresh
   */
  async toggleAutoRefresh(): Promise<void> {
    const toggle = await this.page.$(this.selectors.autoRefreshToggle);
    if (toggle) {
      await toggle.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Apply filters
   */
  async applyFilters(filters: {
    camera?: string;
    location?: string;
    severity?: string;
    status?: string;
  }): Promise<void> {
    if (filters.camera) {
      await this.selectOption(this.selectors.cameraFilter, filters.camera);
    }

    if (filters.location) {
      await this.selectOption(this.selectors.locationFilter, filters.location);
    }

    if (filters.severity) {
      await this.selectOption(this.selectors.severityFilter, filters.severity);
    }

    if (filters.status) {
      await this.selectOption(this.selectors.statusFilter, filters.status);
    }

    // Wait for filters to apply
    await this.waitForLoadingToComplete();
  }

  /**
   * Clear all filters
   */
  async clearFilters(): Promise<void> {
    const clearButton = await this.page.$(this.selectors.clearFiltersBtn);
    if (clearButton) {
      await clearButton.click();
      await this.waitForLoadingToComplete();
    }
  }

  /**
   * Export dashboard data
   */
  async exportData(format: 'csv' | 'pdf' | 'excel'): Promise<void> {
    await this.clickElement(this.selectors.exportButton);

    // Wait for export options to appear
    await this.waitForElement(this.selectors.exportOptions);

    // Select format
    const formatSelector = `[data-format="${format}"], [value="${format}"]`;
    await this.clickElement(formatSelector);

    // Wait for download to start
    await this.page.waitForTimeout(2000);
  }

  /**
   * Get incidents list from incidents dashboard
   */
  async getIncidentsList(): Promise<Array<{
    id: string;
    type: string;
    severity: string;
    location: string;
    timestamp: string;
    status: string;
  }>> {
    const incidents: Array<{
      id: string;
      type: string;
      severity: string;
      location: string;
      timestamp: string;
      status: string;
    }> = [];

    const incidentElements = await this.page.$$(this.selectors.incidentItem);

    for (const element of incidentElements) {
      try {
        const id = await element.evaluate(el => el.getAttribute('data-incident-id') || '');
        const type = await element.evaluate(el =>
          el.querySelector('[data-testid="incident-type"]')?.textContent || ''
        );
        const severity = await element.evaluate(el =>
          el.querySelector('[data-testid="incident-severity"]')?.textContent || ''
        );
        const location = await element.evaluate(el =>
          el.querySelector('[data-testid="incident-location"]')?.textContent || ''
        );
        const timestamp = await element.evaluate(el =>
          el.querySelector('[data-testid="incident-timestamp"]')?.textContent || ''
        );
        const status = await element.evaluate(el =>
          el.querySelector('[data-testid="incident-status"]')?.textContent || ''
        );

        incidents.push({ id, type, severity, location, timestamp, status });
      } catch (error) {
        console.warn('Error parsing incident data:', error);
      }
    }

    return incidents;
  }

  /**
   * Get camera status overview
   */
  async getCameraStatusOverview(): Promise<{
    total: number;
    online: number;
    offline: number;
    maintenance: number;
  }> {
    const overview = await this.page.$(this.selectors.cameraStatusOverview);

    if (!overview) {
      return { total: 0, online: 0, offline: 0, maintenance: 0 };
    }

    return await overview.evaluate(el => {
      const total = parseInt(el.querySelector('[data-testid="total-cameras"]')?.textContent || '0');
      const online = parseInt(el.querySelector('[data-testid="online-cameras"]')?.textContent || '0');
      const offline = parseInt(el.querySelector('[data-testid="offline-cameras"]')?.textContent || '0');
      const maintenance = parseInt(el.querySelector('[data-testid="maintenance-cameras"]')?.textContent || '0');

      return { total, online, offline, maintenance };
    });
  }

  /**
   * Check if map is loaded (for maps dashboard)
   */
  async isMapLoaded(): Promise<boolean> {
    const mapElement = await this.page.$(this.selectors.mapVisualization);

    if (!mapElement) {
      return false;
    }

    return await mapElement.evaluate(el => {
      // Check for common map indicators
      const mapContainer = el.querySelector('.mapboxgl-map, .leaflet-container, .gm-style');
      const mapTiles = el.querySelectorAll('.mapboxgl-canvas, .leaflet-tile, img[src*="maps"]');

      return mapContainer !== null && mapTiles.length > 0;
    });
  }

  /**
   * Toggle map layers
   */
  async toggleMapLayer(layer: 'traffic' | 'incidents' | 'cameras'): Promise<void> {
    const layerSelectors = {
      traffic: this.selectors.trafficLayer,
      incidents: this.selectors.incidentsLayer,
      cameras: this.selectors.camerasLayer,
    };

    const selector = layerSelectors[layer];
    await this.clickElement(selector);
    await this.page.waitForTimeout(1000);
  }

  /**
   * Generate report (for reports dashboard)
   */
  async generateReport(reportType: string, filters?: any): Promise<void> {
    // Apply filters if provided
    if (filters) {
      await this.applyFilters(filters);
    }

    // Click generate report button
    await this.clickElement(this.selectors.generateReportBtn);

    // Wait for report generation
    await this.waitForLoadingToComplete();
  }

  /**
   * Download generated report
   */
  async downloadReport(): Promise<void> {
    await this.clickElement(this.selectors.downloadReportBtn);
    await this.page.waitForTimeout(2000);
  }

  /**
   * Check if dashboard has real-time updates
   */
  async hasRealTimeUpdates(): Promise<boolean> {
    const liveIndicator = await this.page.$(this.selectors.liveIndicator);
    const connectionStatus = await this.page.$(this.selectors.connectionStatus);

    if (liveIndicator) {
      const isLive = await liveIndicator.evaluate(el =>
        el.textContent?.toLowerCase().includes('live') ||
        el.classList.contains('live') ||
        el.classList.contains('connected')
      );
      return isLive;
    }

    if (connectionStatus) {
      const isConnected = await connectionStatus.evaluate(el =>
        el.textContent?.toLowerCase().includes('connected') ||
        el.classList.contains('connected')
      );
      return isConnected;
    }

    return false;
  }

  /**
   * Get last updated timestamp
   */
  async getLastUpdatedTime(): Promise<string | null> {
    const lastUpdatedElement = await this.page.$(this.selectors.lastUpdated);

    if (lastUpdatedElement) {
      return await lastUpdatedElement.evaluate(el => el.textContent?.trim() || null);
    }

    return null;
  }

  /**
   * Enter fullscreen mode
   */
  async enterFullscreen(): Promise<void> {
    const fullscreenButton = await this.page.$(this.selectors.fullscreenButton);

    if (fullscreenButton) {
      await fullscreenButton.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Check for error states
   */
  async hasErrorState(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.errorMessage);
  }

  /**
   * Get error message
   */
  async getErrorMessage(): Promise<string | null> {
    if (await this.hasErrorState()) {
      return await this.getElementText(this.selectors.errorMessage);
    }
    return null;
  }

  /**
   * Check for empty/no data state
   */
  async hasNoDataState(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.emptyState) ||
           await this.isElementVisible(this.selectors.noDataMessage);
  }

  /**
   * Verify dashboard responsiveness
   */
  async verifyResponsiveness(): Promise<{
    mobile: boolean;
    tablet: boolean;
    desktop: boolean;
  }> {
    const results = { mobile: false, tablet: false, desktop: false };

    // Test mobile viewport
    await this.page.setViewport({ width: 375, height: 667 });
    await this.page.waitForTimeout(1000);
    results.mobile = await this.areChartsLoaded();

    // Test tablet viewport
    await this.page.setViewport({ width: 768, height: 1024 });
    await this.page.waitForTimeout(1000);
    results.tablet = await this.areChartsLoaded();

    // Test desktop viewport
    await this.page.setViewport({ width: 1920, height: 1080 });
    await this.page.waitForTimeout(1000);
    results.desktop = await this.areChartsLoaded();

    return results;
  }

  /**
   * Measure dashboard performance
   */
  async measurePerformance(): Promise<{
    loadTime: number;
    chartRenderTime: number;
    dataFetchTime: number;
  }> {
    const startTime = Date.now();

    // Measure total load time
    await this.waitForLoadingToComplete();
    const loadTime = Date.now() - startTime;

    // Measure chart render time
    const chartStartTime = Date.now();
    await this.waitForFunction(
      () => document.querySelectorAll('[data-testid*="chart"]').length > 0,
      { timeout: 10000 }
    );
    const chartRenderTime = Date.now() - chartStartTime;

    // Measure data fetch time by triggering refresh
    const dataStartTime = Date.now();
    await this.refreshData();
    const dataFetchTime = Date.now() - dataStartTime;

    return {
      loadTime,
      chartRenderTime,
      dataFetchTime,
    };
  }
}