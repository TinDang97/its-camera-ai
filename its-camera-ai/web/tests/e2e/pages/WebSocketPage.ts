import { Page } from 'puppeteer';
import { BasePage } from './BasePage';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp?: string;
}

export interface RealTimeUpdate {
  type: 'camera_status' | 'traffic_data' | 'incident_created' | 'system_alert';
  data: any;
  timestamp: string;
}

export class WebSocketPage extends BasePage {
  private selectors = {
    // Connection status
    connectionStatus: '[data-testid="connection-status"], [data-testid="ws-status"]',
    connectionIndicator: '[data-testid="connection-indicator"]',
    reconnectButton: '[data-testid="reconnect-ws"], button:has-text("Reconnect")',

    // Real-time data containers
    realTimeSection: '[data-testid="real-time-section"]',
    liveDataContainer: '[data-testid="live-data"], [data-testid="real-time-data"]',
    dataTimestamp: '[data-testid="data-timestamp"]',
    lastUpdated: '[data-testid="last-updated"]',

    // Camera status updates
    cameraStatusContainer: '[data-testid="camera-status-updates"]',
    cameraStatusItem: '[data-testid="camera-status-item"]',
    cameraOnlineIndicator: '[data-testid="camera-online"], .status-online',
    cameraOfflineIndicator: '[data-testid="camera-offline"], .status-offline',

    // Traffic data updates
    trafficDataContainer: '[data-testid="traffic-data-updates"]',
    vehicleCountDisplay: '[data-testid="vehicle-count"]',
    averageSpeedDisplay: '[data-testid="average-speed"]',
    congestionLevelDisplay: '[data-testid="congestion-level"]',

    // Incident alerts
    incidentAlertsContainer: '[data-testid="incident-alerts"]',
    newIncidentAlert: '[data-testid="new-incident-alert"]',
    incidentSeverity: '[data-testid="incident-severity"]',
    alertDismissButton: '[data-testid="dismiss-alert"]',

    // System alerts
    systemAlertsContainer: '[data-testid="system-alerts"]',
    systemAlertItem: '[data-testid="system-alert-item"]',
    alertMessage: '[data-testid="alert-message"]',
    alertSeverity: '[data-testid="alert-severity"]',

    // Live charts and graphs
    liveChart: '[data-testid="live-chart"], .live-chart, .real-time-chart',
    chartContainer: '[data-testid="chart-container"]',
    chartData: '[data-testid="chart-data"]',
    chartUpdateIndicator: '[data-testid="chart-updating"]',

    // Dashboard metrics
    dashboardMetrics: '[data-testid="dashboard-metrics"]',
    totalCamerasMetric: '[data-testid="total-cameras"]',
    onlineCamerasMetric: '[data-testid="online-cameras"]',
    activeIncidentsMetric: '[data-testid="active-incidents"]',
    systemHealthMetric: '[data-testid="system-health"]',

    // Notification system
    notificationContainer: '[data-testid="notifications"], .notification-container',
    notificationItem: '[data-testid="notification-item"]',
    notificationTitle: '[data-testid="notification-title"]',
    notificationBody: '[data-testid="notification-body"]',
    notificationClose: '[data-testid="notification-close"]',

    // Loading states
    loadingIndicator: '[data-testid="loading"], .loading, .spinner',
    dataLoadingIndicator: '[data-testid="data-loading"]',

    // Error states
    connectionError: '[data-testid="connection-error"]',
    dataError: '[data-testid="data-error"]',
    retryButton: '[data-testid="retry-connection"]',

    // Settings and controls
    realTimeToggle: '[data-testid="real-time-toggle"]',
    autoRefreshToggle: '[data-testid="auto-refresh-toggle"]',
    refreshIntervalSelect: '[data-testid="refresh-interval"]',
    pauseUpdatesButton: '[data-testid="pause-updates"]',
    resumeUpdatesButton: '[data-testid="resume-updates"]',
  } as const;

  private wsMessages: WebSocketMessage[] = [];
  private isMonitoringWS = false;

  constructor(page: Page) {
    super(page);
  }

  /**
   * Start monitoring WebSocket messages
   */
  async startWebSocketMonitoring(): Promise<void> {
    if (this.isMonitoringWS) return;

    this.isMonitoringWS = true;
    this.wsMessages = [];

    // Intercept WebSocket connections
    await this.page.evaluateOnNewDocument(() => {
      const originalWebSocket = window.WebSocket;

      // Override WebSocket constructor
      window.WebSocket = class extends originalWebSocket {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);

          // Store reference for testing
          (window as any).__testWebSocket = this;

          // Log connection events
          this.addEventListener('open', () => {
            console.log('WebSocket connected:', url);
            (window as any).__wsConnected = true;
          });

          this.addEventListener('close', () => {
            console.log('WebSocket disconnected:', url);
            (window as any).__wsConnected = false;
          });

          this.addEventListener('error', (error) => {
            console.error('WebSocket error:', error);
            (window as any).__wsError = error;
          });

          this.addEventListener('message', (event) => {
            try {
              const data = JSON.parse(event.data);
              if (!(window as any).__wsMessages) {
                (window as any).__wsMessages = [];
              }
              (window as any).__wsMessages.push({
                type: data.type || 'unknown',
                data: data.data || data,
                timestamp: new Date().toISOString(),
              });
            } catch (e) {
              console.warn('Failed to parse WebSocket message:', event.data);
            }
          });
        }
      };
    });
  }

  /**
   * Stop monitoring WebSocket messages
   */
  async stopWebSocketMonitoring(): Promise<void> {
    this.isMonitoringWS = false;
  }

  /**
   * Get captured WebSocket messages
   */
  async getWebSocketMessages(): Promise<WebSocketMessage[]> {
    return await this.page.evaluate(() => {
      return (window as any).__wsMessages || [];
    });
  }

  /**
   * Check if WebSocket is connected
   */
  async isWebSocketConnected(): Promise<boolean> {
    return await this.page.evaluate(() => {
      return (window as any).__wsConnected === true;
    });
  }

  /**
   * Wait for WebSocket connection
   */
  async waitForWebSocketConnection(timeout = 10000): Promise<void> {
    await this.page.waitForFunction(
      () => (window as any).__wsConnected === true,
      { timeout }
    );
  }

  /**
   * Send test WebSocket message
   */
  async sendWebSocketMessage(message: any): Promise<void> {
    await this.page.evaluate((msg) => {
      const ws = (window as any).__testWebSocket;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
      }
    }, message);
  }

  /**
   * Simulate WebSocket message reception
   */
  async simulateWebSocketMessage(message: RealTimeUpdate): Promise<void> {
    await this.page.evaluate((msg) => {
      const event = new MessageEvent('message', {
        data: JSON.stringify(msg)
      });

      const ws = (window as any).__testWebSocket;
      if (ws) {
        ws.dispatchEvent(event);
      }
    }, message);
  }

  /**
   * Check connection status indicator
   */
  async getConnectionStatus(): Promise<'connected' | 'disconnected' | 'connecting' | 'error'> {
    const statusElement = await this.page.$(this.selectors.connectionStatus);

    if (!statusElement) {
      return 'error';
    }

    const statusText = await statusElement.evaluate(el => el.textContent?.toLowerCase() || '');
    const statusClass = await statusElement.evaluate(el => el.className || '');

    if (statusText.includes('connected') || statusClass.includes('connected')) {
      return 'connected';
    } else if (statusText.includes('connecting') || statusClass.includes('connecting')) {
      return 'connecting';
    } else if (statusText.includes('disconnected') || statusClass.includes('disconnected')) {
      return 'disconnected';
    }

    return 'error';
  }

  /**
   * Wait for real-time data to update
   */
  async waitForDataUpdate(timeout = 5000): Promise<void> {
    const initialTimestamp = await this.getLastUpdateTimestamp();

    await this.page.waitForFunction(
      (initial) => {
        const current = document.querySelector('[data-testid="last-updated"], [data-testid="data-timestamp"]')?.textContent;
        return current && current !== initial;
      },
      { timeout },
      initialTimestamp
    );
  }

  /**
   * Get last update timestamp
   */
  async getLastUpdateTimestamp(): Promise<string | null> {
    const timestampElement = await this.page.$(this.selectors.lastUpdated + ', ' + this.selectors.dataTimestamp);

    if (timestampElement) {
      return await timestampElement.evaluate(el => el.textContent?.trim() || null);
    }

    return null;
  }

  /**
   * Check if data is loading
   */
  async isDataLoading(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.dataLoadingIndicator);
  }

  /**
   * Get camera status updates
   */
  async getCameraStatusUpdates(): Promise<Array<{
    id: string;
    status: 'online' | 'offline' | 'maintenance';
    timestamp: string;
  }>> {
    const updates: Array<{
      id: string;
      status: 'online' | 'offline' | 'maintenance';
      timestamp: string;
    }> = [];

    const statusItems = await this.page.$$(this.selectors.cameraStatusItem);

    for (const item of statusItems) {
      try {
        const id = await item.evaluate(el => el.getAttribute('data-camera-id') || '');
        const statusText = await item.evaluate(el => el.textContent?.toLowerCase() || '');
        const timestamp = await item.evaluate(el =>
          el.querySelector('[data-testid="timestamp"]')?.textContent || ''
        );

        let status: 'online' | 'offline' | 'maintenance' = 'offline';
        if (statusText.includes('online')) status = 'online';
        else if (statusText.includes('maintenance')) status = 'maintenance';

        updates.push({ id, status, timestamp });
      } catch (error) {
        console.warn('Error parsing camera status update:', error);
      }
    }

    return updates;
  }

  /**
   * Get traffic data updates
   */
  async getTrafficDataUpdates(): Promise<{
    vehicleCount: number;
    averageSpeed: number;
    congestionLevel: string;
    timestamp: string;
  } | null> {
    try {
      const vehicleCount = await this.getElementText(this.selectors.vehicleCountDisplay);
      const averageSpeed = await this.getElementText(this.selectors.averageSpeedDisplay);
      const congestionLevel = await this.getElementText(this.selectors.congestionLevelDisplay);
      const timestamp = await this.getLastUpdateTimestamp();

      return {
        vehicleCount: parseInt(vehicleCount || '0', 10),
        averageSpeed: parseFloat(averageSpeed || '0'),
        congestionLevel: congestionLevel || 'unknown',
        timestamp: timestamp || '',
      };
    } catch (error) {
      console.warn('Error getting traffic data updates:', error);
      return null;
    }
  }

  /**
   * Get active incident alerts
   */
  async getActiveIncidentAlerts(): Promise<Array<{
    id: string;
    title: string;
    severity: string;
    timestamp: string;
  }>> {
    const alerts: Array<{
      id: string;
      title: string;
      severity: string;
      timestamp: string;
    }> = [];

    const alertItems = await this.page.$$(this.selectors.newIncidentAlert);

    for (const item of alertItems) {
      try {
        const id = await item.evaluate(el => el.getAttribute('data-incident-id') || '');
        const title = await item.evaluate(el =>
          el.querySelector('[data-testid="incident-title"]')?.textContent || ''
        );
        const severity = await item.evaluate(el =>
          el.querySelector('[data-testid="incident-severity"]')?.textContent || ''
        );
        const timestamp = await item.evaluate(el =>
          el.querySelector('[data-testid="timestamp"]')?.textContent || ''
        );

        alerts.push({ id, title, severity, timestamp });
      } catch (error) {
        console.warn('Error parsing incident alert:', error);
      }
    }

    return alerts;
  }

  /**
   * Dismiss incident alert
   */
  async dismissIncidentAlert(incidentId: string): Promise<void> {
    const alertElement = await this.page.$(`[data-incident-id="${incidentId}"]`);

    if (alertElement) {
      const dismissButton = await alertElement.$(this.selectors.alertDismissButton);
      if (dismissButton) {
        await dismissButton.click();
        await this.page.waitForTimeout(500);
      }
    }
  }

  /**
   * Get system alerts
   */
  async getSystemAlerts(): Promise<Array<{
    id: string;
    message: string;
    severity: string;
    timestamp: string;
  }>> {
    const alerts: Array<{
      id: string;
      message: string;
      severity: string;
      timestamp: string;
    }> = [];

    const alertItems = await this.page.$$(this.selectors.systemAlertItem);

    for (const item of alertItems) {
      try {
        const id = await item.evaluate(el => el.getAttribute('data-alert-id') || '');
        const message = await item.evaluate(el =>
          el.querySelector('[data-testid="alert-message"]')?.textContent || ''
        );
        const severity = await item.evaluate(el =>
          el.querySelector('[data-testid="alert-severity"]')?.textContent || ''
        );
        const timestamp = await item.evaluate(el =>
          el.querySelector('[data-testid="timestamp"]')?.textContent || ''
        );

        alerts.push({ id, message, severity, timestamp });
      } catch (error) {
        console.warn('Error parsing system alert:', error);
      }
    }

    return alerts;
  }

  /**
   * Toggle real-time updates
   */
  async toggleRealTimeUpdates(): Promise<void> {
    const toggleElement = await this.page.$(this.selectors.realTimeToggle);

    if (toggleElement) {
      await toggleElement.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Pause real-time updates
   */
  async pauseRealTimeUpdates(): Promise<void> {
    const pauseButton = await this.page.$(this.selectors.pauseUpdatesButton);

    if (pauseButton) {
      await pauseButton.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Resume real-time updates
   */
  async resumeRealTimeUpdates(): Promise<void> {
    const resumeButton = await this.page.$(this.selectors.resumeUpdatesButton);

    if (resumeButton) {
      await resumeButton.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Check if live chart is updating
   */
  async isLiveChartUpdating(): Promise<boolean> {
    const chartContainer = await this.page.$(this.selectors.liveChart);

    if (!chartContainer) {
      return false;
    }

    // Check for update indicator or animation classes
    const hasUpdateIndicator = await this.isElementVisible(this.selectors.chartUpdateIndicator);
    const hasAnimationClass = await chartContainer.evaluate(el =>
      el.classList.contains('updating') || el.classList.contains('animated')
    );

    return hasUpdateIndicator || hasAnimationClass;
  }

  /**
   * Wait for chart data to update
   */
  async waitForChartUpdate(timeout = 5000): Promise<void> {
    await this.page.waitForFunction(
      (selector) => {
        const chart = document.querySelector(selector);
        return chart && (
          chart.classList.contains('updated') ||
          chart.querySelector('[data-testid="chart-data"]')
        );
      },
      { timeout },
      this.selectors.liveChart
    );
  }

  /**
   * Get dashboard metrics
   */
  async getDashboardMetrics(): Promise<{
    totalCameras: number;
    onlineCameras: number;
    activeIncidents: number;
    systemHealth: string;
  }> {
    const totalCameras = parseInt(await this.getElementText(this.selectors.totalCamerasMetric) || '0', 10);
    const onlineCameras = parseInt(await this.getElementText(this.selectors.onlineCamerasMetric) || '0', 10);
    const activeIncidents = parseInt(await this.getElementText(this.selectors.activeIncidentsMetric) || '0', 10);
    const systemHealth = await this.getElementText(this.selectors.systemHealthMetric) || 'unknown';

    return {
      totalCameras,
      onlineCameras,
      activeIncidents,
      systemHealth,
    };
  }

  /**
   * Simulate network disconnection
   */
  async simulateNetworkDisconnection(): Promise<void> {
    await this.page.setOfflineMode(true);
    await this.page.waitForTimeout(1000);
  }

  /**
   * Restore network connection
   */
  async restoreNetworkConnection(): Promise<void> {
    await this.page.setOfflineMode(false);
    await this.page.waitForTimeout(1000);
  }

  /**
   * Click reconnect button
   */
  async clickReconnect(): Promise<void> {
    await this.clickElement(this.selectors.reconnectButton);
    await this.page.waitForTimeout(1000);
  }

  /**
   * Check for connection errors
   */
  async hasConnectionError(): Promise<boolean> {
    return await this.isElementVisible(this.selectors.connectionError);
  }

  /**
   * Get connection error message
   */
  async getConnectionErrorMessage(): Promise<string | null> {
    if (await this.hasConnectionError()) {
      return await this.getElementText(this.selectors.connectionError);
    }
    return null;
  }

  /**
   * Wait for notifications
   */
  async waitForNotification(timeout = 5000): Promise<void> {
    await this.waitForElement(this.selectors.notificationItem, timeout);
  }

  /**
   * Get active notifications
   */
  async getActiveNotifications(): Promise<Array<{
    title: string;
    body: string;
    type: string;
  }>> {
    const notifications: Array<{
      title: string;
      body: string;
      type: string;
    }> = [];

    const notificationItems = await this.page.$$(this.selectors.notificationItem);

    for (const item of notificationItems) {
      try {
        const title = await item.evaluate(el =>
          el.querySelector('[data-testid="notification-title"]')?.textContent || ''
        );
        const body = await item.evaluate(el =>
          el.querySelector('[data-testid="notification-body"]')?.textContent || ''
        );
        const type = await item.evaluate(el =>
          el.getAttribute('data-notification-type') || 'info'
        );

        notifications.push({ title, body, type });
      } catch (error) {
        console.warn('Error parsing notification:', error);
      }
    }

    return notifications;
  }

  /**
   * Close notification
   */
  async closeNotification(index = 0): Promise<void> {
    const notifications = await this.page.$$(this.selectors.notificationItem);

    if (notifications[index]) {
      const closeButton = await notifications[index].$(this.selectors.notificationClose);
      if (closeButton) {
        await closeButton.click();
        await this.page.waitForTimeout(500);
      }
    }
  }

  /**
   * Measure real-time performance
   */
  async measureRealTimePerformance(): Promise<{
    wsConnectionTime: number;
    messageLatency: number;
    updateFrequency: number;
  }> {
    const startTime = Date.now();

    // Measure WebSocket connection time
    await this.waitForWebSocketConnection();
    const wsConnectionTime = Date.now() - startTime;

    // Measure message latency
    const messageStartTime = Date.now();
    await this.simulateWebSocketMessage({
      type: 'system_alert',
      data: { id: 'test', message: 'Performance test' },
      timestamp: new Date().toISOString(),
    });

    await this.waitForDataUpdate();
    const messageLatency = Date.now() - messageStartTime;

    // Measure update frequency (count updates over 10 seconds)
    const messages = await this.getWebSocketMessages();
    const recentMessages = messages.filter(msg =>
      Date.now() - new Date(msg.timestamp || 0).getTime() < 10000
    );
    const updateFrequency = recentMessages.length / 10; // per second

    return {
      wsConnectionTime,
      messageLatency,
      updateFrequency,
    };
  }
}