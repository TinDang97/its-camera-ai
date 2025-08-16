/**
 * Analytics Dashboard Components
 *
 * Comprehensive dashboard components for real-time monitoring and analysis
 * of the ITS Camera AI system performance and metrics.
 */

// Core dashboard components
export { DashboardWidget, type WidgetConfig, type WidgetData } from './DashboardWidget';
export { MetricCard, type MetricCardData, type MetricCardProps } from './MetricCard';

// Specialized dashboard views
export { TrafficAnalyticsDashboard, type TrafficAnalyticsDashboardProps, type TrafficAnalyticsData } from './TrafficAnalyticsDashboard';
export { CameraPerformanceDashboard, type CameraPerformanceDashboardProps, type CameraPerformanceData } from './CameraPerformanceDashboard';
export { SystemMetricsDashboard, type SystemMetricsDashboardProps, type SystemMetricsData } from './SystemMetricsDashboard';

// Dashboard presets and configurations
export const DASHBOARD_PRESETS = {
  TRAFFIC_ANALYTICS: {
    autoRefresh: true,
    refreshInterval: 30000,
    showControls: true,
    selectedTimeRange: '6h' as const,
  },

  CAMERA_PERFORMANCE: {
    autoRefresh: true,
    refreshInterval: 15000,
    showControls: true,
    selectedTimeRange: '6h' as const,
  },

  SYSTEM_METRICS: {
    autoRefresh: true,
    refreshInterval: 10000,
    showControls: true,
    selectedTimeRange: '1h' as const,
  },

  SYSTEM_MONITORING: {
    autoRefresh: true,
    refreshInterval: 10000,
    showControls: true,
    selectedTimeRange: '1h' as const,
  },

  INCIDENT_MANAGEMENT: {
    autoRefresh: true,
    refreshInterval: 5000,
    showControls: true,
    selectedTimeRange: '24h' as const,
  },
} as const;

// Dashboard layout configurations
export const DASHBOARD_LAYOUTS = {
  COMPACT: {
    gridCols: 'grid-cols-1 md:grid-cols-2',
    metricCardSize: 'small' as const,
    widgetSize: 'medium' as const,
  },

  STANDARD: {
    gridCols: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
    metricCardSize: 'medium' as const,
    widgetSize: 'large' as const,
  },

  EXPANDED: {
    gridCols: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-6',
    metricCardSize: 'large' as const,
    widgetSize: 'xlarge' as const,
  },
} as const;

// Common dashboard utilities
export const dashboardUtils = {
  /**
   * Creates a metric card configuration with common settings
   */
  createMetricCard: (
    title: string,
    subtitle: string,
    data: Partial<MetricCardData>,
    overrides?: any
  ) => ({
    title,
    subtitle,
    data: {
      value: 0,
      format: 'number' as const,
      status: 'normal' as const,
      ...data,
    },
    size: 'medium' as const,
    showSparkline: true,
    ...overrides,
  }),

  /**
   * Creates a widget configuration with common settings
   */
  createWidget: (
    id: string,
    title: string,
    subtitle: string,
    data: any,
    overrides?: Partial<WidgetConfig>
  ) => ({
    config: {
      id,
      title,
      subtitle,
      size: 'large' as const,
      priority: 'medium' as const,
      category: 'analytics' as const,
      ...overrides,
    },
    data: {
      timestamp: Date.now(),
      isLoading: false,
      data,
    },
  }),

  /**
   * Formats time range for display
   */
  formatTimeRange: (range: '1h' | '6h' | '24h' | '7d'): string => {
    const ranges = {
      '1h': '1 Hour',
      '6h': '6 Hours',
      '24h': '24 Hours',
      '7d': '7 Days',
    };
    return ranges[range];
  },

  /**
   * Calculates health score color based on value
   */
  getHealthScoreColor: (score: number): string => {
    if (score >= 85) return 'text-success';
    if (score >= 70) return 'text-warning';
    return 'text-destructive';
  },

  /**
   * Formats metric values with appropriate units
   */
  formatMetricValue: (value: number, type: 'percentage' | 'count' | 'time' | 'rate'): string => {
    switch (type) {
      case 'percentage':
        return `${Math.round(value)}%`;
      case 'count':
        return value.toLocaleString();
      case 'time':
        return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`;
      case 'rate':
        return `${value.toFixed(1)}/s`;
      default:
        return value.toString();
    }
  },
};

// Type exports for convenience
export type DashboardPreset = keyof typeof DASHBOARD_PRESETS;
export type DashboardLayout = keyof typeof DASHBOARD_LAYOUTS;
export type TimeRange = '1h' | '6h' | '24h' | '7d';
export type MetricType = 'percentage' | 'count' | 'time' | 'rate';

// Default exports for easy importing
export default {
  // Components
  DashboardWidget,
  MetricCard,
  TrafficAnalyticsDashboard,
  CameraPerformanceDashboard,
  SystemMetricsDashboard,

  // Configurations
  DASHBOARD_PRESETS,
  DASHBOARD_LAYOUTS,

  // Utilities
  dashboardUtils,
};
