/**
 * Chart Components Library
 *
 * Comprehensive real-time chart library with D3.js integration for
 * the ITS Camera AI Dashboard. Supports 60fps animations and 1000+ data points.
 */

// Core chart hook
export { useD3Chart, type ChartData, type ChartDimensions, type D3ChartConfig } from './useD3Chart';

// Chart components
export { RealTimeLineChart, type RealTimeLineChartProps } from './RealTimeLineChart';
export { RealTimeAreaChart, type RealTimeAreaChartProps } from './RealTimeAreaChart';
export { RealTimeBarChart, type RealTimeBarChartProps, type BarChartData } from './RealTimeBarChart';
export { TrafficFlowHeatmap, type TrafficFlowHeatmapProps, type TrafficFlowData } from './TrafficFlowHeatmap';

// Utilities
export {
  ChartDataProcessor,
  ChartFormatters,
  ChartPerformance,
  ChartColors,
} from './utils';

// Import components for default export
import { RealTimeLineChart } from './RealTimeLineChart';
import { RealTimeAreaChart } from './RealTimeAreaChart';
import { RealTimeBarChart } from './RealTimeBarChart';
import { TrafficFlowHeatmap } from './TrafficFlowHeatmap';
import { useD3Chart } from './useD3Chart';
import { ChartDataProcessor, ChartFormatters, ChartPerformance, ChartColors } from './utils';

// Re-export default utilities
export { default as ChartUtils } from './utils';

// Chart configuration presets
export const CHART_PRESETS = {
  // Traffic analytics presets
  TRAFFIC_VOLUME: {
    height: 300,
    enableAnimations: true,
    maxDataPoints: 500,
    updateInterval: 100,
    color: 'primary' as const,
    showTrend: true,
    showStats: true,
  },

  TRAFFIC_SPEED: {
    height: 250,
    enableAnimations: true,
    maxDataPoints: 300,
    updateInterval: 200,
    color: 'success' as const,
    showTrend: true,
    formatValue: (value: number) => `${value.toFixed(1)} mph`,
  },

  // Camera health presets
  CAMERA_STATUS: {
    height: 200,
    orientation: 'horizontal' as const,
    showValues: true,
    maxBars: 10,
    sortBars: 'desc' as const,
    barColor: 'hsl(var(--success))',
  },

  CAMERA_RESPONSE_TIME: {
    height: 300,
    gradientColors: ['hsl(var(--warning))', 'transparent'] as [string, string],
    strokeColor: 'hsl(var(--warning))',
    fillOpacity: 0.4,
    maxDataPoints: 1000,
    formatValue: (value: number) => `${value.toFixed(0)}ms`,
  },

  CAMERA_FRAME_RATE: {
    height: 300,
    orientation: 'horizontal' as const,
    showValues: true,
    maxBars: 12,
    sortBars: 'desc' as const,
    barColor: 'hsl(var(--primary))',
    formatValue: (value: number) => `${value.toFixed(1)} fps`,
  },

  CAMERA_HEALTH_SCORE: {
    height: 250,
    gradientColors: ['hsl(var(--success))', 'transparent'] as [string, string],
    strokeColor: 'hsl(var(--success))',
    fillOpacity: 0.3,
    maxDataPoints: 100,
    formatValue: (value: number) => `${value.toFixed(0)}%`,
  },

  CAMERA_UPTIME: {
    height: 200,
    enableAnimations: true,
    maxDataPoints: 50,
    color: 'success' as const,
    showTrend: true,
    formatValue: (value: number) => `${value.toFixed(1)}%`,
  },

  // System metrics presets
  SYSTEM_CPU: {
    height: 250,
    gradientColors: ['hsl(var(--destructive))', 'transparent'] as [string, string],
    strokeColor: 'hsl(var(--destructive))',
    fillOpacity: 0.3,
    maxDataPoints: 200,
    formatValue: (value: number) => `${value.toFixed(1)}%`,
  },

  SYSTEM_MEMORY: {
    height: 250,
    gradientColors: ['hsl(var(--primary))', 'transparent'] as [string, string],
    strokeColor: 'hsl(var(--primary))',
    fillOpacity: 0.3,
    maxDataPoints: 200,
    formatValue: (value: number) => `${(value / 1024 / 1024).toFixed(1)} MB`,
  },

  SYSTEM_NETWORK: {
    height: 300,
    gradientColors: ['hsl(var(--success))', 'transparent'] as [string, string],
    strokeColor: 'hsl(var(--success))',
    fillOpacity: 0.4,
    maxDataPoints: 100,
    formatValue: (value: number) => `${value.toFixed(1)} MB/s`,
  },

  SYSTEM_API_RESPONSE: {
    height: 300,
    color: 'primary' as const,
    enableAnimations: true,
    maxDataPoints: 150,
    formatValue: (value: number) => `${value.toFixed(0)}ms`,
  },

  SYSTEM_DATABASE: {
    height: 300,
    color: 'warning' as const,
    enableAnimations: true,
    maxDataPoints: 100,
    formatValue: (value: number) => `${value.toFixed(0)}ms`,
  },

  SYSTEM_HEALTH_SCORE: {
    height: 200,
    color: 'success' as const,
    showTrend: true,
    showStats: false,
    formatValue: (value: number) => `${value.toFixed(0)}%`,
  },

  // Incident analytics presets
  INCIDENT_TIMELINE: {
    height: 400,
    enableAnimations: true,
    maxDataPoints: 100,
    color: 'destructive' as const,
    showTrend: false,
    showStats: false,
  },

  INCIDENT_CATEGORIES: {
    height: 300,
    orientation: 'vertical' as const,
    showValues: true,
    maxBars: 8,
    sortBars: 'desc' as const,
    barColor: 'hsl(var(--warning))',
  },

  // Performance monitoring presets
  PROCESSING_LATENCY: {
    height: 200,
    enableAnimations: false, // Disabled for high-frequency updates
    maxDataPoints: 1000,
    updateInterval: 50,
    color: 'secondary' as const,
    formatValue: (value: number) => `${value.toFixed(2)}ms`,
  },

  // Traffic flow heatmap preset
  TRAFFIC_HEATMAP: {
    width: 800,
    height: 500,
    showLegend: true,
    showCameraLabels: true,
    enableInteraction: true,
    colorScheme: 'traffic' as const,
    showTrafficFlow: true,
    animationDuration: 500,
  },

  THROUGHPUT: {
    height: 250,
    gradientColors: ['hsl(var(--success))', 'transparent'] as [string, string],
    strokeColor: 'hsl(var(--success))',
    fillOpacity: 0.5,
    maxDataPoints: 500,
    formatValue: (value: number) => `${ChartFormatters.formatLargeNumber(value)}/s`,
  },

  // Compact widget presets
  COMPACT_METRIC: {
    height: 150,
    width: 300,
    enableAnimations: true,
    maxDataPoints: 50,
    showTrend: false,
    showStats: false,
    responsive: true,
  },

  MINI_CHART: {
    height: 100,
    width: 200,
    enableAnimations: false,
    maxDataPoints: 20,
    showGrid: false,
    showValues: false,
    responsive: true,
  },
} as const;

// Chart theme configurations
export const CHART_THEMES = {
  DEFAULT: {
    colors: {
      primary: 'hsl(var(--primary))',
      secondary: 'hsl(var(--secondary))',
      success: 'hsl(var(--success))',
      warning: 'hsl(var(--warning))',
      destructive: 'hsl(var(--destructive))',
    },
    fonts: {
      family: 'Inter, sans-serif',
      sizes: {
        title: '16px',
        label: '12px',
        tick: '10px',
      },
    },
    animations: {
      duration: 300,
      easing: 'ease-in-out',
    },
  },

  HIGH_PERFORMANCE: {
    colors: {
      primary: 'hsl(var(--primary))',
      secondary: 'hsl(var(--secondary))',
      success: 'hsl(var(--success))',
      warning: 'hsl(var(--warning))',
      destructive: 'hsl(var(--destructive))',
    },
    fonts: {
      family: 'Inter, sans-serif',
      sizes: {
        title: '14px',
        label: '10px',
        tick: '9px',
      },
    },
    animations: {
      duration: 0, // Disabled for maximum performance
      easing: 'linear',
    },
  },

  DARK_MODE: {
    colors: {
      primary: 'hsl(210, 100%, 60%)',
      secondary: 'hsl(var(--secondary))',
      success: 'hsl(120, 100%, 50%)',
      warning: 'hsl(45, 100%, 55%)',
      destructive: 'hsl(0, 100%, 55%)',
    },
    fonts: {
      family: 'Inter, sans-serif',
      sizes: {
        title: '16px',
        label: '12px',
        tick: '10px',
      },
    },
    animations: {
      duration: 200,
      easing: 'ease-out',
    },
  },
} as const;

// Helper functions for common chart configurations
export const chartHelpers = {
  /**
   * Creates a traffic volume chart configuration
   */
  createTrafficVolumeChart: (overrides?: Partial<typeof CHART_PRESETS.TRAFFIC_VOLUME>) => ({
    ...CHART_PRESETS.TRAFFIC_VOLUME,
    ...overrides,
  }),

  /**
   * Creates a system metrics area chart configuration
   */
  createSystemMetricsChart: (metric: 'cpu' | 'memory' | 'network', overrides?: any) => ({
    ...(metric === 'cpu' ? CHART_PRESETS.SYSTEM_CPU :
        metric === 'memory' ? CHART_PRESETS.SYSTEM_MEMORY :
        CHART_PRESETS.SYSTEM_NETWORK),
    ...overrides,
  }),

  /**
   * Creates a system API response time chart configuration
   */
  createSystemApiChart: (overrides?: Partial<typeof CHART_PRESETS.SYSTEM_API_RESPONSE>) => ({
    ...CHART_PRESETS.SYSTEM_API_RESPONSE,
    ...overrides,
  }),

  /**
   * Creates a system database performance chart configuration
   */
  createSystemDatabaseChart: (overrides?: Partial<typeof CHART_PRESETS.SYSTEM_DATABASE>) => ({
    ...CHART_PRESETS.SYSTEM_DATABASE,
    ...overrides,
  }),

  /**
   * Creates a system health score chart configuration
   */
  createSystemHealthChart: (overrides?: Partial<typeof CHART_PRESETS.SYSTEM_HEALTH_SCORE>) => ({
    ...CHART_PRESETS.SYSTEM_HEALTH_SCORE,
    ...overrides,
  }),

  /**
   * Creates a camera status bar chart configuration
   */
  createCameraStatusChart: (overrides?: Partial<typeof CHART_PRESETS.CAMERA_STATUS>) => ({
    ...CHART_PRESETS.CAMERA_STATUS,
    ...overrides,
  }),

  /**
   * Creates a camera frame rate chart configuration
   */
  createCameraFrameRateChart: (overrides?: Partial<typeof CHART_PRESETS.CAMERA_FRAME_RATE>) => ({
    ...CHART_PRESETS.CAMERA_FRAME_RATE,
    ...overrides,
  }),

  /**
   * Creates a camera health score chart configuration
   */
  createCameraHealthChart: (overrides?: Partial<typeof CHART_PRESETS.CAMERA_HEALTH_SCORE>) => ({
    ...CHART_PRESETS.CAMERA_HEALTH_SCORE,
    ...overrides,
  }),

  /**
   * Creates a camera uptime chart configuration
   */
  createCameraUptimeChart: (overrides?: Partial<typeof CHART_PRESETS.CAMERA_UPTIME>) => ({
    ...CHART_PRESETS.CAMERA_UPTIME,
    ...overrides,
  }),

  /**
   * Creates a compact widget chart configuration
   */
  createCompactChart: (overrides?: Partial<typeof CHART_PRESETS.COMPACT_METRIC>) => ({
    ...CHART_PRESETS.COMPACT_METRIC,
    ...overrides,
  }),

  /**
   * Creates a traffic flow heatmap configuration
   */
  createTrafficHeatmap: (overrides?: Partial<typeof CHART_PRESETS.TRAFFIC_HEATMAP>) => ({
    ...CHART_PRESETS.TRAFFIC_HEATMAP,
    ...overrides,
  }),
};

// Type exports for convenience
export type ChartPreset = keyof typeof CHART_PRESETS;
export type ChartTheme = keyof typeof CHART_THEMES;
export type ChartHelper = keyof typeof chartHelpers;

// Default exports for easy importing
export default {
  // Components
  RealTimeLineChart,
  RealTimeAreaChart,
  RealTimeBarChart,
  TrafficFlowHeatmap,

  // Hook
  useD3Chart,

  // Utilities
  ChartDataProcessor,
  ChartFormatters,
  ChartPerformance,
  ChartColors,

  // Presets and themes
  CHART_PRESETS,
  CHART_THEMES,
  chartHelpers,
};
