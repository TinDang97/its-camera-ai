/**
 * Real-time Metric Card Component
 *
 * Live updating metric display cards for analytics dashboard
 * with trend indicators, sparklines, and comparative data.
 */

'use client';

import React, { useMemo } from 'react';
import { useOptimistic, useDeferredValue } from 'react';
import { cn } from '@/lib/utils';
import {
  IconTrendingUp,
  IconTrendingDown,
  IconMinus,
  IconAlertTriangle,
  IconCheckCircle,
  IconClock
} from '@tabler/icons-react';
import { DashboardWidget, WidgetConfig, WidgetData } from './DashboardWidget';
import { RealTimeLineChart } from '@/components/charts/RealTimeLineChart';

export interface MetricCardData {
  value: number;
  previousValue?: number;
  target?: number;
  unit?: string;
  format?: 'number' | 'percentage' | 'currency' | 'duration' | 'bytes';
  trend?: 'up' | 'down' | 'stable';
  trendPercentage?: number;
  status?: 'normal' | 'warning' | 'critical' | 'good';
  timeSeries?: Array<{ timestamp: number; value: number }>;
  metadata?: {
    description?: string;
    source?: string;
    lastCalculated?: number;
    confidence?: number;
  };
}

export interface MetricCardProps {
  title: string;
  subtitle?: string;
  data: MetricCardData;
  size?: 'small' | 'medium' | 'large';
  showSparkline?: boolean;
  showTrend?: boolean;
  showTarget?: boolean;
  showStatus?: boolean;
  className?: string;
  onValueClick?: (data: MetricCardData) => void;
}

// Value formatters
const formatValue = (value: number, format?: MetricCardData['format'], unit?: string): string => {
  switch (format) {
    case 'percentage':
      return `${value.toFixed(1)}%`;
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
      }).format(value);
    case 'duration':
      if (value < 1000) return `${value.toFixed(0)}ms`;
      if (value < 60000) return `${(value / 1000).toFixed(1)}s`;
      if (value < 3600000) return `${(value / 60000).toFixed(1)}m`;
      return `${(value / 3600000).toFixed(1)}h`;
    case 'bytes':
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      let size = value;
      let unitIndex = 0;
      while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
      }
      return `${size.toFixed(1)} ${units[unitIndex]}`;
    case 'number':
    default:
      const formatted = value >= 1000000
        ? `${(value / 1000000).toFixed(1)}M`
        : value >= 1000
        ? `${(value / 1000).toFixed(1)}K`
        : value.toFixed(0);
      return unit ? `${formatted} ${unit}` : formatted;
  }
};

// Calculate trend from current and previous values
const calculateTrend = (current: number, previous?: number): {
  trend: 'up' | 'down' | 'stable';
  percentage: number;
} => {
  if (!previous || previous === 0) {
    return { trend: 'stable', percentage: 0 };
  }

  const change = ((current - previous) / previous) * 100;

  if (Math.abs(change) < 1) {
    return { trend: 'stable', percentage: change };
  }

  return {
    trend: change > 0 ? 'up' : 'down',
    percentage: Math.abs(change),
  };
};

// Status indicator configurations
const statusConfig = {
  normal: {
    color: 'text-muted-foreground',
    bgColor: 'bg-muted/20',
    icon: IconClock,
  },
  warning: {
    color: 'text-warning',
    bgColor: 'bg-warning/20',
    icon: IconAlertTriangle,
  },
  critical: {
    color: 'text-destructive',
    bgColor: 'bg-destructive/20',
    icon: IconAlertTriangle,
  },
  good: {
    color: 'text-success',
    bgColor: 'bg-success/20',
    icon: IconCheckCircle,
  },
};

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  subtitle,
  data,
  size = 'medium',
  showSparkline = true,
  showTrend = true,
  showTarget = true,
  showStatus = true,
  className,
  onValueClick,
}) => {
  // Defer expensive calculations
  const deferredData = useDeferredValue(data);

  // Calculate trend information
  const trendInfo = useMemo(() => {
    if (deferredData.trend && deferredData.trendPercentage !== undefined) {
      return {
        trend: deferredData.trend,
        percentage: deferredData.trendPercentage,
      };
    }
    return calculateTrend(deferredData.value, deferredData.previousValue);
  }, [deferredData.value, deferredData.previousValue, deferredData.trend, deferredData.trendPercentage]);

  // Format main value
  const formattedValue = useMemo(() =>
    formatValue(deferredData.value, deferredData.format, deferredData.unit),
    [deferredData.value, deferredData.format, deferredData.unit]
  );

  // Format target value
  const formattedTarget = useMemo(() =>
    deferredData.target ? formatValue(deferredData.target, deferredData.format, deferredData.unit) : null,
    [deferredData.target, deferredData.format, deferredData.unit]
  );

  // Widget configuration
  const widgetConfig: WidgetConfig = {
    id: `metric-${title.toLowerCase().replace(/\s+/g, '-')}`,
    title,
    subtitle,
    size,
    showHeader: false,
    priority: deferredData.status === 'critical' ? 'critical' :
             deferredData.status === 'warning' ? 'high' : 'medium',
    category: 'analytics',
  };

  // Widget data
  const widgetData: WidgetData = {
    timestamp: deferredData.metadata?.lastCalculated || Date.now(),
    isLoading: false,
    data: deferredData,
  };

  // Trend icon
  const TrendIcon = trendInfo.trend === 'up' ? IconTrendingUp :
                   trendInfo.trend === 'down' ? IconTrendingDown : IconMinus;

  // Status configuration
  const status = deferredData.status || 'normal';
  const statusInfo = statusConfig[status];
  const StatusIcon = statusInfo.icon;

  return (
    <DashboardWidget
      config={widgetConfig}
      data={widgetData}
      className={className}
    >
      <div className="space-y-4">
        {/* Header with status */}
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
            {subtitle && (
              <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
            )}
          </div>

          {showStatus && (
            <div className={cn(
              'flex items-center space-x-1 px-2 py-1 rounded-md text-xs',
              statusInfo.bgColor
            )}>
              <StatusIcon className={cn('h-3 w-3', statusInfo.color)} />
              <span className={statusInfo.color}>
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </span>
            </div>
          )}
        </div>

        {/* Main value */}
        <div className="space-y-2">
          <div
            className={cn(
              'text-2xl font-bold text-foreground',
              size === 'large' && 'text-3xl',
              size === 'small' && 'text-xl',
              onValueClick && 'cursor-pointer hover:text-primary transition-colors'
            )}
            onClick={() => onValueClick?.(deferredData)}
          >
            {formattedValue}
          </div>

          {/* Trend and target information */}
          <div className="flex items-center justify-between">
            {showTrend && (
              <div className={cn(
                'flex items-center space-x-1 text-xs',
                trendInfo.trend === 'up' && 'text-success',
                trendInfo.trend === 'down' && 'text-destructive',
                trendInfo.trend === 'stable' && 'text-muted-foreground'
              )}>
                <TrendIcon className="h-3 w-3" />
                <span>{trendInfo.percentage.toFixed(1)}%</span>
                {deferredData.previousValue && (
                  <span className="text-muted-foreground">
                    from {formatValue(deferredData.previousValue, deferredData.format, deferredData.unit)}
                  </span>
                )}
              </div>
            )}

            {showTarget && formattedTarget && (
              <div className="text-xs text-muted-foreground">
                Target: {formattedTarget}
              </div>
            )}
          </div>
        </div>

        {/* Sparkline chart */}
        {showSparkline && deferredData.timeSeries && deferredData.timeSeries.length > 0 && (
          <div className="h-12">
            <RealTimeLineChart
              data={deferredData.timeSeries}
              height={48}
              enableAnimations={false}
              showTrend={false}
              showStats={false}
              responsive={true}
              color={trendInfo.trend === 'up' ? 'success' :
                    trendInfo.trend === 'down' ? 'destructive' : 'primary'}
            />
          </div>
        )}

        {/* Metadata */}
        {deferredData.metadata?.description && (
          <p className="text-xs text-muted-foreground">
            {deferredData.metadata.description}
          </p>
        )}

        {/* Confidence indicator */}
        {deferredData.metadata?.confidence !== undefined && (
          <div className="flex items-center space-x-1 text-xs text-muted-foreground">
            <span>Confidence:</span>
            <div className="flex-1 bg-muted rounded-full h-1">
              <div
                className="bg-primary h-1 rounded-full transition-all duration-300"
                style={{ width: `${deferredData.metadata.confidence * 100}%` }}
              />
            </div>
            <span>{(deferredData.metadata.confidence * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
    </DashboardWidget>
  );
};

export default MetricCard;
