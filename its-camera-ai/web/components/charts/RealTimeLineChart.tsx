/**
 * Real-Time Line Chart Component
 *
 * High-performance line chart with D3.js integration for real-time traffic analytics.
 * Supports up to 1000+ data points with 60fps smooth animations.
 */

'use client';

import React, { useEffect, useMemo, useCallback } from 'react';
import { useDeferredValue, useOptimistic, useTransition } from 'react';
import { useD3Chart, ChartData, ChartDimensions } from './useD3Chart';
import { cn } from '@/lib/utils';
import { IconLoader2, IconTrendingUp, IconTrendingDown, IconMinus } from '@tabler/icons-react';

export interface RealTimeLineChartProps {
  data: ChartData[];
  title?: string;
  className?: string;
  height?: number;
  width?: number;
  showTrend?: boolean;
  showStats?: boolean;
  updateInterval?: number;
  maxDataPoints?: number;
  enableAnimations?: boolean;
  responsive?: boolean;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'destructive';
  onDataPointHover?: (dataPoint: ChartData | null) => void;
  formatValue?: (value: number) => string;
  formatTimestamp?: (timestamp: number) => string;
}

interface OptimisticUpdate {
  action: 'add' | 'update' | 'clear';
  data?: ChartData[];
}

// Color mapping for different chart themes
const colorClasses = {
  primary: 'hsl(var(--primary))',
  secondary: 'hsl(var(--secondary))',
  success: 'hsl(var(--success))',
  warning: 'hsl(var(--warning))',
  destructive: 'hsl(var(--destructive))',
};

// Calculate trend from data
const calculateTrend = (data: ChartData[]): 'up' | 'down' | 'stable' => {
  if (data.length < 2) return 'stable';

  const recent = data.slice(-5); // Last 5 points
  if (recent.length < 2) return 'stable';

  const first = recent[0].value;
  const last = recent[recent.length - 1].value;
  const change = ((last - first) / first) * 100;

  if (Math.abs(change) < 1) return 'stable';
  return change > 0 ? 'up' : 'down';
};

// Calculate basic statistics
const calculateStats = (data: ChartData[]) => {
  if (data.length === 0) return { min: 0, max: 0, avg: 0, latest: 0 };

  const values = data.map(d => d.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
  const latest = values[values.length - 1];

  return { min, max, avg, latest };
};

// Default formatters
const defaultValueFormatter = (value: number): string => {
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
};

const defaultTimestampFormatter = (timestamp: number): string => {
  return new Date(timestamp).toLocaleTimeString();
};

export const RealTimeLineChart: React.FC<RealTimeLineChartProps> = ({
  data: rawData,
  title,
  className,
  height = 300,
  width = 600,
  showTrend = true,
  showStats = true,
  updateInterval = 100,
  maxDataPoints = 1000,
  enableAnimations = true,
  responsive = true,
  color = 'primary',
  onDataPointHover,
  formatValue = defaultValueFormatter,
  formatTimestamp = defaultTimestampFormatter,
}) => {
  const [isPending, startTransition] = useTransition();

  // Optimistic updates for smooth UI
  const [optimisticData, updateOptimisticData] = useOptimistic(
    rawData,
    (state: ChartData[], update: OptimisticUpdate) => {
      switch (update.action) {
        case 'add':
          if (update.data) {
            const newData = [...state, ...update.data];
            return newData.slice(-maxDataPoints);
          }
          return state;
        case 'update':
          return update.data || state;
        case 'clear':
          return [];
        default:
          return state;
      }
    }
  );

  // Defer expensive calculations for better performance
  const deferredData = useDeferredValue(optimisticData);

  // Chart configuration
  const chartConfig = useMemo(() => {
    const dimensions: ChartDimensions = {
      width,
      height,
      margin: { top: 20, right: 30, bottom: 40, left: 50 },
    };

    return {
      dimensions,
      animationDuration: enableAnimations ? 300 : 0,
      enableTransitions: enableAnimations,
      responsive,
      maxDataPoints,
      updateInterval,
    };
  }, [width, height, enableAnimations, responsive, maxDataPoints, updateInterval]);

  // D3 chart hook
  const {
    svgRef,
    containerRef,
    updateChart,
    clearChart,
    isUpdating,
    dimensions,
  } = useD3Chart(chartConfig);

  // Calculate derived data
  const stats = useMemo(() => calculateStats(deferredData), [deferredData]);
  const trend = useMemo(() => calculateTrend(deferredData), [deferredData]);

  // Update chart when data changes
  useEffect(() => {
    if (deferredData.length > 0) {
      updateChart(deferredData);
    }
  }, [deferredData, updateChart]);

  // Handle optimistic updates when new data arrives
  useEffect(() => {
    if (rawData !== optimisticData) {
      startTransition(() => {
        updateOptimisticData({ action: 'update', data: rawData });
      });
    }
  }, [rawData, optimisticData, updateOptimisticData, startTransition]);

  // Mouse event handlers for interactivity
  const handleMouseMove = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!onDataPointHover || deferredData.length === 0) return;

    const svgRect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - svgRect.left - dimensions.margin.left;
    const y = event.clientY - svgRect.top - dimensions.margin.top;

    const chartWidth = dimensions.width - dimensions.margin.left - dimensions.margin.right;
    const chartHeight = dimensions.height - dimensions.margin.top - dimensions.margin.bottom;

    // Check if mouse is within chart area
    if (x < 0 || x > chartWidth || y < 0 || y > chartHeight) {
      onDataPointHover(null);
      return;
    }

    // Find closest data point
    const timeRange = deferredData[deferredData.length - 1].timestamp - deferredData[0].timestamp;
    const relativeX = x / chartWidth;
    const targetTime = deferredData[0].timestamp + (timeRange * relativeX);

    let closestPoint = deferredData[0];
    let minDistance = Math.abs(closestPoint.timestamp - targetTime);

    for (const point of deferredData) {
      const distance = Math.abs(point.timestamp - targetTime);
      if (distance < minDistance) {
        minDistance = distance;
        closestPoint = point;
      }
    }

    onDataPointHover(closestPoint);
  }, [onDataPointHover, deferredData, dimensions]);

  const handleMouseLeave = useCallback(() => {
    if (onDataPointHover) {
      onDataPointHover(null);
    }
  }, [onDataPointHover]);

  // Trend icon component
  const TrendIcon = trend === 'up' ? IconTrendingUp : trend === 'down' ? IconTrendingDown : IconMinus;
  const trendColor = trend === 'up' ? 'text-success' : trend === 'down' ? 'text-destructive' : 'text-muted-foreground';

  return (
    <div className={cn('relative w-full', className)}>
      {/* Header */}
      {(title || showTrend || showStats) && (
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {title && (
              <h3 className="text-lg font-semibold text-foreground">{title}</h3>
            )}
            {isUpdating && (
              <IconLoader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>

          {showTrend && (
            <div className={cn('flex items-center space-x-1', trendColor)}>
              <TrendIcon className="h-4 w-4" />
              <span className="text-sm font-medium capitalize">{trend}</span>
            </div>
          )}
        </div>
      )}

      {/* Statistics */}
      {showStats && (
        <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground">Latest</div>
            <div className="text-sm font-semibold">{formatValue(stats.latest)}</div>
          </div>
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground">Average</div>
            <div className="text-sm font-semibold">{formatValue(stats.avg)}</div>
          </div>
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground">Min</div>
            <div className="text-sm font-semibold">{formatValue(stats.min)}</div>
          </div>
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground">Max</div>
            <div className="text-sm font-semibold">{formatValue(stats.max)}</div>
          </div>
        </div>
      )}

      {/* Chart Container */}
      <div
        ref={containerRef}
        className="relative overflow-hidden rounded-lg border bg-card">
        <svg
          ref={svgRef}
          className="w-full h-auto"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            '--chart-color': colorClasses[color],
          } as React.CSSProperties}
        />

        {/* Loading overlay */}
        {isPending && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50">
            <IconLoader2 className="h-6 w-6 animate-spin text-primary" />
          </div>
        )}

        {/* No data state */}
        {deferredData.length === 0 && !isPending && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <div className="text-sm">No data available</div>
              <div className="text-xs">Waiting for real-time updates...</div>
            </div>
          </div>
        )}
      </div>

      {/* Data info */}
      <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
        <span>{deferredData.length} data points</span>
        {deferredData.length > 0 && (
          <span>
            Last update: {formatTimestamp(deferredData[deferredData.length - 1].timestamp)}
          </span>
        )}
      </div>
    </div>
  );
};

export default RealTimeLineChart;
