/**
 * Real-Time Area Chart Component
 *
 * High-performance area chart with gradient fills for traffic volume visualization.
 * Optimized for real-time data updates with smooth animations.
 */

'use client';

import React, { useEffect, useMemo, useCallback } from 'react';
import { useDeferredValue, useOptimistic, useTransition } from 'react';
import * as d3 from 'd3';
import { useD3Chart, ChartData, ChartDimensions } from './useD3Chart';
import { cn } from '@/lib/utils';
import { IconLoader2 } from '@tabler/icons-react';

export interface RealTimeAreaChartProps {
  data: ChartData[];
  title?: string;
  className?: string;
  height?: number;
  width?: number;
  showGrid?: boolean;
  enableAnimations?: boolean;
  responsive?: boolean;
  gradientColors?: [string, string]; // [start, end] gradient colors
  strokeColor?: string;
  fillOpacity?: number;
  maxDataPoints?: number;
  updateInterval?: number;
  formatValue?: (value: number) => string;
  formatTimestamp?: (timestamp: number) => string;
}

interface OptimisticUpdate {
  action: 'add' | 'update' | 'clear';
  data?: ChartData[];
}

// Extended useD3Chart hook specifically for area charts
const useD3AreaChart = (config: any) => {
  const baseHook = useD3Chart(config);

  const updateAreaChart = useCallback((data: ChartData[], options: {
    gradientColors?: [string, string];
    strokeColor?: string;
    fillOpacity?: number;
    showGrid?: boolean;
  }) => {
    if (!baseHook.svgRef.current || data.length === 0) return;

    const svg = d3.select(baseHook.svgRef.current);
    const { width, height, margin } = baseHook.dimensions;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Decimate data for performance
    const maxPoints = config.maxDataPoints || 1000;
    const processedData = data.length > maxPoints
      ? data.filter((_, i) => i % Math.ceil(data.length / maxPoints) === 0)
      : data;

    // Create scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(processedData, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, chartWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(processedData, d => d.value) || 0])
      .nice()
      .range([chartHeight, 0]);

    // Create area generator
    const area = d3.area<ChartData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(chartHeight)
      .y1(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Create line generator for stroke
    const line = d3.line<ChartData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    const chartGroup = svg.select('.chart-group');

    // Create gradient definition
    const gradient = svg.select('defs')
      .selectAll('.area-gradient')
      .data([1])
      .join('linearGradient')
      .attr('class', 'area-gradient')
      .attr('id', 'area-gradient')
      .attr('gradientUnits', 'objectBoundingBox')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', 0).attr('y2', 1);

    gradient.selectAll('stop')
      .data([
        { offset: '0%', color: options.gradientColors?.[0] || 'hsl(var(--primary))' },
        { offset: '100%', color: options.gradientColors?.[1] || 'transparent' }
      ])
      .join('stop')
      .attr('offset', d => d.offset)
      .attr('stop-color', d => d.color)
      .attr('stop-opacity', (d, i) => i === 0 ? (options.fillOpacity || 0.6) : 0);

    // Update axes
    chartGroup.select('.x-axis')
      .attr('transform', `translate(0,${chartHeight})`)
      .transition()
      .duration(config.animationDuration || 300)
      .call(d3.axisBottom(xScale)
        .tickFormat(d3.timeFormat('%H:%M'))
        .ticks(6)
      );

    chartGroup.select('.y-axis')
      .transition()
      .duration(config.animationDuration || 300)
      .call(d3.axisLeft(yScale)
        .ticks(5)
        .tickFormat(d3.format('.1s'))
      );

    // Update grid
    if (options.showGrid) {
      chartGroup.select('.grid')
        .selectAll('.grid-line')
        .data(yScale.ticks(5))
        .join(
          enter => enter
            .append('line')
            .attr('class', 'grid-line')
            .attr('x1', 0)
            .attr('x2', chartWidth)
            .attr('y1', d => yScale(d))
            .attr('y2', d => yScale(d))
            .attr('stroke', 'hsl(var(--border))')
            .attr('stroke-opacity', 0.2)
            .attr('stroke-width', 1),
          update => update
            .transition()
            .duration(config.animationDuration || 300)
            .attr('y1', d => yScale(d))
            .attr('y2', d => yScale(d)),
          exit => exit.remove()
        );
    }

    // Update area path
    const dataGroup = chartGroup.select('.chart-data');

    dataGroup.selectAll('.area-path')
      .data([processedData])
      .join(
        enter => enter
          .append('path')
          .attr('class', 'area-path')
          .attr('fill', 'url(#area-gradient)')
          .attr('d', area),
        update => {
          if (config.enableTransitions) {
            return update
              .transition()
              .duration(config.animationDuration || 300)
              .attr('d', area);
          } else {
            return update.attr('d', area);
          }
        },
        exit => exit.remove()
      );

    // Update stroke line
    dataGroup.selectAll('.area-stroke')
      .data([processedData])
      .join(
        enter => enter
          .append('path')
          .attr('class', 'area-stroke')
          .attr('fill', 'none')
          .attr('stroke', options.strokeColor || 'hsl(var(--primary))')
          .attr('stroke-width', 2)
          .attr('d', line),
        update => {
          if (config.enableTransitions) {
            return update
              .transition()
              .duration(config.animationDuration || 300)
              .attr('d', line);
          } else {
            return update.attr('d', line);
          }
        },
        exit => exit.remove()
      );
  }, [baseHook.svgRef, baseHook.dimensions, config]);

  return {
    ...baseHook,
    updateAreaChart,
  };
};

export const RealTimeAreaChart: React.FC<RealTimeAreaChartProps> = ({
  data: rawData,
  title,
  className,
  height = 300,
  width = 600,
  showGrid = true,
  enableAnimations = true,
  responsive = true,
  gradientColors,
  strokeColor,
  fillOpacity = 0.6,
  maxDataPoints = 1000,
  updateInterval = 100,
  formatValue = (value: number) => value.toLocaleString(),
  formatTimestamp = (timestamp: number) => new Date(timestamp).toLocaleTimeString(),
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

  // Defer expensive calculations
  const deferredData = useDeferredValue(optimisticData);

  // Chart configuration
  const chartConfig = useMemo(() => {
    const dimensions: ChartDimensions = {
      width,
      height,
      margin: { top: 20, right: 30, bottom: 40, left: 60 },
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

  // Extended D3 area chart hook
  const {
    svgRef,
    containerRef,
    updateAreaChart,
    clearChart,
    isUpdating,
  } = useD3AreaChart(chartConfig);

  // Update chart when data changes
  useEffect(() => {
    if (deferredData.length > 0) {
      updateAreaChart(deferredData, {
        gradientColors,
        strokeColor,
        fillOpacity,
        showGrid,
      });
    }
  }, [deferredData, updateAreaChart, gradientColors, strokeColor, fillOpacity, showGrid]);

  // Handle optimistic updates
  useEffect(() => {
    if (rawData !== optimisticData) {
      startTransition(() => {
        updateOptimisticData({ action: 'update', data: rawData });
      });
    }
  }, [rawData, optimisticData, updateOptimisticData, startTransition]);

  // Calculate current value
  const currentValue = deferredData.length > 0 ? deferredData[deferredData.length - 1].value : 0;

  return (
    <div className={cn('relative w-full', className)}>
      {/* Header */}
      {title && (
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <h3 className="text-lg font-semibold text-foreground">{title}</h3>
            {isUpdating && (
              <IconLoader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>

          <div className="text-right">
            <div className="text-2xl font-bold text-foreground">
              {formatValue(currentValue)}
            </div>
            <div className="text-xs text-muted-foreground">
              Current Value
            </div>
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

export default RealTimeAreaChart;
