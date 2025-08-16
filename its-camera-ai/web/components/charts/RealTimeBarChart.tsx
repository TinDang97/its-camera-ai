/**
 * Real-Time Bar Chart Component
 *
 * High-performance animated bar chart for categorical real-time data visualization.
 * Supports horizontal and vertical orientations with smooth transitions.
 */

'use client';

import React, { useEffect, useMemo, useCallback } from 'react';
import { useDeferredValue, useOptimistic, useTransition } from 'react';
import * as d3 from 'd3';
import { useD3Chart, ChartDimensions } from './useD3Chart';
import { cn } from '@/lib/utils';
import { IconLoader2 } from '@tabler/icons-react';

export interface BarChartData {
  category: string;
  value: number;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface RealTimeBarChartProps {
  data: BarChartData[];
  title?: string;
  className?: string;
  height?: number;
  width?: number;
  orientation?: 'vertical' | 'horizontal';
  showValues?: boolean;
  showGrid?: boolean;
  enableAnimations?: boolean;
  responsive?: boolean;
  barColor?: string;
  maxBars?: number;
  sortBars?: 'asc' | 'desc' | 'none';
  updateInterval?: number;
  formatValue?: (value: number) => string;
  formatCategory?: (category: string) => string;
  onBarClick?: (data: BarChartData) => void;
}

interface OptimisticUpdate {
  action: 'add' | 'update' | 'clear';
  data?: BarChartData[];
}

// Extended hook for bar charts
const useD3BarChart = (config: any) => {
  const baseHook = useD3Chart(config);

  const updateBarChart = useCallback((data: BarChartData[], options: {
    orientation?: 'vertical' | 'horizontal';
    showValues?: boolean;
    showGrid?: boolean;
    barColor?: string;
    sortBars?: 'asc' | 'desc' | 'none';
    onBarClick?: (data: BarChartData) => void;
  }) => {
    if (!baseHook.svgRef.current || data.length === 0) return;

    const svg = d3.select(baseHook.svgRef.current);
    const { width, height, margin } = baseHook.dimensions;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Sort data if requested
    let processedData = [...data];
    if (options.sortBars === 'asc') {
      processedData.sort((a, b) => a.value - b.value);
    } else if (options.sortBars === 'desc') {
      processedData.sort((a, b) => b.value - a.value);
    }

    const isVertical = options.orientation === 'vertical';

    // Create scales
    const categoryScale = d3.scaleBand()
      .domain(processedData.map(d => d.category))
      .range(isVertical ? [0, chartWidth] : [0, chartHeight])
      .padding(0.1);

    const valueScale = d3.scaleLinear()
      .domain([0, d3.max(processedData, d => d.value) || 0])
      .nice()
      .range(isVertical ? [chartHeight, 0] : [0, chartWidth]);

    const chartGroup = svg.select('.chart-group');

    // Update axes
    if (isVertical) {
      // X-axis (categories)
      chartGroup.select('.x-axis')
        .attr('transform', `translate(0,${chartHeight})`)
        .transition()
        .duration(config.animationDuration || 300)
        .call(d3.axisBottom(categoryScale)
          .tickFormat(d => d.length > 10 ? d.slice(0, 10) + '...' : d)
        );

      // Y-axis (values)
      chartGroup.select('.y-axis')
        .transition()
        .duration(config.animationDuration || 300)
        .call(d3.axisLeft(valueScale)
          .ticks(5)
          .tickFormat(d3.format('.1s'))
        );
    } else {
      // X-axis (values)
      chartGroup.select('.x-axis')
        .attr('transform', `translate(0,${chartHeight})`)
        .transition()
        .duration(config.animationDuration || 300)
        .call(d3.axisBottom(valueScale)
          .ticks(5)
          .tickFormat(d3.format('.1s'))
        );

      // Y-axis (categories)
      chartGroup.select('.y-axis')
        .transition()
        .duration(config.animationDuration || 300)
        .call(d3.axisLeft(categoryScale)
          .tickFormat(d => d.length > 15 ? d.slice(0, 15) + '...' : d)
        );
    }

    // Update grid
    if (options.showGrid) {
      const gridTicks = isVertical ? valueScale.ticks(5) : categoryScale.domain();

      chartGroup.select('.grid')
        .selectAll('.grid-line')
        .data(gridTicks)
        .join(
          enter => {
            const line = enter
              .append('line')
              .attr('class', 'grid-line')
              .attr('stroke', 'hsl(var(--border))')
              .attr('stroke-opacity', 0.2)
              .attr('stroke-width', 1);

            if (isVertical) {
              line
                .attr('x1', 0)
                .attr('x2', chartWidth)
                .attr('y1', d => valueScale(d as number))
                .attr('y2', d => valueScale(d as number));
            } else {
              line
                .attr('x1', d => categoryScale(d as string)! + categoryScale.bandwidth() / 2)
                .attr('x2', d => categoryScale(d as string)! + categoryScale.bandwidth() / 2)
                .attr('y1', 0)
                .attr('y2', chartHeight);
            }

            return line;
          },
          update => {
            if (isVertical) {
              return update
                .transition()
                .duration(config.animationDuration || 300)
                .attr('y1', d => valueScale(d as number))
                .attr('y2', d => valueScale(d as number));
            } else {
              return update
                .transition()
                .duration(config.animationDuration || 300)
                .attr('x1', d => categoryScale(d as string)! + categoryScale.bandwidth() / 2)
                .attr('x2', d => categoryScale(d as string)! + categoryScale.bandwidth() / 2);
            }
          },
          exit => exit.remove()
        );
    }

    // Update bars
    const dataGroup = chartGroup.select('.chart-data');

    const bars = dataGroup.selectAll('.bar')
      .data(processedData, (d: any) => d.category);

    // Enter
    const barsEnter = bars.enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('fill', options.barColor || 'hsl(var(--primary))')
      .attr('cursor', options.onBarClick ? 'pointer' : 'default');

    if (isVertical) {
      barsEnter
        .attr('x', d => categoryScale(d.category)!)
        .attr('y', chartHeight)
        .attr('width', categoryScale.bandwidth())
        .attr('height', 0);
    } else {
      barsEnter
        .attr('x', 0)
        .attr('y', d => categoryScale(d.category)!)
        .attr('width', 0)
        .attr('height', categoryScale.bandwidth());
    }

    // Update
    const barsUpdate = barsEnter.merge(bars);

    if (config.enableTransitions) {
      const transition = barsUpdate
        .transition()
        .duration(config.animationDuration || 300);

      if (isVertical) {
        transition
          .attr('x', d => categoryScale(d.category)!)
          .attr('y', d => valueScale(d.value))
          .attr('width', categoryScale.bandwidth())
          .attr('height', d => chartHeight - valueScale(d.value));
      } else {
        transition
          .attr('x', 0)
          .attr('y', d => categoryScale(d.category)!)
          .attr('width', d => valueScale(d.value))
          .attr('height', categoryScale.bandwidth());
      }
    } else {
      if (isVertical) {
        barsUpdate
          .attr('x', d => categoryScale(d.category)!)
          .attr('y', d => valueScale(d.value))
          .attr('width', categoryScale.bandwidth())
          .attr('height', d => chartHeight - valueScale(d.value));
      } else {
        barsUpdate
          .attr('x', 0)
          .attr('y', d => categoryScale(d.category)!)
          .attr('width', d => valueScale(d.value))
          .attr('height', categoryScale.bandwidth());
      }
    }

    // Add click handlers
    if (options.onBarClick) {
      barsUpdate.on('click', (event, d) => {
        options.onBarClick!(d);
      });
    }

    // Exit
    bars.exit()
      .transition()
      .duration(config.animationDuration || 300)
      .attr(isVertical ? 'height' : 'width', 0)
      .remove();

    // Add value labels if requested
    if (options.showValues) {
      const labels = dataGroup.selectAll('.bar-label')
        .data(processedData, (d: any) => d.category);

      // Enter
      const labelsEnter = labels.enter()
        .append('text')
        .attr('class', 'bar-label')
        .attr('fill', 'hsl(var(--foreground))')
        .attr('font-size', '12px')
        .attr('font-weight', '500')
        .attr('text-anchor', isVertical ? 'middle' : 'start');

      // Update
      const labelsUpdate = labelsEnter.merge(labels);

      if (config.enableTransitions) {
        const transition = labelsUpdate
          .transition()
          .duration(config.animationDuration || 300);

        if (isVertical) {
          transition
            .attr('x', d => categoryScale(d.category)! + categoryScale.bandwidth() / 2)
            .attr('y', d => valueScale(d.value) - 5)
            .text(d => d3.format('.1s')(d.value));
        } else {
          transition
            .attr('x', d => valueScale(d.value) + 5)
            .attr('y', d => categoryScale(d.category)! + categoryScale.bandwidth() / 2 + 4)
            .text(d => d3.format('.1s')(d.value));
        }
      } else {
        if (isVertical) {
          labelsUpdate
            .attr('x', d => categoryScale(d.category)! + categoryScale.bandwidth() / 2)
            .attr('y', d => valueScale(d.value) - 5)
            .text(d => d3.format('.1s')(d.value));
        } else {
          labelsUpdate
            .attr('x', d => valueScale(d.value) + 5)
            .attr('y', d => categoryScale(d.category)! + categoryScale.bandwidth() / 2 + 4)
            .text(d => d3.format('.1s')(d.value));
        }
      }

      // Exit
      labels.exit().remove();
    }
  }, [baseHook.svgRef, baseHook.dimensions, config]);

  return {
    ...baseHook,
    updateBarChart,
  };
};

export const RealTimeBarChart: React.FC<RealTimeBarChartProps> = ({
  data: rawData,
  title,
  className,
  height = 300,
  width = 600,
  orientation = 'vertical',
  showValues = true,
  showGrid = true,
  enableAnimations = true,
  responsive = true,
  barColor,
  maxBars = 20,
  sortBars = 'desc',
  updateInterval = 100,
  formatValue = (value: number) => value.toLocaleString(),
  formatCategory = (category: string) => category,
  onBarClick,
}) => {
  const [isPending, startTransition] = useTransition();

  // Optimistic updates
  const [optimisticData, updateOptimisticData] = useOptimistic(
    rawData,
    (state: BarChartData[], update: OptimisticUpdate) => {
      switch (update.action) {
        case 'add':
          if (update.data) {
            return [...state, ...update.data].slice(-maxBars);
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
      margin: {
        top: 20,
        right: 30,
        bottom: orientation === 'vertical' ? 60 : 40,
        left: orientation === 'vertical' ? 60 : 100
      },
    };

    return {
      dimensions,
      animationDuration: enableAnimations ? 300 : 0,
      enableTransitions: enableAnimations,
      responsive,
      updateInterval,
    };
  }, [width, height, orientation, enableAnimations, responsive, updateInterval]);

  // Extended D3 bar chart hook
  const {
    svgRef,
    containerRef,
    updateBarChart,
    clearChart,
    isUpdating,
  } = useD3BarChart(chartConfig);

  // Update chart when data changes
  useEffect(() => {
    if (deferredData.length > 0) {
      updateBarChart(deferredData, {
        orientation,
        showValues,
        showGrid,
        barColor,
        sortBars,
        onBarClick,
      });
    }
  }, [deferredData, updateBarChart, orientation, showValues, showGrid, barColor, sortBars, onBarClick]);

  // Handle optimistic updates
  useEffect(() => {
    if (rawData !== optimisticData) {
      startTransition(() => {
        updateOptimisticData({ action: 'update', data: rawData });
      });
    }
  }, [rawData, optimisticData, updateOptimisticData, startTransition]);

  // Calculate totals
  const totalValue = deferredData.reduce((sum, d) => sum + d.value, 0);
  const maxValue = Math.max(...deferredData.map(d => d.value));

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
            <div className="text-sm text-muted-foreground">Total: {formatValue(totalValue)}</div>
            <div className="text-xs text-muted-foreground">Max: {formatValue(maxValue)}</div>
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
        <span>{deferredData.length} categories</span>
        <span>Sort: {sortBars === 'none' ? 'Original' : sortBars === 'asc' ? 'Ascending' : 'Descending'}</span>
      </div>
    </div>
  );
};

export default RealTimeBarChart;
