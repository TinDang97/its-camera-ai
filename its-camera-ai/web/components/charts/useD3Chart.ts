/**
 * D3.js Chart Hook for React 19+ Integration
 *
 * Provides optimized D3.js integration with React 19 concurrent features
 * for high-performance real-time data visualization.
 */

'use client';

import { useRef, useEffect, useCallback, useMemo, useTransition } from 'react';
import * as d3 from 'd3';
import { useDeferredValue } from 'react';

export interface ChartDimensions {
  width: number;
  height: number;
  margin: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

export interface D3ChartConfig {
  dimensions: ChartDimensions;
  animationDuration?: number;
  enableTransitions?: boolean;
  responsive?: boolean;
  maxDataPoints?: number;
  updateInterval?: number;
}

export interface ChartData {
  timestamp: number;
  value: number;
  metadata?: Record<string, any>;
}

export interface D3ChartHookReturn {
  svgRef: React.RefObject<SVGSVGElement>;
  containerRef: React.RefObject<HTMLDivElement>;
  updateChart: (data: ChartData[]) => void;
  clearChart: () => void;
  resizeChart: () => void;
  isUpdating: boolean;
  dimensions: ChartDimensions;
}

// Data decimation utility for performance optimization
const decimateData = (data: ChartData[], maxPoints: number): ChartData[] => {
  if (data.length <= maxPoints) return data;

  const step = Math.ceil(data.length / maxPoints);
  const decimated: ChartData[] = [];

  for (let i = 0; i < data.length; i += step) {
    // Use the last point in each step for more recent data
    const endIndex = Math.min(i + step - 1, data.length - 1);
    decimated.push(data[endIndex]);
  }

  // Always include the last data point
  if (decimated[decimated.length - 1] !== data[data.length - 1]) {
    decimated.push(data[data.length - 1]);
  }

  return decimated;
};

// Responsive dimension calculator
const calculateResponsiveDimensions = (
  container: HTMLDivElement | null,
  baseConfig: ChartDimensions
): ChartDimensions => {
  if (!container) return baseConfig;

  const containerRect = container.getBoundingClientRect();
  const aspectRatio = baseConfig.height / baseConfig.width;

  return {
    width: Math.max(300, containerRect.width - 20), // Min width with padding
    height: Math.max(200, (containerRect.width - 20) * aspectRatio), // Maintain aspect ratio
    margin: {
      ...baseConfig.margin,
      // Adjust margins for smaller screens
      left: containerRect.width < 400 ? 40 : baseConfig.margin.left,
      right: containerRect.width < 400 ? 20 : baseConfig.margin.right,
    },
  };
};

export const useD3Chart = (config: D3ChartConfig): D3ChartHookReturn => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPending, startTransition] = useTransition();

  // Responsive dimensions calculation
  const dimensions = useMemo(() => {
    if (config.responsive) {
      return calculateResponsiveDimensions(containerRef.current, config.dimensions);
    }
    return config.dimensions;
  }, [config.dimensions, config.responsive]);

  // Initialize chart structure
  const initializeChart = useCallback(() => {
    const svg = d3.select(svgRef.current);
    const { width, height, margin } = dimensions;

    // Clear existing content
    svg.selectAll('*').remove();

    // Set SVG dimensions
    svg
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // Create main group with margins
    const chartGroup = svg
      .append('g')
      .attr('class', 'chart-group')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create clipping path for chart area
    svg
      .append('defs')
      .append('clipPath')
      .attr('id', 'chart-clip')
      .append('rect')
      .attr('width', width - margin.left - margin.right)
      .attr('height', height - margin.top - margin.bottom);

    // Create chart containers
    chartGroup.append('g').attr('class', 'chart-data').attr('clip-path', 'url(#chart-clip)');
    chartGroup.append('g').attr('class', 'x-axis');
    chartGroup.append('g').attr('class', 'y-axis');
    chartGroup.append('g').attr('class', 'grid');

    return chartGroup;
  }, [dimensions]);

  // Update chart with new data
  const updateChart = useCallback((data: ChartData[]) => {
    if (!svgRef.current || data.length === 0) return;

    startTransition(() => {
      const svg = d3.select(svgRef.current);
      const { width, height, margin } = dimensions;
      const chartWidth = width - margin.left - margin.right;
      const chartHeight = height - margin.top - margin.bottom;

      // Decimate data for performance
      const processedData = decimateData(data, config.maxDataPoints || 1000);

      // Create scales
      const xScale = d3.scaleTime()
        .domain(d3.extent(processedData, d => new Date(d.timestamp)) as [Date, Date])
        .range([0, chartWidth]);

      const yScale = d3.scaleLinear()
        .domain(d3.extent(processedData, d => d.value) as [number, number])
        .nice()
        .range([chartHeight, 0]);

      // Create line generator
      const line = d3.line<ChartData>()
        .x(d => xScale(new Date(d.timestamp)))
        .y(d => yScale(d.value))
        .curve(d3.curveMonotoneX);

      // Update axes
      const chartGroup = svg.select('.chart-group');

      // X-axis
      chartGroup.select('.x-axis')
        .attr('transform', `translate(0,${chartHeight})`)
        .transition()
        .duration(config.animationDuration || 300)
        .call(d3.axisBottom(xScale)
          .tickFormat(d3.timeFormat('%H:%M:%S'))
          .ticks(5)
        );

      // Y-axis
      chartGroup.select('.y-axis')
        .transition()
        .duration(config.animationDuration || 300)
        .call(d3.axisLeft(yScale)
          .ticks(5)
        );

      // Grid lines
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
            .attr('stroke-opacity', 0.3)
            .attr('stroke-width', 1),
          update => update
            .transition()
            .duration(config.animationDuration || 300)
            .attr('y1', d => yScale(d))
            .attr('y2', d => yScale(d)),
          exit => exit.remove()
        );

      // Update data line
      const dataGroup = chartGroup.select('.chart-data');

      dataGroup.selectAll('.data-line')
        .data([processedData])
        .join(
          enter => enter
            .append('path')
            .attr('class', 'data-line')
            .attr('fill', 'none')
            .attr('stroke', 'hsl(var(--primary))')
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

      // Add data points
      dataGroup.selectAll('.data-point')
        .data(processedData.slice(-10)) // Show only last 10 points
        .join(
          enter => enter
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(new Date(d.timestamp)))
            .attr('cy', d => yScale(d.value))
            .attr('r', 3)
            .attr('fill', 'hsl(var(--primary))')
            .attr('opacity', 0)
            .transition()
            .duration(config.animationDuration || 300)
            .attr('opacity', 1),
          update => {
            if (config.enableTransitions) {
              return update
                .transition()
                .duration(config.animationDuration || 300)
                .attr('cx', d => xScale(new Date(d.timestamp)))
                .attr('cy', d => yScale(d.value));
            } else {
              return update
                .attr('cx', d => xScale(new Date(d.timestamp)))
                .attr('cy', d => yScale(d.value));
            }
          },
          exit => exit
            .transition()
            .duration(config.animationDuration || 300)
            .attr('opacity', 0)
            .remove()
        );
    });
  }, [dimensions, config, startTransition]);

  // Clear chart
  const clearChart = useCallback(() => {
    if (svgRef.current) {
      d3.select(svgRef.current).selectAll('*').remove();
      initializeChart();
    }
  }, [initializeChart]);

  // Resize chart
  const resizeChart = useCallback(() => {
    if (config.responsive) {
      initializeChart();
    }
  }, [config.responsive, initializeChart]);

  // Initialize chart on mount
  useEffect(() => {
    initializeChart();
  }, [initializeChart]);

  // Handle responsive resizing
  useEffect(() => {
    if (!config.responsive) return;

    const handleResize = () => {
      resizeChart();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [config.responsive, resizeChart]);

  return {
    svgRef,
    containerRef,
    updateChart,
    clearChart,
    resizeChart,
    isUpdating: isPending,
    dimensions,
  };
};

export default useD3Chart;
