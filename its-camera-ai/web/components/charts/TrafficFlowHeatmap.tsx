/**
 * Traffic Flow Heatmap Component
 *
 * Geographic visualization of traffic density and flow patterns using D3.js.
 * Shows real-time traffic intensity across camera locations with interactive features.
 */

'use client';

import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import { useDeferredValue, useTransition } from 'react';
import * as d3 from 'd3';
import { cn } from '@/lib/utils';
import { useD3Chart, type D3ChartConfig } from './useD3Chart';
import { ChartDataProcessor, ChartPerformance } from './utils';

export interface TrafficFlowData {
  cameraId: string;
  location: {
    lat: number;
    lng: number;
    name: string;
  };
  intensity: number; // 0-100 traffic density
  speed: number;
  vehicleCount: number;
  timestamp: number;
  congestionLevel: 'free' | 'light' | 'moderate' | 'heavy' | 'severe';
  incidents?: Array<{
    id: string;
    type: 'accident' | 'congestion' | 'roadwork';
    severity: 'low' | 'medium' | 'high' | 'critical';
  }>;
}

export interface TrafficFlowHeatmapProps {
  data: TrafficFlowData[];
  width?: number;
  height?: number;
  className?: string;
  showLegend?: boolean;
  showCameraLabels?: boolean;
  enableInteraction?: boolean;
  colorScheme?: 'traffic' | 'speed' | 'density';
  onLocationClick?: (data: TrafficFlowData) => void;
  onLocationHover?: (data: TrafficFlowData | null) => void;
  zoomLevel?: number;
  centerLocation?: { lat: number; lng: number };
  showTrafficFlow?: boolean;
  animationDuration?: number;
}

// Color scales for different metrics
const COLOR_SCHEMES = {
  traffic: d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 100]),
  speed: d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 80]),
  density: d3.scaleSequential(d3.interpolateViridis).domain([0, 100]),
} as const;

// Congestion level configurations
const CONGESTION_CONFIG = {
  free: { color: '#22c55e', radius: 8, opacity: 0.7 },
  light: { color: '#84cc16', radius: 10, opacity: 0.8 },
  moderate: { color: '#eab308', radius: 12, opacity: 0.8 },
  heavy: { color: '#f97316', radius: 14, opacity: 0.9 },
  severe: { color: '#ef4444', radius: 16, opacity: 1.0 },
};

export const TrafficFlowHeatmap: React.FC<TrafficFlowHeatmapProps> = ({
  data,
  width = 800,
  height = 500,
  className,
  showLegend = true,
  showCameraLabels = true,
  enableInteraction = true,
  colorScheme = 'traffic',
  onLocationClick,
  onLocationHover,
  zoomLevel = 1,
  centerLocation = { lat: 37.7749, lng: -122.4194 }, // Default to San Francisco
  showTrafficFlow = true,
  animationDuration = 500,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [isPending, startTransition] = useTransition();
  const deferredData = useDeferredValue(data);

  // Chart configuration
  const chartConfig: D3ChartConfig = useMemo(() => ({
    width,
    height,
    margin: { top: 20, right: 20, bottom: 40, left: 20 },
    enableAnimations: true,
    responsive: true,
  }), [width, height]);

  const { svgRef: d3SvgRef, dimensions } = useD3Chart(chartConfig);

  // Process data for visualization
  const processedData = useMemo(() => {
    const processor = new ChartDataProcessor();

    // Calculate bounds for the geographic area
    const latitudes = deferredData.map(d => d.location.lat);
    const longitudes = deferredData.map(d => d.location.lng);

    const bounds = {
      north: Math.max(...latitudes),
      south: Math.min(...latitudes),
      east: Math.max(...longitudes),
      west: Math.min(...longitudes),
    };

    // Add padding to bounds
    const latPadding = (bounds.north - bounds.south) * 0.1;
    const lngPadding = (bounds.east - bounds.west) * 0.1;

    return {
      locations: deferredData,
      bounds: {
        north: bounds.north + latPadding,
        south: bounds.south - latPadding,
        east: bounds.east + lngPadding,
        west: bounds.west - lngPadding,
      },
    };
  }, [deferredData]);

  // Create projection for geographic coordinates
  const projection = useMemo(() => {
    if (!processedData.bounds) return null;

    const { bounds } = processedData;

    // Create a simple linear projection for the local area
    const xScale = d3.scaleLinear()
      .domain([bounds.west, bounds.east])
      .range([dimensions.margin.left, dimensions.width - dimensions.margin.right]);

    const yScale = d3.scaleLinear()
      .domain([bounds.south, bounds.north])
      .range([dimensions.height - dimensions.margin.bottom, dimensions.margin.top]);

    return (coords: [number, number]) => [
      xScale(coords[0]), // longitude -> x
      yScale(coords[1])  // latitude -> y
    ];
  }, [processedData.bounds, dimensions]);

  // Color scale based on selected scheme
  const colorScale = useMemo(() => {
    const scale = COLOR_SCHEMES[colorScheme];

    if (colorScheme === 'speed') {
      const maxSpeed = Math.max(...processedData.locations.map(d => d.speed));
      return scale.domain([0, maxSpeed]);
    } else if (colorScheme === 'density') {
      const maxDensity = Math.max(...processedData.locations.map(d => d.vehicleCount));
      return scale.domain([0, maxDensity]);
    }

    return scale; // traffic intensity 0-100
  }, [colorScheme, processedData.locations]);

  // Render heatmap
  const renderHeatmap = useCallback(() => {
    if (!svgRef.current || !projection || !processedData.locations.length) return;

    ChartPerformance.startMeasure('heatmap-render');

    const svg = d3.select(svgRef.current);
    const container = svg.select('.heatmap-container');

    // Create container if it doesn't exist
    if (container.empty()) {
      svg.append('g').attr('class', 'heatmap-container');
    }

    const heatmapContainer = svg.select('.heatmap-container');

    // Clear previous elements
    heatmapContainer.selectAll('*').remove();

    // Create background grid for better context
    if (showTrafficFlow) {
      const gridLines = heatmapContainer.append('g').attr('class', 'grid-lines');

      // Add horizontal grid lines
      const yTicks = d3.scaleLinear()
        .domain([processedData.bounds.south, processedData.bounds.north])
        .ticks(5);

      gridLines.selectAll('.grid-line-horizontal')
        .data(yTicks)
        .enter()
        .append('line')
        .attr('class', 'grid-line-horizontal')
        .attr('x1', dimensions.margin.left)
        .attr('x2', dimensions.width - dimensions.margin.right)
        .attr('y1', d => projection([0, d])[1])
        .attr('y2', d => projection([0, d])[1])
        .attr('stroke', 'hsl(var(--border))')
        .attr('stroke-width', 0.5)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);

      // Add vertical grid lines
      const xTicks = d3.scaleLinear()
        .domain([processedData.bounds.west, processedData.bounds.east])
        .ticks(5);

      gridLines.selectAll('.grid-line-vertical')
        .data(xTicks)
        .enter()
        .append('line')
        .attr('class', 'grid-line-vertical')
        .attr('x1', d => projection([d, 0])[0])
        .attr('x2', d => projection([d, 0])[0])
        .attr('y1', dimensions.margin.top)
        .attr('y2', dimensions.height - dimensions.margin.bottom)
        .attr('stroke', 'hsl(var(--border))')
        .attr('stroke-width', 0.5)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);
    }

    // Create traffic flow circles
    const locations = heatmapContainer.selectAll('.traffic-location')
      .data(processedData.locations, (d: any) => d.cameraId);

    // Enter selection
    const locationsEnter = locations.enter()
      .append('g')
      .attr('class', 'traffic-location')
      .attr('transform', d => {
        const [x, y] = projection([d.location.lng, d.location.lat]);
        return `translate(${x}, ${y})`;
      });

    // Add outer glow circle for severe congestion
    locationsEnter.append('circle')
      .attr('class', 'glow-circle')
      .attr('r', 0)
      .attr('fill', 'none')
      .attr('stroke', d => CONGESTION_CONFIG[d.congestionLevel].color)
      .attr('stroke-width', 2)
      .attr('opacity', 0);

    // Add main traffic circle
    locationsEnter.append('circle')
      .attr('class', 'traffic-circle')
      .attr('r', 0)
      .attr('fill', d => {
        const value = colorScheme === 'speed' ? d.speed :
                     colorScheme === 'density' ? d.vehicleCount :
                     d.intensity;
        return colorScale(value);
      })
      .attr('stroke', d => CONGESTION_CONFIG[d.congestionLevel].color)
      .attr('stroke-width', 1.5)
      .attr('opacity', d => CONGESTION_CONFIG[d.congestionLevel].opacity);

    // Add incident indicators
    locationsEnter.append('circle')
      .attr('class', 'incident-indicator')
      .attr('r', 0)
      .attr('fill', '#ef4444')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .attr('opacity', 0)
      .attr('transform', 'translate(8, -8)');

    // Add camera labels if enabled
    if (showCameraLabels) {
      locationsEnter.append('text')
        .attr('class', 'camera-label')
        .attr('x', 0)
        .attr('y', d => CONGESTION_CONFIG[d.congestionLevel].radius + 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('font-weight', '500')
        .attr('fill', 'hsl(var(--foreground))')
        .attr('opacity', 0)
        .text(d => d.cameraId);
    }

    // Update selection
    const locationsUpdate = locationsEnter.merge(locations);

    // Animate to new positions and styles
    locationsUpdate.transition()
      .duration(animationDuration)
      .attr('transform', d => {
        const [x, y] = projection([d.location.lng, d.location.lat]);
        return `translate(${x}, ${y})`;
      });

    // Update glow circles
    locationsUpdate.select('.glow-circle')
      .transition()
      .duration(animationDuration)
      .attr('r', d => d.congestionLevel === 'severe' ? CONGESTION_CONFIG[d.congestionLevel].radius + 8 : 0)
      .attr('opacity', d => d.congestionLevel === 'severe' ? 0.4 : 0)
      .attr('stroke', d => CONGESTION_CONFIG[d.congestionLevel].color);

    // Update main circles
    locationsUpdate.select('.traffic-circle')
      .transition()
      .duration(animationDuration)
      .attr('r', d => CONGESTION_CONFIG[d.congestionLevel].radius)
      .attr('fill', d => {
        const value = colorScheme === 'speed' ? d.speed :
                     colorScheme === 'density' ? d.vehicleCount :
                     d.intensity;
        return colorScale(value);
      })
      .attr('stroke', d => CONGESTION_CONFIG[d.congestionLevel].color)
      .attr('opacity', d => CONGESTION_CONFIG[d.congestionLevel].opacity);

    // Update incident indicators
    locationsUpdate.select('.incident-indicator')
      .transition()
      .duration(animationDuration)
      .attr('r', d => d.incidents && d.incidents.length > 0 ? 4 : 0)
      .attr('opacity', d => d.incidents && d.incidents.length > 0 ? 1 : 0);

    // Update labels
    if (showCameraLabels) {
      locationsUpdate.select('.camera-label')
        .transition()
        .duration(animationDuration)
        .attr('y', d => CONGESTION_CONFIG[d.congestionLevel].radius + 15)
        .attr('opacity', 0.8)
        .text(d => d.cameraId);
    }

    // Add interaction handlers
    if (enableInteraction) {
      locationsUpdate
        .style('cursor', 'pointer')
        .on('mouseenter', function(event, d) {
          d3.select(this).select('.traffic-circle')
            .transition()
            .duration(150)
            .attr('r', CONGESTION_CONFIG[d.congestionLevel].radius * 1.2)
            .attr('stroke-width', 3);

          onLocationHover?.(d);
        })
        .on('mouseleave', function(event, d) {
          d3.select(this).select('.traffic-circle')
            .transition()
            .duration(150)
            .attr('r', CONGESTION_CONFIG[d.congestionLevel].radius)
            .attr('stroke-width', 1.5);

          onLocationHover?.(null);
        })
        .on('click', function(event, d) {
          onLocationClick?.(d);
        });
    }

    // Exit selection
    locations.exit()
      .transition()
      .duration(animationDuration)
      .attr('opacity', 0)
      .remove();

    ChartPerformance.endMeasure('heatmap-render');
  }, [
    projection,
    processedData,
    colorScale,
    colorScheme,
    showTrafficFlow,
    showCameraLabels,
    enableInteraction,
    animationDuration,
    dimensions,
    onLocationClick,
    onLocationHover,
  ]);

  // Render legend
  const renderLegend = useCallback(() => {
    if (!showLegend || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    let legend = svg.select('.legend');

    if (legend.empty()) {
      legend = svg.append('g').attr('class', 'legend');
    }

    legend.selectAll('*').remove();

    const legendX = dimensions.width - 120;
    const legendY = dimensions.margin.top + 20;

    // Add legend background
    legend.append('rect')
      .attr('x', legendX - 10)
      .attr('y', legendY - 15)
      .attr('width', 110)
      .attr('height', 140)
      .attr('fill', 'hsl(var(--background))')
      .attr('stroke', 'hsl(var(--border))')
      .attr('stroke-width', 1)
      .attr('rx', 4)
      .attr('opacity', 0.95);

    // Add legend title
    legend.append('text')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('font-size', '12px')
      .attr('font-weight', '600')
      .attr('fill', 'hsl(var(--foreground))')
      .text(colorScheme === 'speed' ? 'Speed (mph)' :
            colorScheme === 'density' ? 'Vehicle Count' :
            'Traffic Intensity');

    // Add color gradient
    const gradient = legend.append('defs')
      .append('linearGradient')
      .attr('id', 'legend-gradient')
      .attr('x1', '0%')
      .attr('x2', '0%')
      .attr('y1', '0%')
      .attr('y2', '100%');

    const colorStops = 10;
    for (let i = 0; i <= colorStops; i++) {
      const t = i / colorStops;
      const value = colorScale.domain()[0] + t * (colorScale.domain()[1] - colorScale.domain()[0]);
      gradient.append('stop')
        .attr('offset', `${t * 100}%`)
        .attr('stop-color', colorScale(value));
    }

    // Add gradient rectangle
    legend.append('rect')
      .attr('x', legendX)
      .attr('y', legendY + 15)
      .attr('width', 20)
      .attr('height', 80)
      .attr('fill', 'url(#legend-gradient)')
      .attr('stroke', 'hsl(var(--border))')
      .attr('stroke-width', 1);

    // Add scale labels
    const scale = colorScale.domain();
    [scale[1], scale[1] * 0.75, scale[1] * 0.5, scale[1] * 0.25, scale[0]].forEach((value, i) => {
      legend.append('text')
        .attr('x', legendX + 25)
        .attr('y', legendY + 20 + (i * 20))
        .attr('font-size', '10px')
        .attr('fill', 'hsl(var(--muted-foreground))')
        .attr('alignment-baseline', 'middle')
        .text(Math.round(value).toString());
    });

    // Add congestion level legend
    const congestionY = legendY + 110;
    legend.append('text')
      .attr('x', legendX)
      .attr('y', congestionY)
      .attr('font-size', '10px')
      .attr('font-weight', '600')
      .attr('fill', 'hsl(var(--foreground))')
      .text('Congestion:');

    Object.entries(CONGESTION_CONFIG).forEach(([level, config], i) => {
      legend.append('circle')
        .attr('cx', legendX + (i * 16))
        .attr('cy', congestionY + 12)
        .attr('r', 4)
        .attr('fill', config.color)
        .attr('opacity', config.opacity);
    });
  }, [showLegend, colorScale, colorScheme, dimensions]);

  // Sync refs
  useEffect(() => {
    if (d3SvgRef.current) {
      svgRef.current = d3SvgRef.current;
    }
  }, [d3SvgRef]);

  // Render when data changes
  useEffect(() => {
    startTransition(() => {
      renderHeatmap();
      renderLegend();
    });
  }, [renderHeatmap, renderLegend]);

  return (
    <div className={cn('relative', className)}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full h-full"
        style={{ background: 'hsl(var(--background))' }}
      >
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
      </svg>

      {isPending && (
        <div className="absolute top-2 right-2">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
        </div>
      )}
    </div>
  );
};

export default TrafficFlowHeatmap;
