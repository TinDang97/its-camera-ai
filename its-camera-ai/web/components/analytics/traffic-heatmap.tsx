'use client'

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { MapPin, Clock, Activity, Info } from 'lucide-react'
import { motion } from 'framer-motion'
import { format } from 'date-fns'
import { cn } from '@/lib/utils'

interface HeatmapDataPoint {
  cameraId: string
  cameraName: string
  location: {
    lat: number
    lng: number
    intersection?: string
  }
  timestamp: string
  vehicleCount: number
  averageSpeed: number
  congestionLevel: 'low' | 'moderate' | 'high' | 'severe'
  incidentCount: number
}

interface TrafficHeatmapProps {
  data: HeatmapDataPoint[]
  timeRange: '1h' | '4h' | '24h' | '7d'
  viewMode?: 'density' | 'speed' | 'incidents'
  showLabels?: boolean
  loading?: boolean
  onCameraSelect?: (cameraId: string) => void
  onTimeRangeChange?: (range: '1h' | '4h' | '24h' | '7d') => void
}

export function TrafficHeatmap({
  data,
  timeRange,
  viewMode = 'density',
  showLabels = true,
  loading = false,
  onCameraSelect,
  onTimeRangeChange
}: TrafficHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [hoveredCamera, setHoveredCamera] = useState<string | null>(null)
  const [selectedMetric, setSelectedMetric] = useState<'density' | 'speed' | 'incidents'>(viewMode)

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width } = containerRef.current.getBoundingClientRect()
        setDimensions({
          width: width,
          height: Math.min(600, width * 0.75)
        })
      }
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Draw heatmap with D3
  useEffect(() => {
    if (!svgRef.current || !data.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const margin = { top: 40, right: 40, bottom: 60, left: 60 }
    const width = dimensions.width - margin.left - margin.right
    const height = dimensions.height - margin.top - margin.bottom

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Group data by hour and camera for grid layout
    const timeGroups = d3.group(data, d => {
      const date = new Date(d.timestamp)
      return format(date, timeRange === '1h' ? 'HH:mm' : 'HH:00')
    })

    const cameras = Array.from(new Set(data.map(d => d.cameraId))).sort()
    const times = Array.from(timeGroups.keys()).sort()

    // Create scales
    const xScale = d3.scaleBand()
      .domain(times)
      .range([0, width])
      .padding(0.05)

    const yScale = d3.scaleBand()
      .domain(cameras)
      .range([0, height])
      .padding(0.05)

    // Color scales based on metric
    const getColorScale = () => {
      switch (selectedMetric) {
        case 'speed':
          return d3.scaleSequential()
            .domain([0, 80])
            .interpolator(d3.interpolateRdYlGn)
        case 'incidents':
          return d3.scaleSequential()
            .domain([0, 5])
            .interpolator(d3.interpolateOrRd)
        default: // density
          return d3.scaleSequential()
            .domain([0, 50])
            .interpolator(d3.interpolateViridis)
      }
    }

    const colorScale = getColorScale()

    // Create tooltip
    const tooltip = d3.select('body')
      .append('div')
      .attr('class', 'heatmap-tooltip')
      .style('position', 'absolute')
      .style('padding', '10px')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('border-radius', '4px')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .style('font-size', '12px')
      .style('z-index', 1000)

    // Draw cells
    const cells = g.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => xScale(format(new Date(d.timestamp), timeRange === '1h' ? 'HH:mm' : 'HH:00')) || 0)
      .attr('y', d => yScale(d.cameraId) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('rx', 2)
      .style('fill', d => {
        const value = selectedMetric === 'speed' 
          ? d.averageSpeed 
          : selectedMetric === 'incidents' 
          ? d.incidentCount 
          : d.vehicleCount
        return colorScale(value)
      })
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(100)
          .style('stroke', '#fff')
          .style('stroke-width', 2)

        setHoveredCamera(d.cameraId)

        tooltip
          .transition()
          .duration(200)
          .style('opacity', 1)

        const metricValue = selectedMetric === 'speed' 
          ? `${d.averageSpeed.toFixed(1)} km/h`
          : selectedMetric === 'incidents' 
          ? `${d.incidentCount} incidents`
          : `${d.vehicleCount} vehicles`

        tooltip.html(`
          <div><strong>${d.cameraName}</strong></div>
          <div>${d.location.intersection || 'Unknown location'}</div>
          <div>${format(new Date(d.timestamp), 'MMM dd, HH:mm')}</div>
          <div>${metricValue}</div>
          <div>Congestion: ${d.congestionLevel}</div>
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px')
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(100)
          .style('stroke', 'none')

        setHoveredCamera(null)

        tooltip
          .transition()
          .duration(500)
          .style('opacity', 0)
      })
      .on('click', (event, d) => {
        onCameraSelect?.(d.cameraId)
      })

    // Animate cells on initial render
    cells
      .style('opacity', 0)
      .transition()
      .duration(500)
      .delay((d, i) => i * 2)
      .style('opacity', 1)

    // Add x-axis
    const xAxis = d3.axisBottom(xScale)
      .tickValues(xScale.domain().filter((d, i) => {
        const interval = timeRange === '1h' ? 4 : timeRange === '4h' ? 2 : 4
        return i % interval === 0
      }))

    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)')

    // Add y-axis with camera names
    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d => {
        const camera = data.find(item => item.cameraId === d)
        return camera ? camera.cameraName.slice(0, 10) : d.slice(0, 8)
      })

    g.append('g')
      .call(yAxis)

    // Add axis labels
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', '#666')
      .text('Cameras')

    g.append('text')
      .attr('transform', `translate(${width / 2}, ${height + margin.bottom})`)
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', '#666')
      .text('Time')

    // Add color legend
    const legendWidth = 200
    const legendHeight = 10

    const legendScale = d3.scaleLinear()
      .domain(colorScale.domain())
      .range([0, legendWidth])

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => {
        if (selectedMetric === 'speed') return `${d} km/h`
        if (selectedMetric === 'incidents') return `${d}`
        return `${d}`
      })

    const legend = g.append('g')
      .attr('transform', `translate(${width - legendWidth - 20}, -30)`)

    // Create gradient for legend
    const gradientId = `gradient-${selectedMetric}`
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '100%')
      .attr('y1', '0%')
      .attr('y2', '0%')

    const colorRange = d3.range(0, 1.01, 0.01)
    colorRange.forEach(t => {
      gradient.append('stop')
        .attr('offset', `${t * 100}%`)
        .attr('stop-color', colorScale(t * colorScale.domain()[1]))
    })

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', `url(#${gradientId})`)

    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis)
      .selectAll('text')
      .style('font-size', '10px')

    // Cleanup on unmount
    return () => {
      tooltip.remove()
    }
  }, [data, dimensions, selectedMetric, showLabels, timeRange])

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <div className="h-6 bg-gray-200 rounded w-1/3 animate-pulse" />
        </CardHeader>
        <CardContent>
          <div className="h-96 bg-gray-200 rounded animate-pulse" />
        </CardContent>
      </Card>
    )
  }

  // Calculate statistics
  const stats = {
    totalVehicles: data.reduce((sum, d) => sum + d.vehicleCount, 0),
    avgSpeed: data.reduce((sum, d) => sum + d.averageSpeed, 0) / data.length,
    totalIncidents: data.reduce((sum, d) => sum + d.incidentCount, 0),
    severeCount: data.filter(d => d.congestionLevel === 'severe').length
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <CardTitle className="flex items-center gap-2">
              <MapPin className="w-5 h-5" />
              Traffic Density Heatmap
            </CardTitle>
            
            {/* Time Range Selector */}
            <div className="flex gap-2">
              {(['1h', '4h', '24h', '7d'] as const).map(range => (
                <Button
                  key={range}
                  variant={timeRange === range ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onTimeRangeChange?.(range)}
                >
                  {range}
                </Button>
              ))}
            </div>
          </div>

          {/* Metric Selector */}
          <div className="flex gap-2 mt-4">
            <Button
              variant={selectedMetric === 'density' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMetric('density')}
              className="flex items-center gap-1"
            >
              <Activity className="w-4 h-4" />
              Density
            </Button>
            <Button
              variant={selectedMetric === 'speed' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMetric('speed')}
            >
              Speed
            </Button>
            <Button
              variant={selectedMetric === 'incidents' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMetric('incidents')}
            >
              Incidents
            </Button>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
            <div className="bg-blue-50 rounded-lg p-2">
              <p className="text-xs text-blue-600">Total Vehicles</p>
              <p className="text-lg font-semibold text-blue-900">{stats.totalVehicles.toLocaleString()}</p>
            </div>
            <div className="bg-green-50 rounded-lg p-2">
              <p className="text-xs text-green-600">Avg Speed</p>
              <p className="text-lg font-semibold text-green-900">{stats.avgSpeed.toFixed(1)} km/h</p>
            </div>
            <div className="bg-yellow-50 rounded-lg p-2">
              <p className="text-xs text-yellow-600">Incidents</p>
              <p className="text-lg font-semibold text-yellow-900">{stats.totalIncidents}</p>
            </div>
            <div className="bg-red-50 rounded-lg p-2">
              <p className="text-xs text-red-600">Severe Congestion</p>
              <p className="text-lg font-semibold text-red-900">{stats.severeCount}</p>
            </div>
          </div>

          {/* Info text */}
          {hoveredCamera && (
            <div className="mt-2 p-2 bg-gray-50 rounded flex items-center gap-2">
              <Info className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600">
                Click on a cell to view detailed camera analytics
              </span>
            </div>
          )}
        </CardHeader>
        
        <CardContent>
          <div ref={containerRef} className="w-full">
            <svg
              ref={svgRef}
              width={dimensions.width}
              height={dimensions.height}
              className="w-full h-auto"
            />
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}