'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { 
  AreaChart, 
  Area, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  ComposedChart,
  ReferenceLine
} from 'recharts'
import { motion } from 'framer-motion'
import { Clock, TrendingUp, TrendingDown, Activity } from 'lucide-react'
import { useState, useEffect, useMemo } from 'react'
import { format } from 'date-fns'
import { cn } from '@/lib/utils'

interface TrafficDataPoint {
  timestamp: string
  vehicleCount: number
  averageSpeed: number
  congestionLevel: 'low' | 'moderate' | 'high' | 'severe'
  occupancy: number
}

interface TrafficFlowChartProps {
  data: TrafficDataPoint[]
  timeRange: '1h' | '4h' | '24h' | '7d'
  chartType?: 'area' | 'line' | 'bar' | 'composed'
  showPredictions?: boolean
  predictions?: TrafficDataPoint[]
  loading?: boolean
  onTimeRangeChange?: (range: '1h' | '4h' | '24h' | '7d') => void
}

export function TrafficFlowChart({
  data,
  timeRange,
  chartType = 'area',
  showPredictions = false,
  predictions = [],
  loading = false,
  onTimeRangeChange
}: TrafficFlowChartProps) {
  const [selectedMetric, setSelectedMetric] = useState<'vehicleCount' | 'averageSpeed' | 'occupancy'>('vehicleCount')
  const [animationComplete, setAnimationComplete] = useState(false)

  useEffect(() => {
    setAnimationComplete(false)
    const timer = setTimeout(() => setAnimationComplete(true), 300)
    return () => clearTimeout(timer)
  }, [data, timeRange])

  // Combine actual and prediction data
  const combinedData = useMemo(() => {
    const actualData = data.map(d => ({
      ...d,
      actual: d[selectedMetric],
      predicted: null
    }))

    if (showPredictions && predictions.length > 0) {
      const predictionData = predictions.map(p => ({
        ...p,
        actual: null,
        predicted: p[selectedMetric]
      }))
      return [...actualData, ...predictionData]
    }

    return actualData
  }, [data, predictions, showPredictions, selectedMetric])

  // Format time based on range
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    switch (timeRange) {
      case '1h':
        return format(date, 'HH:mm')
      case '4h':
        return format(date, 'HH:mm')
      case '24h':
        return format(date, 'HH:00')
      case '7d':
        return format(date, 'EEE')
      default:
        return format(date, 'HH:mm')
    }
  }

  // Calculate statistics
  const stats = useMemo(() => {
    if (data.length === 0) return null

    const values = data.map(d => d[selectedMetric])
    const avg = values.reduce((sum, val) => sum + val, 0) / values.length
    const max = Math.max(...values)
    const min = Math.min(...values)
    const latest = values[values.length - 1]
    const previous = values[values.length - 2] || latest
    const trend = ((latest - previous) / previous) * 100

    return { avg, max, min, latest, trend }
  }, [data, selectedMetric])

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border">
          <p className="text-sm font-medium">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toFixed(1)}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  // Render chart based on type
  const renderChart = () => {
    const commonProps = {
      data: combinedData,
      margin: { top: 10, right: 30, left: 0, bottom: 0 }
    }

    const axisProps = {
      xAxis: {
        dataKey: 'timestamp',
        tickFormatter: formatTime,
        stroke: '#94a3b8'
      },
      yAxis: {
        stroke: '#94a3b8'
      }
    }

    switch (chartType) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis {...axisProps.xAxis} />
            <YAxis {...axisProps.yAxis} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Actual"
              isAnimationActive={!animationComplete}
            />
            {showPredictions && (
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#8b5cf6"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Predicted"
                isAnimationActive={!animationComplete}
              />
            )}
          </LineChart>
        )

      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis {...axisProps.xAxis} />
            <YAxis {...axisProps.yAxis} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar
              dataKey="actual"
              fill="#3b82f6"
              name="Actual"
              isAnimationActive={!animationComplete}
            />
            {showPredictions && (
              <Bar
                dataKey="predicted"
                fill="#8b5cf6"
                opacity={0.6}
                name="Predicted"
                isAnimationActive={!animationComplete}
              />
            )}
          </BarChart>
        )

      case 'composed':
        return (
          <ComposedChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis {...axisProps.xAxis} />
            <YAxis yAxisId="left" orientation="left" {...axisProps.yAxis} />
            <YAxis yAxisId="right" orientation="right" {...axisProps.yAxis} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar
              yAxisId="left"
              dataKey="vehicleCount"
              fill="#3b82f6"
              name="Vehicles"
              isAnimationActive={!animationComplete}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="averageSpeed"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              name="Speed (km/h)"
              isAnimationActive={!animationComplete}
            />
            {stats && <ReferenceLine yAxisId="left" y={stats.avg} stroke="#ef4444" strokeDasharray="3 3" />}
          </ComposedChart>
        )

      default: // area
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis {...axisProps.xAxis} />
            <YAxis {...axisProps.yAxis} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Area
              type="monotone"
              dataKey="actual"
              stroke="#3b82f6"
              fill="#3b82f6"
              fillOpacity={0.3}
              strokeWidth={2}
              name="Actual"
              isAnimationActive={!animationComplete}
            />
            {showPredictions && (
              <Area
                type="monotone"
                dataKey="predicted"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.2}
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Predicted"
                isAnimationActive={!animationComplete}
              />
            )}
          </AreaChart>
        )
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <div className="h-6 bg-gray-200 rounded w-1/3 animate-pulse" />
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-gray-200 rounded animate-pulse" />
        </CardContent>
      </Card>
    )
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
              <Activity className="w-5 h-5" />
              Traffic Flow Analysis
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
              variant={selectedMetric === 'vehicleCount' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMetric('vehicleCount')}
            >
              Vehicles
            </Button>
            <Button
              variant={selectedMetric === 'averageSpeed' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMetric('averageSpeed')}
            >
              Speed
            </Button>
            <Button
              variant={selectedMetric === 'occupancy' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedMetric('occupancy')}
            >
              Occupancy
            </Button>
          </div>

          {/* Statistics */}
          {stats && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
              <div className="bg-gray-50 rounded-lg p-2">
                <p className="text-xs text-gray-500">Current</p>
                <p className="text-lg font-semibold">{stats.latest.toFixed(1)}</p>
                <p className={cn('text-xs flex items-center gap-1', stats.trend > 0 ? 'text-green-600' : 'text-red-600')}>
                  {stats.trend > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  {Math.abs(stats.trend).toFixed(1)}%
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-2">
                <p className="text-xs text-gray-500">Average</p>
                <p className="text-lg font-semibold">{stats.avg.toFixed(1)}</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-2">
                <p className="text-xs text-gray-500">Maximum</p>
                <p className="text-lg font-semibold">{stats.max.toFixed(1)}</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-2">
                <p className="text-xs text-gray-500">Minimum</p>
                <p className="text-lg font-semibold">{stats.min.toFixed(1)}</p>
              </div>
            </div>
          )}
        </CardHeader>
        
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            {renderChart()}
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </motion.div>
  )
}