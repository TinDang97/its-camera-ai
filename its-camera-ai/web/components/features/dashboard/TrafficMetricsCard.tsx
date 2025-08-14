'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ArrowUp, ArrowDown, Car, Activity, AlertTriangle } from 'lucide-react'

interface TrafficMetric {
  label: string
  value: string | number
  change?: number
  trend?: 'up' | 'down' | 'stable'
  severity?: 'low' | 'medium' | 'high'
  icon?: React.ReactNode
}

interface TrafficMetricsCardProps {
  metrics?: TrafficMetric[]
  title?: string
  description?: string
}

// Mock data for demonstration
const defaultMetrics: TrafficMetric[] = [
  {
    label: 'Vehicle Count',
    value: '1,284',
    change: 12.5,
    trend: 'up',
    icon: <Car className="h-4 w-4" />
  },
  {
    label: 'Avg Speed',
    value: '42 km/h',
    change: -8.3,
    trend: 'down',
    icon: <Activity className="h-4 w-4" />
  },
  {
    label: 'Congestion Level',
    value: 'Moderate',
    severity: 'medium',
    icon: <AlertTriangle className="h-4 w-4" />
  },
  {
    label: 'Active Incidents',
    value: 3,
    severity: 'high',
    trend: 'up'
  }
]

export function TrafficMetricsCard({
  metrics = defaultMetrics,
  title = 'Traffic Overview',
  description = 'Real-time traffic metrics from all monitored zones'
}: TrafficMetricsCardProps) {
  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'high': return 'destructive'
      case 'medium': return 'secondary'
      case 'low': return 'outline'
      default: return 'default'
    }
  }

  const getTrendIcon = (trend?: string) => {
    if (trend === 'up') return <ArrowUp className="h-3 w-3" />
    if (trend === 'down') return <ArrowDown className="h-3 w-3" />
    return null
  }

  const getTrendColor = (trend?: string, change?: number) => {
    if (!trend || !change) return 'text-muted-foreground'
    if (trend === 'up') return change > 0 ? 'text-green-600' : 'text-red-600'
    if (trend === 'down') return change < 0 ? 'text-green-600' : 'text-red-600'
    return 'text-muted-foreground'
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {metrics.map((metric, index) => (
            <div key={index} className="space-y-2">
              <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                {metric.icon}
                <span>{metric.label}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl font-bold">{metric.value}</span>
                {metric.severity && (
                  <Badge variant={getSeverityColor(metric.severity)}>
                    {metric.severity}
                  </Badge>
                )}
              </div>
              {metric.change !== undefined && (
                <div className={`flex items-center space-x-1 text-sm ${getTrendColor(metric.trend, metric.change)}`}>
                  {getTrendIcon(metric.trend)}
                  <span>{Math.abs(metric.change)}%</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}