'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { IconArrowUp, IconArrowDown, IconCar, IconActivity, IconAlertTriangle } from '@tabler/icons-react'

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
    icon: <IconCar className="h-4 w-4" />
  },
  {
    label: 'Avg Speed',
    value: '42 km/h',
    change: -8.3,
    trend: 'down',
    icon: <IconActivity className="h-4 w-4" />
  },
  {
    label: 'Congestion Level',
    value: 'Moderate',
    severity: 'medium',
    icon: <IconAlertTriangle className="h-4 w-4" />
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
      case 'medium': return 'default'
      case 'low': return 'secondary'
      default: return 'outline'
    }
  }

  const getTrendIcon = (trend?: string) => {
    if (trend === 'up') return <IconArrowUp className="h-3 w-3" />
    if (trend === 'down') return <IconArrowDown className="h-3 w-3" />
    return null
  }

  const getTrendColor = (trend?: string, change?: number) => {
    if (!trend || !change) return 'text-muted-foreground'
    if (trend === 'up') return change > 0 ? 'text-online' : 'text-offline'
    if (trend === 'down') return change < 0 ? 'text-online' : 'text-offline'
    return 'text-muted-foreground'
  }

  return (
    <Card className="border-border/50 shadow-soft hover:shadow-medium transition-all duration-300">
      <CardHeader className="pb-4 border-b border-border/50">
        <CardTitle className="text-xl font-semibold text-foreground flex items-center gap-3">
          <div className="p-2 rounded-lg bg-secondary/10">
            <IconActivity className="h-5 w-5 text-secondary" />
          </div>
          {title}
        </CardTitle>
        <CardDescription className="text-muted-foreground">{description}</CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="metrics-grid">
          {metrics.map((metric, index) => (
            <div key={index} className="space-y-3 p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-colors duration-200">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className="p-1.5 rounded-lg bg-card">
                  {metric.icon}
                </div>
                <span className="font-medium">{metric.label}</span>
              </div>
              <div className="flex items-end justify-between">
                <div className="space-y-1">
                  <div className="metric-value text-2xl font-bold text-foreground">
                    {metric.value}
                  </div>
                  {metric.severity && (
                    <Badge variant={getSeverityColor(metric.severity)} className="text-2xs font-medium">
                      {metric.severity}
                    </Badge>
                  )}
                </div>
                {metric.change !== undefined && (
                  <div className={`flex items-center gap-1 text-sm font-medium ${getTrendColor(metric.trend, metric.change)}`}>
                    {getTrendIcon(metric.trend)}
                    <span>{Math.abs(metric.change)}%</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
