'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Activity, 
  Camera, 
  Gauge, 
  TrendingUp, 
  AlertTriangle,
  Cpu,
  BarChart3,
  Users
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAnalyticsWebSocket } from '@/hooks/use-analytics-websocket'

interface MetricCardProps {
  title: string
  value: string | number
  unit?: string
  change?: number
  icon: React.ReactNode
  status?: 'excellent' | 'good' | 'warning' | 'poor'
  loading?: boolean
}

function MetricCard({ title, value, unit, change, icon, status = 'good', loading }: MetricCardProps) {
  const statusColors = {
    excellent: 'text-green-500 bg-green-50 border-green-200',
    good: 'text-blue-500 bg-blue-50 border-blue-200',
    warning: 'text-amber-500 bg-amber-50 border-amber-200',
    poor: 'text-red-500 bg-red-50 border-red-200'
  }

  const changeColor = change && change > 0 ? 'text-green-600' : 'text-red-600'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={cn('relative overflow-hidden', statusColors[status])}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-600">{title}</CardTitle>
          <div className="text-gray-400">{icon}</div>
        </CardHeader>
        <CardContent>
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-8 bg-gray-200 rounded animate-pulse"
              />
            ) : (
              <motion.div
                key="value"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.2 }}
              >
                <div className="text-2xl font-bold">
                  {value}
                  {unit && <span className="text-sm font-normal ml-1">{unit}</span>}
                </div>
                {change !== undefined && (
                  <p className={cn('text-xs mt-1', changeColor)}>
                    {change > 0 ? '+' : ''}{change}% from last hour
                  </p>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
        
        {/* Animated background decoration */}
        <motion.div
          className={cn('absolute -right-4 -bottom-4 w-24 h-24 rounded-full opacity-10', statusColors[status].split(' ')[0])}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.1, 0.2, 0.1]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
      </Card>
    </motion.div>
  )
}

export function RealTimeMetrics() {
  const { metrics, incidents, vehicleCounts, isConnected } = useAnalyticsWebSocket()

  // Calculate derived metrics
  const totalVehicles = vehicleCounts.reduce((sum, vc) => sum + vc.count, 0)
  const activeIncidents = incidents.filter(i => !i.acknowledged).length
  const criticalIncidents = incidents.filter(i => i.severity === 'critical' && !i.acknowledged).length

  // Determine status based on thresholds
  const getLatencyStatus = (latency: number): MetricCardProps['status'] => {
    if (latency < 50) return 'excellent'
    if (latency < 100) return 'good'
    if (latency < 200) return 'warning'
    return 'poor'
  }

  const getAccuracyStatus = (accuracy: number): MetricCardProps['status'] => {
    if (accuracy >= 95) return 'excellent'
    if (accuracy >= 90) return 'good'
    if (accuracy >= 80) return 'warning'
    return 'poor'
  }

  const getCongestionStatus = (level: string): MetricCardProps['status'] => {
    switch (level) {
      case 'low': return 'excellent'
      case 'moderate': return 'good'
      case 'high': return 'warning'
      case 'severe': return 'poor'
      default: return 'good'
    }
  }

  const getIncidentStatus = (count: number): MetricCardProps['status'] => {
    if (count === 0) return 'excellent'
    if (count < 3) return 'good'
    if (count < 5) return 'warning'
    return 'poor'
  }

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      {!isConnected && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-center gap-2"
        >
          <AlertTriangle className="w-4 h-4 text-amber-600" />
          <span className="text-sm text-amber-800">
            Real-time updates paused. Showing cached data.
          </span>
        </motion.div>
      )}

      {/* Metrics Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Vehicles"
          value={totalVehicles.toLocaleString()}
          unit="vehicles"
          change={12}
          icon={<Users className="w-4 h-4" />}
          status="good"
          loading={!metrics && totalVehicles === 0}
        />

        <MetricCard
          title="Average Speed"
          value={metrics?.average_speed || 0}
          unit="km/h"
          change={-5}
          icon={<Gauge className="w-4 h-4" />}
          status={getCongestionStatus(metrics?.congestion_level || 'low')}
          loading={!metrics}
        />

        <MetricCard
          title="Active Cameras"
          value={`${metrics?.active_cameras || 0}/${metrics?.total_cameras || 52}`}
          change={0}
          icon={<Camera className="w-4 h-4" />}
          status={metrics && metrics.active_cameras >= 48 ? 'excellent' : 'warning'}
          loading={!metrics}
        />

        <MetricCard
          title="Active Incidents"
          value={activeIncidents}
          unit={criticalIncidents > 0 ? `(${criticalIncidents} critical)` : undefined}
          icon={<AlertTriangle className="w-4 h-4" />}
          status={getIncidentStatus(activeIncidents)}
          loading={false}
        />
      </div>

      {/* Performance Metrics */}
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard
          title="Inference Latency"
          value={metrics?.inference_latency || 0}
          unit="ms"
          icon={<Activity className="w-4 h-4" />}
          status={getLatencyStatus(metrics?.inference_latency || 0)}
          loading={!metrics}
        />

        <MetricCard
          title="AI Accuracy"
          value={metrics?.ai_accuracy || 0}
          unit="%"
          icon={<Cpu className="w-4 h-4" />}
          status={getAccuracyStatus(metrics?.ai_accuracy || 0)}
          loading={!metrics}
        />

        <MetricCard
          title="Congestion Level"
          value={metrics?.congestion_level?.toUpperCase() || 'UNKNOWN'}
          icon={<BarChart3 className="w-4 h-4" />}
          status={getCongestionStatus(metrics?.congestion_level || 'low')}
          loading={!metrics}
        />
      </div>

      {/* Last Updated */}
      {metrics && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-xs text-gray-500 text-right"
        >
          Last updated: {new Date(metrics.timestamp).toLocaleTimeString()}
        </motion.div>
      )}
    </div>
  )
}