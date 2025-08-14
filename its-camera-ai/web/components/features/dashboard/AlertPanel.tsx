'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { AlertCircle, AlertTriangle, Info, XCircle, Clock, MapPin } from 'lucide-react'

export interface Alert {
  id: string
  type: 'incident' | 'congestion' | 'detection' | 'system'
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info'
  title: string
  description: string
  location?: string
  timestamp: Date
  acknowledged?: boolean
  cameraId?: string
}

interface AlertPanelProps {
  alerts?: Alert[]
  maxHeight?: string
  onAlertClick?: (alert: Alert) => void
}

// Mock data generator
const generateMockAlerts = (): Alert[] => [
  {
    id: '1',
    type: 'incident',
    severity: 'critical',
    title: 'Vehicle Collision Detected',
    description: 'Multiple vehicles involved in collision at intersection',
    location: 'Main St & 5th Ave',
    timestamp: new Date(Date.now() - 2 * 60000),
    cameraId: 'CAM-001'
  },
  {
    id: '2',
    type: 'congestion',
    severity: 'high',
    title: 'Heavy Traffic Congestion',
    description: 'Traffic speed below 20 km/h for more than 5 minutes',
    location: 'Highway I-95 North',
    timestamp: new Date(Date.now() - 10 * 60000),
    cameraId: 'CAM-015'
  },
  {
    id: '3',
    type: 'detection',
    severity: 'medium',
    title: 'Unauthorized Vehicle in Bus Lane',
    description: 'Non-bus vehicle detected in dedicated bus lane',
    location: 'Downtown Transit Corridor',
    timestamp: new Date(Date.now() - 15 * 60000),
    acknowledged: true,
    cameraId: 'CAM-008'
  },
  {
    id: '4',
    type: 'system',
    severity: 'low',
    title: 'Camera Feed Quality Degraded',
    description: 'Video quality below threshold on Camera CAM-022',
    timestamp: new Date(Date.now() - 30 * 60000),
    cameraId: 'CAM-022'
  },
  {
    id: '5',
    type: 'detection',
    severity: 'info',
    title: 'Pedestrian Crossing Activity',
    description: 'High pedestrian traffic detected at crosswalk',
    location: 'School Zone - Oak Elementary',
    timestamp: new Date(Date.now() - 45 * 60000),
    cameraId: 'CAM-012'
  }
]

export function AlertPanel({ 
  alerts = generateMockAlerts(), 
  maxHeight = '400px',
  onAlertClick 
}: AlertPanelProps) {
  const [localAlerts, setLocalAlerts] = useState(alerts)
  
  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Randomly add a new alert every 30 seconds
      if (Math.random() > 0.7) {
        const newAlert: Alert = {
          id: Date.now().toString(),
          type: ['incident', 'congestion', 'detection', 'system'][Math.floor(Math.random() * 4)] as Alert['type'],
          severity: ['critical', 'high', 'medium', 'low', 'info'][Math.floor(Math.random() * 5)] as Alert['severity'],
          title: 'New Traffic Event Detected',
          description: 'Automated detection system triggered',
          location: 'Zone ' + Math.floor(Math.random() * 10),
          timestamp: new Date(),
          cameraId: 'CAM-' + Math.floor(Math.random() * 100).toString().padStart(3, '0')
        }
        setLocalAlerts(prev => [newAlert, ...prev].slice(0, 20)) // Keep max 20 alerts
      }
    }, 30000)
    
    return () => clearInterval(interval)
  }, [])

  const getSeverityIcon = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical': return <XCircle className="h-4 w-4" />
      case 'high': return <AlertCircle className="h-4 w-4" />
      case 'medium': return <AlertTriangle className="h-4 w-4" />
      default: return <Info className="h-4 w-4" />
    }
  }

  const getSeverityColor = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical': return 'destructive'
      case 'high': return 'destructive'
      case 'medium': return 'secondary'
      case 'low': return 'outline'
      default: return 'default'
    }
  }

  const getTypeColor = (type: Alert['type']) => {
    switch (type) {
      case 'incident': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      case 'congestion': return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
      case 'detection': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
      case 'system': return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const formatTime = (date: Date) => {
    const mins = Math.floor((Date.now() - date.getTime()) / 60000)
    if (mins < 1) return 'Just now'
    if (mins < 60) return `${mins}m ago`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h ago`
    return `${Math.floor(hours / 24)}d ago`
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Active Alerts</span>
          <Badge variant="outline">{localAlerts.filter(a => !a.acknowledged).length} New</Badge>
        </CardTitle>
        <CardDescription>Real-time traffic incidents and system notifications</CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className={`pr-4`} style={{ height: maxHeight }}>
          <div className="space-y-3">
            {localAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-3 rounded-lg border cursor-pointer transition-colors hover:bg-accent ${
                  alert.acknowledged ? 'opacity-60' : ''
                }`}
                onClick={() => onAlertClick?.(alert)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {getSeverityIcon(alert.severity)}
                    <span className="font-semibold">{alert.title}</span>
                  </div>
                  <Badge variant={getSeverityColor(alert.severity)}>
                    {alert.severity}
                  </Badge>
                </div>
                
                <p className="text-sm text-muted-foreground mb-2">
                  {alert.description}
                </p>
                
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(alert.type)}`}>
                      {alert.type}
                    </span>
                    {alert.location && (
                      <div className="flex items-center space-x-1">
                        <MapPin className="h-3 w-3" />
                        <span>{alert.location}</span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>{formatTime(alert.timestamp)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}