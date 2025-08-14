'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { useWebSocket } from '@/components/providers/websocket-provider'

export interface AnalyticsMetrics {
  timestamp: string
  vehicle_count: number
  average_speed: number
  congestion_level: 'low' | 'moderate' | 'high' | 'severe'
  active_cameras: number
  total_cameras: number
  inference_latency: number
  ai_accuracy: number
}

export interface IncidentAlert {
  id: string
  timestamp: string
  severity: 'info' | 'warning' | 'critical' | 'emergency'
  location: string
  camera_id: string
  description: string
  confidence: number
  acknowledged: boolean
}

export interface VehicleCount {
  timestamp: string
  camera_id: string
  count: number
  vehicle_types: {
    car: number
    truck: number
    bus: number
    motorcycle: number
  }
}

export interface SpeedUpdate {
  timestamp: string
  location: string
  average_speed: number
  max_speed: number
  min_speed: number
  speed_limit: number
}

export interface PredictionData {
  timestamp: string
  horizon: '15min' | '1hour' | '4hours' | '24hours'
  predicted_volume: number
  confidence_interval: {
    lower: number
    upper: number
  }
  confidence_score: number
}

export type AnalyticsMessage = 
  | { type: 'metrics'; data: AnalyticsMetrics }
  | { type: 'incident'; data: IncidentAlert }
  | { type: 'vehicle_count'; data: VehicleCount }
  | { type: 'speed_update'; data: SpeedUpdate }
  | { type: 'prediction'; data: PredictionData }

interface UseAnalyticsWebSocketOptions {
  enabled?: boolean
  onMetrics?: (metrics: AnalyticsMetrics) => void
  onIncident?: (incident: IncidentAlert) => void
  onVehicleCount?: (count: VehicleCount) => void
  onSpeedUpdate?: (speed: SpeedUpdate) => void
  onPrediction?: (prediction: PredictionData) => void
}

export function useAnalyticsWebSocket(options: UseAnalyticsWebSocketOptions = {}) {
  const { 
    enabled = true,
    onMetrics,
    onIncident,
    onVehicleCount,
    onSpeedUpdate,
    onPrediction
  } = options

  const { sendMessage, lastMessage, connectionState } = useWebSocket()
  const [metrics, setMetrics] = useState<AnalyticsMetrics | null>(null)
  const [incidents, setIncidents] = useState<IncidentAlert[]>([])
  const [vehicleCounts, setVehicleCounts] = useState<Map<string, VehicleCount>>(new Map())
  const [speedUpdates, setSpeedUpdates] = useState<SpeedUpdate[]>([])
  const [predictions, setPredictions] = useState<PredictionData[]>([])
  const [isSubscribed, setIsSubscribed] = useState(false)

  // Subscribe to analytics channel
  const subscribe = useCallback(() => {
    if (connectionState === 'connected' && !isSubscribed && enabled) {
      sendMessage({
        type: 'subscribe',
        channel: 'analytics',
        filters: {
          metrics: true,
          incidents: true,
          vehicle_counts: true,
          speed_updates: true,
          predictions: true
        }
      })
      setIsSubscribed(true)
    }
  }, [connectionState, isSubscribed, enabled, sendMessage])

  // Unsubscribe from analytics channel
  const unsubscribe = useCallback(() => {
    if (connectionState === 'connected' && isSubscribed) {
      sendMessage({
        type: 'unsubscribe',
        channel: 'analytics'
      })
      setIsSubscribed(false)
    }
  }, [connectionState, isSubscribed, sendMessage])

  // Process incoming messages
  useEffect(() => {
    if (!lastMessage) return

    try {
      const message = JSON.parse(lastMessage) as AnalyticsMessage

      switch (message.type) {
        case 'metrics':
          setMetrics(message.data)
          onMetrics?.(message.data)
          break
        case 'incident':
          setIncidents(prev => {
            const updated = [message.data, ...prev].slice(0, 100) // Keep last 100
            return updated.sort((a, b) => 
              new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
            )
          })
          onIncident?.(message.data)
          break
        case 'vehicle_count':
          setVehicleCounts(prev => {
            const updated = new Map(prev)
            updated.set(message.data.camera_id, message.data)
            return updated
          })
          onVehicleCount?.(message.data)
          break
        case 'speed_update':
          setSpeedUpdates(prev => {
            const updated = [message.data, ...prev].slice(0, 50) // Keep last 50
            return updated.sort((a, b) => 
              new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
            )
          })
          onSpeedUpdate?.(message.data)
          break
        case 'prediction':
          setPredictions(prev => {
            const updated = [message.data, ...prev].slice(0, 20) // Keep last 20
            return updated.sort((a, b) => 
              new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
            )
          })
          onPrediction?.(message.data)
          break
      }
    } catch (error) {
      console.error('Error parsing analytics message:', error)
    }
  }, [lastMessage, onMetrics, onIncident, onVehicleCount, onSpeedUpdate, onPrediction])

  // Manage subscription lifecycle
  useEffect(() => {
    if (enabled && connectionState === 'connected' && !isSubscribed) {
      subscribe()
    }

    return () => {
      if (isSubscribed) {
        unsubscribe()
      }
    }
  }, [enabled, connectionState, isSubscribed, subscribe, unsubscribe])

  // Acknowledge incident
  const acknowledgeIncident = useCallback((incidentId: string) => {
    if (connectionState === 'connected') {
      sendMessage({
        type: 'acknowledge_incident',
        incident_id: incidentId
      })
      
      // Optimistically update local state
      setIncidents(prev => prev.map(incident => 
        incident.id === incidentId 
          ? { ...incident, acknowledged: true }
          : incident
      ))
    }
  }, [connectionState, sendMessage])

  // Request specific data
  const requestHistoricalData = useCallback((
    startTime: string,
    endTime: string,
    dataType: 'metrics' | 'incidents' | 'predictions'
  ) => {
    if (connectionState === 'connected') {
      sendMessage({
        type: 'request_historical',
        data_type: dataType,
        start_time: startTime,
        end_time: endTime
      })
    }
  }, [connectionState, sendMessage])

  return {
    // Connection state
    isConnected: connectionState === 'connected',
    connectionState,
    isSubscribed,
    
    // Real-time data
    metrics,
    incidents,
    vehicleCounts: Array.from(vehicleCounts.values()),
    speedUpdates,
    predictions,
    
    // Actions
    acknowledgeIncident,
    requestHistoricalData,
    subscribe,
    unsubscribe,
    
    // Computed values
    activeIncidents: incidents.filter(i => !i.acknowledged),
    criticalIncidents: incidents.filter(i => i.severity === 'critical' && !i.acknowledged),
    latestMetrics: metrics,
    averageSpeed: speedUpdates.length > 0 
      ? speedUpdates.reduce((sum, s) => sum + s.average_speed, 0) / speedUpdates.length
      : 0,
    totalVehicleCount: Array.from(vehicleCounts.values())
      .reduce((sum, vc) => sum + vc.count, 0)
  }
}

// Mock data generator for development
export function generateMockAnalyticsData(): AnalyticsMessage {
  const types: AnalyticsMessage['type'][] = ['metrics', 'incident', 'vehicle_count', 'speed_update', 'prediction']
  const type = types[Math.floor(Math.random() * types.length)]
  
  const timestamp = new Date().toISOString()
  
  switch (type) {
    case 'metrics':
      return {
        type: 'metrics',
        data: {
          timestamp,
          vehicle_count: Math.floor(Math.random() * 500) + 500,
          average_speed: Math.floor(Math.random() * 30) + 30,
          congestion_level: ['low', 'moderate', 'high', 'severe'][Math.floor(Math.random() * 4)] as any,
          active_cameras: Math.floor(Math.random() * 10) + 40,
          total_cameras: 52,
          inference_latency: Math.floor(Math.random() * 50) + 20,
          ai_accuracy: Math.floor(Math.random() * 10) + 90
        }
      }
    case 'incident':
      return {
        type: 'incident',
        data: {
          id: `INC-${Date.now()}`,
          timestamp,
          severity: ['info', 'warning', 'critical', 'emergency'][Math.floor(Math.random() * 4)] as any,
          location: `Main St & ${Math.floor(Math.random() * 10) + 1}th Ave`,
          camera_id: `CAM-${String(Math.floor(Math.random() * 52) + 1).padStart(3, '0')}`,
          description: 'Traffic congestion detected',
          confidence: Math.floor(Math.random() * 20) + 80,
          acknowledged: false
        }
      }
    case 'vehicle_count':
      return {
        type: 'vehicle_count',
        data: {
          timestamp,
          camera_id: `CAM-${String(Math.floor(Math.random() * 52) + 1).padStart(3, '0')}`,
          count: Math.floor(Math.random() * 100) + 20,
          vehicle_types: {
            car: Math.floor(Math.random() * 60) + 10,
            truck: Math.floor(Math.random() * 20) + 5,
            bus: Math.floor(Math.random() * 10) + 2,
            motorcycle: Math.floor(Math.random() * 10) + 3
          }
        }
      }
    case 'speed_update':
      return {
        type: 'speed_update',
        data: {
          timestamp,
          location: `Zone ${Math.floor(Math.random() * 10) + 1}`,
          average_speed: Math.floor(Math.random() * 30) + 30,
          max_speed: Math.floor(Math.random() * 40) + 60,
          min_speed: Math.floor(Math.random() * 20) + 10,
          speed_limit: 60
        }
      }
    case 'prediction':
      const predicted = Math.floor(Math.random() * 500) + 1000
      return {
        type: 'prediction',
        data: {
          timestamp,
          horizon: ['15min', '1hour', '4hours', '24hours'][Math.floor(Math.random() * 4)] as any,
          predicted_volume: predicted,
          confidence_interval: {
            lower: predicted - Math.floor(Math.random() * 100),
            upper: predicted + Math.floor(Math.random() * 100)
          },
          confidence_score: Math.floor(Math.random() * 15) + 85
        }
      }
  }
}