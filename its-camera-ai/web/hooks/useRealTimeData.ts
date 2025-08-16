'use client'

import { useEffect, useCallback, useRef } from 'react'
import { realtimeUtils, sseClient, wsClient, APIError, ENDPOINTS } from '@/lib/api'
import { useAuth } from './useAuth'
import { useIsMounted } from './useIsMounted'
import { useConnectionState, useRealtimeEvents, useRealtimeActions } from '@/stores/useRealtimeStore'

export interface RealtimeEvent {
  id: string
  type: string
  data: any
  timestamp: string
  source: 'sse' | 'websocket'
}

interface UseRealTimeDataOptions {
  enabled?: boolean
  reconnectAttempts?: number
  reconnectDelay?: number
  eventTypes?: string[]
  onConnect?: () => void
  onDisconnect?: (error?: Error) => void
  onEvent?: (event: RealtimeEvent) => void
  onError?: (error: Error) => void
}

interface RealtimeConnectionState {
  isConnected: boolean
  connectionType: 'sse' | 'websocket' | 'none'
  reconnectAttempts: number
  lastError?: Error
  events: RealtimeEvent[]
  eventCount: number
}

export function useRealTimeData(
  endpoint: string,
  options: UseRealTimeDataOptions = {}
) {
  const {
    enabled = true,
    reconnectAttempts = 5,
    reconnectDelay = 1000,
    eventTypes = [],
    onConnect,
    onDisconnect,
    onEvent,
    onError,
  } = options

  const { isAuthenticated } = useAuth()
  const isMounted = useIsMounted()
  const connectionState = useConnectionState(endpoint)
  const { events, eventCount } = useRealtimeEvents(endpoint)
  const {
    setConnectionState,
    removeConnection,
    addEvent,
    clearEvents,
    getEventsByType,
    getLatestEvent,
  } = useRealtimeActions()

  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const cleanupFunctionsRef = useRef<(() => void)[]>([])

  // SSE connection management
  const connectSSE = useCallback(async () => {
    if (!isAuthenticated || !isMounted()) return

    try {
      if (isMounted()) {
        setConnectionState(endpoint, {
          connectionType: 'sse',
          lastError: undefined,
        })
      }

      await realtimeUtils.connectSSE(endpoint, {
        event_types: eventTypes.join(','),
      })

      if (isMounted()) {
        setConnectionState(endpoint, {
          isConnected: true,
          reconnectAttempts: 0,
        })
        onConnect?.()
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error('SSE connection failed')
      if (isMounted()) {
        setConnectionState(endpoint, {
          isConnected: false,
          lastError: err,
          reconnectAttempts: connectionState.reconnectAttempts + 1,
        })
        onError?.(err)
        scheduleReconnect()
      }
    }
  }, [isAuthenticated, isMounted, endpoint, eventTypes, onConnect, onError, setConnectionState, connectionState.reconnectAttempts])

  // WebSocket connection management (fallback)
  const connectWebSocket = useCallback(async () => {
    if (!isAuthenticated || !isMounted()) return

    try {
      if (isMounted()) {
        setConnectionState(endpoint, {
          connectionType: 'websocket',
          lastError: undefined,
        })
      }

      await realtimeUtils.connectWebSocket(endpoint)

      if (isMounted()) {
        setConnectionState(endpoint, {
          isConnected: true,
          reconnectAttempts: 0,
        })
        onConnect?.()
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error('WebSocket connection failed')
      if (isMounted()) {
        setConnectionState(endpoint, {
          isConnected: false,
          lastError: err,
          reconnectAttempts: connectionState.reconnectAttempts + 1,
        })
        onError?.(err)
        scheduleReconnect()
      }
    }
  }, [isAuthenticated, isMounted, endpoint, onConnect, onError, setConnectionState, connectionState.reconnectAttempts])

  // Reconnection logic
  const scheduleReconnect = useCallback(() => {
    if (connectionState.reconnectAttempts >= reconnectAttempts || !isMounted()) {
      return
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }

    const delay = reconnectDelay * Math.pow(2, connectionState.reconnectAttempts)
    reconnectTimeoutRef.current = setTimeout(() => {
      if (isMounted()) {
        // Try SSE first, fallback to WebSocket
        connectSSE().catch(() => connectWebSocket())
      }
    }, delay)
  }, [connectionState.reconnectAttempts, reconnectAttempts, reconnectDelay, isMounted, connectSSE, connectWebSocket])

  // Event handling
  const handleEvent = useCallback((data: any, source: 'sse' | 'websocket') => {
    if (!isMounted()) return

    const event: RealtimeEvent = {
      id: data.id || `${Date.now()}-${Math.random()}`,
      type: data.type || 'unknown',
      data: data.data || data,
      timestamp: data.timestamp || new Date().toISOString(),
      source,
    }

    // Add event to Zustand store
    addEvent(endpoint, event)
    onEvent?.(event)
  }, [isMounted, endpoint, addEvent, onEvent])

  // Set up event listeners
  useEffect(() => {
    const sseHandler = (data: any) => handleEvent(data, 'sse')
    const wsHandler = (data: any) => handleEvent(data, 'websocket')

    realtimeUtils.onSSEMessage(sseHandler)
    realtimeUtils.onWebSocketMessage(wsHandler)

    // Store cleanup functions
    const cleanup = () => {
      sseClient.off('message', sseHandler)
      wsClient.off('message', wsHandler)
    }

    cleanupFunctionsRef.current.push(cleanup)

    return cleanup
  }, [handleEvent])

  // Connection lifecycle and comprehensive cleanup
  useEffect(() => {
    if (enabled && isAuthenticated && isMounted()) {
      // Try SSE first, fallback to WebSocket
      connectSSE().catch(() => connectWebSocket())
    }

    return () => {
      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = undefined
      }

      // Disconnect all connections
      realtimeUtils.disconnectSSE()
      realtimeUtils.disconnectWebSocket()

      // Run all cleanup functions
      cleanupFunctionsRef.current.forEach(cleanup => {
        try {
          cleanup()
        } catch (error) {
          console.warn('Error during cleanup:', error)
        }
      })
      cleanupFunctionsRef.current = []

      // Remove connection from store
      removeConnection(endpoint)
    }
  }, [enabled, isAuthenticated, isMounted, connectSSE, connectWebSocket])

  // Manual connection control
  const connect = useCallback(() => {
    if (!isMounted()) return

    setConnectionState(endpoint, { reconnectAttempts: 0 })
    connectSSE().catch(() => connectWebSocket())
  }, [isMounted, endpoint, setConnectionState, connectSSE, connectWebSocket])

  const disconnect = useCallback(() => {
    // Clear reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }

    // Disconnect connections
    realtimeUtils.disconnectSSE()
    realtimeUtils.disconnectWebSocket()

    if (isMounted()) {
      setConnectionState(endpoint, {
        isConnected: false,
        connectionType: 'none',
        reconnectAttempts: 0,
      })
    }

    onDisconnect?.()
  }, [isMounted, endpoint, setConnectionState, onDisconnect])

  // Event filtering and querying
  const getEventsByTypeLocal = useCallback((eventType: string) => {
    return getEventsByType(endpoint, eventType)
  }, [endpoint, getEventsByType])

  const getEventsAfter = useCallback((timestamp: string) => {
    return events.filter(event => event.timestamp > timestamp)
  }, [events])

  const getLatestEventLocal = useCallback((eventType?: string) => {
    return getLatestEvent(endpoint, eventType)
  }, [endpoint, getLatestEvent])

  const clearEventsLocal = useCallback(() => {
    if (isMounted()) {
      clearEvents(endpoint)
    }
  }, [isMounted, endpoint, clearEvents])

  return {
    // Connection state
    isConnected: connectionState.isConnected,
    connectionType: connectionState.connectionType,
    reconnectAttempts: connectionState.reconnectAttempts,
    lastError: connectionState.lastError,

    // Event data
    events,
    eventCount,
    latestEvent: getLatestEventLocal(),

    // Connection control
    connect,
    disconnect,

    // Event utilities
    getEventsByType: getEventsByTypeLocal,
    getEventsAfter,
    getLatestEvent: getLatestEventLocal,
    clearEvents: clearEventsLocal,

    // Status helpers
    isReconnecting: connectionState.reconnectAttempts > 0 && !connectionState.isConnected,
    canReconnect: connectionState.reconnectAttempts < reconnectAttempts,
  }
}

// Specialized hooks for specific data types
export function useCameraEvents(cameraId?: string) {
  const endpoint = cameraId
    ? `${ENDPOINTS.REALTIME.SSE.CAMERAS}?camera_ids=${cameraId}`
    : ENDPOINTS.REALTIME.SSE.CAMERAS

  return useRealTimeData(endpoint, {
    eventTypes: ['status_change', 'detection_result', 'health_update'],
  })
}

export function useAnalyticsEvents() {
  return useRealTimeData(ENDPOINTS.REALTIME.SSE.ANALYTICS, {
    eventTypes: ['metrics_update', 'incident_alert', 'prediction_update'],
  })
}

export function useSystemEvents() {
  return useRealTimeData(ENDPOINTS.REALTIME.SSE.SYSTEM, {
    eventTypes: ['health_update', 'alert', 'maintenance'],
  })
}

// Hook for combining multiple real-time data sources
export function useMultipleRealTimeData(
  endpoints: { endpoint: string; options?: UseRealTimeDataOptions }[]
) {
  const connections = endpoints.map(({ endpoint, options }) =>
    useRealTimeData(endpoint, options)
  )

  const allEvents = connections.flatMap(conn => conn.events)
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())

  const isAnyConnected = connections.some(conn => conn.isConnected)
  const allConnected = connections.every(conn => conn.isConnected)
  const totalEventCount = connections.reduce((sum, conn) => sum + conn.eventCount, 0)

  return {
    connections,
    allEvents,
    isAnyConnected,
    allConnected,
    totalEventCount,
    connectAll: () => connections.forEach(conn => conn.connect()),
    disconnectAll: () => connections.forEach(conn => conn.disconnect()),
  }
}
