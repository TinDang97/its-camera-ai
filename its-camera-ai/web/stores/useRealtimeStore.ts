'use client'

import { create } from 'zustand'
import { devtools, subscribeWithSelector } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import type { RealtimeEvent } from '@/hooks/useRealTimeData'
import type { User } from '@/hooks/useAuth'

interface ConnectionState {
  isConnected: boolean
  connectionType: 'sse' | 'websocket' | 'none'
  reconnectAttempts: number
  lastError?: Error
  endpoint: string
}

interface RealtimeState {
  // Authentication
  user: User | null
  isAuthenticated: boolean
  isAuthLoading: boolean
  authError: string | null

  // Connections
  connections: Record<string, ConnectionState>

  // Events
  events: Record<string, RealtimeEvent[]>
  eventCounts: Record<string, number>

  // Camera data
  cameras: Record<string, any>
  cameraEvents: RealtimeEvent[]

  // Analytics
  analytics: {
    metrics: any[]
    incidents: any[]
    predictions: any[]
  }

  // System
  systemHealth: any
  systemEvents: RealtimeEvent[]
}

interface RealtimeActions {
  // Authentication actions
  setUser: (user: User | null) => void
  setAuthLoading: (loading: boolean) => void
  setAuthError: (error: string | null) => void
  clearAuthError: () => void

  // Connection actions
  setConnectionState: (endpoint: string, state: Partial<ConnectionState>) => void
  removeConnection: (endpoint: string) => void

  // Event actions
  addEvent: (endpoint: string, event: RealtimeEvent) => void
  clearEvents: (endpoint?: string) => void
  getEventsByType: (endpoint: string, eventType: string) => RealtimeEvent[]
  getLatestEvent: (endpoint: string, eventType?: string) => RealtimeEvent | null

  // Camera actions
  updateCamera: (cameraId: string, data: any) => void
  removeCamera: (cameraId: string) => void

  // Analytics actions
  updateAnalytics: (type: 'metrics' | 'incidents' | 'predictions', data: any) => void

  // System actions
  updateSystemHealth: (health: any) => void

  // Cleanup actions
  cleanup: () => void
}

export type RealtimeStore = RealtimeState & RealtimeActions

export const useRealtimeStore = create<RealtimeStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        // Initial state
        user: null,
        isAuthenticated: false,
        isAuthLoading: true,
        authError: null,
        connections: {},
        events: {},
        eventCounts: {},
        cameras: {},
        cameraEvents: [],
        analytics: {
          metrics: [],
          incidents: [],
          predictions: [],
        },
        systemHealth: null,
        systemEvents: [],

        // Authentication actions
        setUser: (user) =>
          set((state) => {
            state.user = user
            state.isAuthenticated = !!user
            state.isAuthLoading = false
            state.authError = null
          }),

        setAuthLoading: (loading) =>
          set((state) => {
            state.isAuthLoading = loading
          }),

        setAuthError: (error) =>
          set((state) => {
            state.authError = error
            state.isAuthLoading = false
          }),

        clearAuthError: () =>
          set((state) => {
            state.authError = null
          }),

        // Connection actions
        setConnectionState: (endpoint, connectionState) =>
          set((state) => {
            if (!state.connections[endpoint]) {
              state.connections[endpoint] = {
                isConnected: false,
                connectionType: 'none',
                reconnectAttempts: 0,
                endpoint,
              }
            }
            Object.assign(state.connections[endpoint], connectionState)
          }),

        removeConnection: (endpoint) =>
          set((state) => {
            delete state.connections[endpoint]
            delete state.events[endpoint]
            delete state.eventCounts[endpoint]
          }),

        // Event actions
        addEvent: (endpoint, event) =>
          set((state) => {
            if (!state.events[endpoint]) {
              state.events[endpoint] = []
              state.eventCounts[endpoint] = 0
            }

            // Keep only last 100 events to prevent memory leaks
            state.events[endpoint] = [event, ...state.events[endpoint]].slice(0, 100)
            state.eventCounts[endpoint] += 1

            // Update specific event arrays based on endpoint
            if (endpoint.includes('/cameras')) {
              state.cameraEvents = [event, ...state.cameraEvents].slice(0, 100)

              // Update camera data if it's a camera event
              if (event.type === 'status_change' || event.type === 'detection_result') {
                const cameraId = event.data?.camera_id
                if (cameraId) {
                  if (!state.cameras[cameraId]) {
                    state.cameras[cameraId] = {}
                  }
                  Object.assign(state.cameras[cameraId], event.data)
                }
              }
            } else if (endpoint.includes('/analytics')) {
              // Update analytics data
              if (event.type === 'metrics_update') {
                state.analytics.metrics = [event.data, ...state.analytics.metrics].slice(0, 50)
              } else if (event.type === 'incident_alert') {
                state.analytics.incidents = [event.data, ...state.analytics.incidents].slice(0, 100)
              } else if (event.type === 'prediction_update') {
                state.analytics.predictions = [event.data, ...state.analytics.predictions].slice(0, 20)
              }
            } else if (endpoint.includes('/system')) {
              state.systemEvents = [event, ...state.systemEvents].slice(0, 50)

              if (event.type === 'health_update') {
                state.systemHealth = event.data
              }
            }
          }),

        clearEvents: (endpoint) =>
          set((state) => {
            if (endpoint) {
              state.events[endpoint] = []
              state.eventCounts[endpoint] = 0
            } else {
              state.events = {}
              state.eventCounts = {}
              state.cameraEvents = []
              state.systemEvents = []
              state.analytics = {
                metrics: [],
                incidents: [],
                predictions: [],
              }
            }
          }),

        getEventsByType: (endpoint, eventType) => {
          const events = get().events[endpoint] || []
          return events.filter(event => event.type === eventType)
        },

        getLatestEvent: (endpoint, eventType) => {
          const events = eventType
            ? get().getEventsByType(endpoint, eventType)
            : get().events[endpoint] || []
          return events[0] || null
        },

        // Camera actions
        updateCamera: (cameraId, data) =>
          set((state) => {
            if (!state.cameras[cameraId]) {
              state.cameras[cameraId] = {}
            }
            Object.assign(state.cameras[cameraId], data)
          }),

        removeCamera: (cameraId) =>
          set((state) => {
            delete state.cameras[cameraId]
          }),

        // Analytics actions
        updateAnalytics: (type, data) =>
          set((state) => {
            state.analytics[type] = Array.isArray(data) ? data : [data, ...state.analytics[type]].slice(0, 100)
          }),

        // System actions
        updateSystemHealth: (health) =>
          set((state) => {
            state.systemHealth = health
          }),

        // Cleanup actions
        cleanup: () =>
          set((state) => {
            state.connections = {}
            state.events = {}
            state.eventCounts = {}
            state.cameras = {}
            state.cameraEvents = []
            state.systemEvents = []
            state.analytics = {
              metrics: [],
              incidents: [],
              predictions: [],
            }
            state.systemHealth = null
          }),
      })),
      {
        name: 'realtime-store',
      }
    )
  )
)

// Selectors for performance optimization
export const useAuthState = () => useRealtimeStore((state) => ({
  user: state.user,
  isAuthenticated: state.isAuthenticated,
  isAuthLoading: state.isAuthLoading,
  authError: state.authError,
}))

export const useConnectionState = (endpoint: string) =>
  useRealtimeStore((state) => state.connections[endpoint] || {
    isConnected: false,
    connectionType: 'none' as const,
    reconnectAttempts: 0,
    endpoint,
  })

export const useRealtimeEvents = (endpoint: string) =>
  useRealtimeStore((state) => ({
    events: state.events[endpoint] || [],
    eventCount: state.eventCounts[endpoint] || 0,
  }))

export const useCameraData = () =>
  useRealtimeStore((state) => ({
    cameras: state.cameras,
    cameraEvents: state.cameraEvents,
  }))

export const useAnalyticsData = () =>
  useRealtimeStore((state) => state.analytics)

export const useSystemData = () =>
  useRealtimeStore((state) => ({
    systemHealth: state.systemHealth,
    systemEvents: state.systemEvents,
  }))

// Action hooks
export const useRealtimeActions = () =>
  useRealtimeStore((state) => ({
    setUser: state.setUser,
    setAuthLoading: state.setAuthLoading,
    setAuthError: state.setAuthError,
    clearAuthError: state.clearAuthError,
    setConnectionState: state.setConnectionState,
    removeConnection: state.removeConnection,
    addEvent: state.addEvent,
    clearEvents: state.clearEvents,
    getEventsByType: state.getEventsByType,
    getLatestEvent: state.getLatestEvent,
    updateCamera: state.updateCamera,
    removeCamera: state.removeCamera,
    updateAnalytics: state.updateAnalytics,
    updateSystemHealth: state.updateSystemHealth,
    cleanup: state.cleanup,
  }))

// Subscription helpers
export const subscribeToAuth = (callback: (auth: any) => void) =>
  useRealtimeStore.subscribe(
    (state) => ({
      user: state.user,
      isAuthenticated: state.isAuthenticated,
      isAuthLoading: state.isAuthLoading,
      authError: state.authError,
    }),
    callback
  )

export const subscribeToConnection = (endpoint: string, callback: (connection: ConnectionState) => void) =>
  useRealtimeStore.subscribe(
    (state) => state.connections[endpoint],
    callback
  )

export const subscribeToEvents = (endpoint: string, callback: (events: RealtimeEvent[]) => void) =>
  useRealtimeStore.subscribe(
    (state) => state.events[endpoint] || [],
    callback
  )
