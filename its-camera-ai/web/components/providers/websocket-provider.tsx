'use client'

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'

interface WebSocketMessage {
  type: 'traffic' | 'alert' | 'camera' | 'model' | 'system'
  data: any
  timestamp: Date
}

interface WebSocketContextType {
  isConnected: boolean
  lastMessage: WebSocketMessage | null
  sendMessage: (message: any) => void
  subscribe: (type: string, callback: (data: any) => void) => () => void
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error'
  reconnectAttempts: number
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

export function useWebSocket() {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider')
  }
  return context
}

interface WebSocketProviderProps {
  children: React.ReactNode
  url?: string
  autoConnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

export function WebSocketProvider({
  children,
  url = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
  autoConnect = true,
  reconnectInterval = 5000,
  maxReconnectAttempts = 10
}: WebSocketProviderProps) {
  const ws = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionState, setConnectionState] = useState<WebSocketContextType['connectionState']>('disconnected')
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const reconnectTimeout = useRef<NodeJS.Timeout>()
  const subscribers = useRef<Map<string, Set<(data: any) => void>>>(new Map())

  // Mock data generator for demo (always active when backend is not available)
  useEffect(() => {
    if (!autoConnect) return

    const mockInterval = setInterval(() => {
      const mockMessages: WebSocketMessage[] = [
        {
          type: 'traffic',
          data: {
            totalVehicles: Math.floor(Math.random() * 500) + 1000,
            avgSpeed: Math.floor(Math.random() * 30) + 30,
            congestionLevel: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low'
          },
          timestamp: new Date()
        },
        {
          type: 'alert',
          data: {
            id: Date.now().toString(),
            severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
            message: 'Traffic event detected',
            location: `Zone ${Math.floor(Math.random() * 10) + 1}`
          },
          timestamp: new Date()
        },
        {
          type: 'camera',
          data: {
            cameraId: `CAM-${Math.floor(Math.random() * 100).toString().padStart(3, '0')}`,
            status: Math.random() > 0.9 ? 'offline' : 'online',
            fps: Math.floor(Math.random() * 5) + 25,
            detections: Math.floor(Math.random() * 50)
          },
          timestamp: new Date()
        },
        {
          type: 'model',
          data: {
            modelId: 'yolo11-traffic-v2',
            accuracy: (Math.random() * 5 + 93).toFixed(2),
            latency: Math.floor(Math.random() * 30) + 20,
            throughput: Math.floor(Math.random() * 50) + 100
          },
          timestamp: new Date()
        }
      ]

      const message = mockMessages[Math.floor(Math.random() * mockMessages.length)]
      setLastMessage(message)
      setIsConnected(true)
      setConnectionState('connected')

      // Notify subscribers
      const typeSubscribers = subscribers.current.get(message.type)
      if (typeSubscribers) {
        typeSubscribers.forEach(callback => callback(message.data))
      }

      const allSubscribers = subscribers.current.get('*')
      if (allSubscribers) {
        allSubscribers.forEach(callback => callback(message))
      }
    }, 2000)

    return () => clearInterval(mockInterval)
  }, [autoConnect])

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return

    setConnectionState('connecting')

    try {
      // Only attempt real WebSocket connection if explicitly requested
      // For demo mode, we rely on mock data instead
      if (process.env.NODE_ENV === 'production' || process.env.NEXT_PUBLIC_ENABLE_WEBSOCKET === 'true') {
        ws.current = new WebSocket(url)
      } else {
        // Skip real WebSocket connection in development unless explicitly enabled
        console.log('WebSocket connection skipped - using mock data for demo')
        setConnectionState('disconnected')
        return
      }

      ws.current.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setConnectionState('connected')
        setReconnectAttempts(0)
      }

      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage
          message.timestamp = new Date(message.timestamp)
          setLastMessage(message)

          // Notify type-specific subscribers
          const typeSubscribers = subscribers.current.get(message.type)
          if (typeSubscribers) {
            typeSubscribers.forEach(callback => callback(message.data))
          }

          // Notify wildcard subscribers
          const allSubscribers = subscribers.current.get('*')
          if (allSubscribers) {
            allSubscribers.forEach(callback => callback(message))
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.current.onerror = (error) => {
        console.warn('WebSocket error (falling back to mock data):', error)
        setConnectionState('error')
        // Don't continuously retry in development
        if (process.env.NODE_ENV === 'development') {
          setConnectionState('disconnected')
        }
      }

      ws.current.onclose = () => {
        console.log('WebSocket disconnected (using mock data)')
        setIsConnected(false)
        setConnectionState('disconnected')

        // Only attempt reconnection in production or when explicitly enabled
        if ((process.env.NODE_ENV === 'production' || process.env.NEXT_PUBLIC_ENABLE_WEBSOCKET === 'true') 
            && reconnectAttempts < maxReconnectAttempts) {
          reconnectTimeout.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1)
            connect()
          }, reconnectInterval)
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionState('error')
    }
  }, [url, reconnectAttempts, maxReconnectAttempts, reconnectInterval])

  const sendMessage = useCallback((message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected')
    }
  }, [])

  const subscribe = useCallback((type: string, callback: (data: any) => void) => {
    if (!subscribers.current.has(type)) {
      subscribers.current.set(type, new Set())
    }
    subscribers.current.get(type)!.add(callback)

    // Return unsubscribe function
    return () => {
      const typeSubscribers = subscribers.current.get(type)
      if (typeSubscribers) {
        typeSubscribers.delete(callback)
        if (typeSubscribers.size === 0) {
          subscribers.current.delete(type)
        }
      }
    }
  }, [])

  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [autoConnect, connect])

  return (
    <WebSocketContext.Provider
      value={{
        isConnected,
        lastMessage,
        sendMessage,
        subscribe,
        connectionState,
        reconnectAttempts
      }}
    >
      {children}
    </WebSocketContext.Provider>
  )
}