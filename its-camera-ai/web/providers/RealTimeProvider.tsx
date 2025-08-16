/**
 * Real-Time Data Provider
 *
 * Integrates WebSocket and Analytics stores to provide seamless real-time
 * data processing and state management across the ITS Camera AI Dashboard.
 */

'use client';

import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { useWebSocketStore } from '@/stores/websocket';
import { useAnalyticsStore } from '@/stores/analytics';
import { EventType, WebSocketMessage } from '@/lib/websocket/client';

// Real-time provider context
interface RealTimeContextValue {
  isConnected: boolean;
  connectionQuality: 'excellent' | 'good' | 'poor' | 'critical' | 'unknown';
  lastUpdate: number;
  messageCount: number;
  processingLatency: number;
}

const RealTimeContext = createContext<RealTimeContextValue>({
  isConnected: false,
  connectionQuality: 'unknown',
  lastUpdate: 0,
  messageCount: 0,
  processingLatency: 0,
});

// Provider component props
interface RealTimeProviderProps {
  children: React.ReactNode;
  autoConnect?: boolean;
  endpoint?: 'analytics' | 'cameraFeed' | 'systemMonitoring';
  debugMode?: boolean;
}

// Provider component
export const RealTimeProvider: React.FC<RealTimeProviderProps> = ({
  children,
  autoConnect = true,
  endpoint = 'analytics',
  debugMode = false,
}) => {
  const messageCountRef = useRef(0);
  const lastUpdateRef = useRef(0);
  const processingLatencyRef = useRef(0);

  // WebSocket store actions and state
  const {
    connectionState,
    connect,
    disconnect,
    subscribe,
    clearErrors,
    updateSettings,
    getConnectionQuality,
    isHealthy,
    messageHistory,
  } = useWebSocketStore();

  // Analytics store actions
  const {
    processMessage,
    clearOldData,
    updateSettings: updateAnalyticsSettings,
  } = useAnalyticsStore();

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && connectionState === 'disconnected') {
      console.log('Auto-connecting to WebSocket...');
      connect(endpoint);
    }
  }, [autoConnect, connectionState, connect, endpoint]);

  // Update debug mode
  useEffect(() => {
    updateSettings({ debugMode });
  }, [debugMode, updateSettings]);

  // Subscribe to all event types for analytics processing
  useEffect(() => {
    if (connectionState === 'connected') {
      const eventTypes: EventType[] = [
        'metrics',
        'incident',
        'vehicle_count',
        'speed_update',
        'prediction',
        'camera_status',
        'system_health',
      ];

      eventTypes.forEach(eventType => {
        subscribe(eventType);
      });

      if (debugMode) {
        console.log('Subscribed to all event types for analytics processing');
      }
    }
  }, [connectionState, subscribe, debugMode]);

  // Process messages through event processing pipeline and analytics store
  useEffect(() => {
    // Import event processing integration
    import('@/lib/events').then(({ useEventIntegration }) => {
      const eventIntegration = useEventIntegration({
        enableProcessing: true,
        enableChartUpdates: true,
        enableAlerts: true,
        chartUpdateInterval: 100,
        maxChartDataPoints: 1000,
        enableDataDecimation: true,
      });

      // Set up message processing interval
      const processMessages = () => {
        const newMessages = messageHistory.slice(messageCountRef.current);

        if (newMessages.length > 0) {
          const startTime = performance.now();

          newMessages.forEach((message: WebSocketMessage) => {
            try {
              // Process through event pipeline first
              eventIntegration.processMessage(message).then(() => {
                // Then process through analytics store
                processMessage(message);
                messageCountRef.current++;
              }).catch(error => {
                console.error('Error in event processing pipeline:', error);
                // Fallback to direct analytics processing
                processMessage(message);
                messageCountRef.current++;
              });
            } catch (error) {
              console.error('Error processing message for analytics:', error);
            }
          });

          const processingTime = performance.now() - startTime;
          processingLatencyRef.current = processingTime;
          lastUpdateRef.current = Date.now();

          if (debugMode && newMessages.length > 0) {
            console.log(`Processed ${newMessages.length} messages in ${processingTime.toFixed(2)}ms`);

            // Log event processing metrics if in debug mode
            const metrics = eventIntegration.getMetrics();
            console.log('Event processing metrics:', {
              totalProcessed: metrics.totalProcessed,
              averageProcessingTime: metrics.averageProcessingTime,
              errorRate: metrics.errorRate,
              queueSize: metrics.queueSize,
            });
          }
        }
      };

      // Process messages every 100ms for real-time responsiveness
      const interval = setInterval(processMessages, 100);

      return () => clearInterval(interval);
    });
  }, [messageHistory, processMessage, debugMode]);

  // Cleanup old data periodically
  useEffect(() => {
    const cleanupInterval = setInterval(() => {
      clearOldData(); // Clear data older than 24 hours
    }, 60 * 60 * 1000); // Run every hour

    return () => clearInterval(cleanupInterval);
  }, [clearOldData]);

  // Error handling and recovery
  useEffect(() => {
    if (connectionState === 'error') {
      console.warn('WebSocket connection error detected, attempting recovery...');

      // Clear errors and attempt reconnection after 5 seconds
      setTimeout(() => {
        clearErrors();
        if (autoConnect) {
          connect(endpoint);
        }
      }, 5000);
    }
  }, [connectionState, clearErrors, connect, endpoint, autoConnect]);

  // Handle page visibility changes for connection management
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden - could suspend some processing
        if (debugMode) {
          console.log('Page hidden - maintaining connection');
        }
      } else {
        // Page is visible - ensure connection is active
        if (autoConnect && connectionState === 'disconnected') {
          connect(endpoint);
        }
        if (debugMode) {
          console.log('Page visible - ensuring active connection');
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [connectionState, connect, endpoint, autoConnect, debugMode]);

  // Performance monitoring with event processing metrics
  useEffect(() => {
    if (debugMode) {
      const logPerformance = async () => {
        const quality = getConnectionQuality();
        const healthy = isHealthy();

        try {
          // Get event processing metrics
          const { useEventIntegration } = await import('@/lib/events');
          const eventIntegration = useEventIntegration();
          const eventMetrics = eventIntegration.getMetrics();
          const healthStatus = eventIntegration.getHealthStatus();

          console.log('Real-time performance:', {
            connection: {
              state: connectionState,
              quality,
              healthy,
            },
            processing: {
              messageCount: messageCountRef.current,
              processingLatency: processingLatencyRef.current,
              lastUpdate: new Date(lastUpdateRef.current).toLocaleTimeString(),
            },
            eventPipeline: {
              totalProcessed: eventMetrics.totalProcessed,
              totalErrors: eventMetrics.totalErrors,
              averageProcessingTime: eventMetrics.averageProcessingTime,
              throughputPerSecond: eventMetrics.throughputPerSecond,
              errorRate: eventMetrics.errorRate,
              queueSize: eventMetrics.queueSize,
              healthStatus: healthStatus.status,
            },
          });
        } catch (error) {
          // Fallback to basic logging if event integration fails
          console.log('Real-time performance:', {
            connectionState,
            quality,
            healthy,
            messageCount: messageCountRef.current,
            processingLatency: processingLatencyRef.current,
            lastUpdate: new Date(lastUpdateRef.current).toLocaleTimeString(),
          });
        }
      };

      const performanceInterval = setInterval(logPerformance, 30000); // Every 30 seconds
      return () => clearInterval(performanceInterval);
    }
  }, [debugMode, connectionState, getConnectionQuality, isHealthy]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debugMode) {
        console.log('RealTimeProvider unmounting, disconnecting WebSocket...');
      }
      disconnect();
    };
  }, [disconnect, debugMode]);

  // Context value
  const contextValue: RealTimeContextValue = {
    isConnected: connectionState === 'connected',
    connectionQuality: getConnectionQuality(),
    lastUpdate: lastUpdateRef.current,
    messageCount: messageCountRef.current,
    processingLatency: processingLatencyRef.current,
  };

  return (
    <RealTimeContext.Provider value={contextValue}>
      {children}
    </RealTimeContext.Provider>
  );
};

// Hook to access real-time context
export const useRealTimeContext = (): RealTimeContextValue => {
  const context = useContext(RealTimeContext);
  if (!context) {
    throw new Error('useRealTimeContext must be used within a RealTimeProvider');
  }
  return context;
};

// Hook for components that need WebSocket connection status
export const useConnectionStatus = () => {
  const { isConnected, connectionQuality } = useRealTimeContext();
  const { connectionState, lastError } = useWebSocketStore(state => ({
    connectionState: state.connectionState,
    lastError: state.lastError,
  }));

  return {
    isConnected,
    connectionState,
    connectionQuality,
    lastError,
    hasError: !!lastError,
  };
};

// Hook for components that need processing status
export const useProcessingStatus = () => {
  const { messageCount, processingLatency, lastUpdate } = useRealTimeContext();
  const { isProcessing, processingStats } = useAnalyticsStore(state => ({
    isProcessing: state.isProcessing,
    processingStats: state.processingStats,
  }));

  return {
    isProcessing,
    messageCount,
    processingLatency,
    lastUpdate,
    processingStats,
  };
};

// Hook for real-time metrics with automatic updates and event processing integration
export const useRealTimeMetrics = () => {
  const { isConnected } = useRealTimeContext();
  const currentMetrics = useAnalyticsStore(state => state.currentMetrics);
  const [eventMetrics, setEventMetrics] = React.useState<any>(null);

  // Get event processing metrics
  React.useEffect(() => {
    const updateEventMetrics = async () => {
      try {
        const { useEventIntegration } = await import('@/lib/events');
        const eventIntegration = useEventIntegration();
        const metrics = eventIntegration.getMetrics();
        setEventMetrics(metrics);
      } catch (error) {
        console.warn('Failed to load event processing metrics:', error);
      }
    };

    updateEventMetrics();
    const interval = setInterval(updateEventMetrics, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return {
    isConnected,
    metrics: currentMetrics,
    eventMetrics,
    hasData: currentMetrics.lastUpdate > 0,
    isStale: Date.now() - currentMetrics.lastUpdate > 30000, // 30 seconds
    processingHealth: eventMetrics ? {
      totalProcessed: eventMetrics.totalProcessed,
      errorRate: eventMetrics.errorRate,
      averageProcessingTime: eventMetrics.averageProcessingTime,
      throughputPerSecond: eventMetrics.throughputPerSecond,
    } : null,
  };
};

// Hook for WebSocket controls with event processing integration
export const useWebSocketControls = () => {
  const { connect, disconnect, reconnect, clearErrors } = useWebSocketStore(state => ({
    connect: state.connect,
    disconnect: state.disconnect,
    reconnect: state.reconnect,
    clearErrors: state.clearErrors,
  }));

  const [eventIntegration, setEventIntegration] = React.useState<any>(null);

  // Initialize event integration
  React.useEffect(() => {
    import('@/lib/events').then(({ useEventIntegration }) => {
      const integration = useEventIntegration();
      setEventIntegration(integration);
    }).catch(error => {
      console.warn('Failed to initialize event integration:', error);
    });
  }, []);

  return {
    connect,
    disconnect,
    reconnect,
    clearErrors,
    // Event processing controls
    clearEventData: eventIntegration?.clearData,
    exportEventData: eventIntegration?.exportData,
    getEventHealth: eventIntegration?.getHealthStatus,
  };
};

export default RealTimeProvider;
