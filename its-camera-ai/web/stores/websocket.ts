/**
 * WebSocket State Management with Zustand
 *
 * Centralized store for managing WebSocket connections, real-time data,
 * and connection state across the ITS Camera AI Dashboard.
 */

import { create } from 'zustand';
import { subscribeWithSelector, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import {
  AnalyticsWebSocketClient,
  createWebSocketClient,
  WebSocketMessage,
  ConnectionState,
  EventType,
  SequenceGap,
  SequenceStats,
  ConnectionStats,
  MessagePriority,
  QueueStats,
  buildWebSocketURL,
  WS_ENDPOINTS
} from '@/lib/websocket/client';
import { authUtils } from '@/lib/auth/jwt';

// WebSocket store state interface
export interface WebSocketState {
  // Connection management
  client: AnalyticsWebSocketClient | null;
  connectionState: ConnectionState;
  isConnecting: boolean;
  isReconnecting: boolean;
  lastError: string | null;

  // Connection configuration
  endpoints: {
    analytics: string;
    cameraFeed: string;
    systemMonitoring: string;
  };

  // Real-time data streams
  messages: {
    [K in EventType]: WebSocketMessage[];
  };

  // Performance metrics
  connectionStats: ConnectionStats | null;
  sequenceStats: SequenceStats | null;
  queueStats: QueueStats | null;

  // Message management
  sequenceGaps: SequenceGap[];
  messageHistory: WebSocketMessage[];
  maxHistorySize: number;

  // Subscriptions and listeners
  activeSubscriptions: Set<EventType>;
  pausedEventTypes: Set<EventType>;

  // Settings
  autoReconnect: boolean;
  debugMode: boolean;

  // Actions
  connect: (endpoint?: keyof WebSocketState['endpoints']) => Promise<void>;
  disconnect: () => void;
  reconnect: () => void;
  subscribe: (eventType: EventType) => void;
  unsubscribe: (eventType: EventType) => void;
  pauseEventType: (eventType: EventType) => void;
  resumeEventType: (eventType: EventType) => void;
  clearMessages: (eventType?: EventType) => void;
  clearErrors: () => void;
  requestMissingMessages: (gaps: SequenceGap[]) => void;
  updateSettings: (settings: Partial<Pick<WebSocketState, 'autoReconnect' | 'debugMode' | 'maxHistorySize'>>) => void;

  // Data retrieval helpers
  getLatestMessage: (eventType: EventType) => WebSocketMessage | null;
  getMessages: (eventType: EventType, limit?: number) => WebSocketMessage[];
  getConnectionQuality: () => 'excellent' | 'good' | 'poor' | 'critical' | 'unknown';
  isHealthy: () => boolean;
}

// Default message structure for each event type
const createDefaultMessages = (): WebSocketState['messages'] => ({
  metrics: [],
  incident: [],
  vehicle_count: [],
  speed_update: [],
  prediction: [],
  camera_status: [],
  system_health: [],
});

// WebSocket store implementation
export const useWebSocketStore = create<WebSocketState>()(
  subscribeWithSelector(
    persist(
      immer((set, get) => ({
        // Initial state
        client: null,
        connectionState: 'disconnected',
        isConnecting: false,
        isReconnecting: false,
        lastError: null,

        endpoints: {
          analytics: buildWebSocketURL(WS_ENDPOINTS.CAMERA_ANALYTICS),
          cameraFeed: buildWebSocketURL(WS_ENDPOINTS.CAMERA_FEED),
          systemMonitoring: buildWebSocketURL(WS_ENDPOINTS.SYSTEM_MONITORING),
        },

        messages: createDefaultMessages(),

        connectionStats: null,
        sequenceStats: null,
        queueStats: null,

        sequenceGaps: [],
        messageHistory: [],
        maxHistorySize: 1000,

        activeSubscriptions: new Set(),
        pausedEventTypes: new Set(),

        autoReconnect: true,
        debugMode: false,

        // Actions
        connect: async (endpoint = 'analytics') => {
          const state = get();

          if (state.client || state.isConnecting) {
            console.warn('WebSocket already connected or connecting');
            return;
          }

          set(draft => {
            draft.isConnecting = true;
            draft.lastError = null;
          });

          try {
            const wsUrl = state.endpoints[endpoint];

            const client = createWebSocketClient({
              url: wsUrl,
              enableAuthentication: true,
              adaptiveHeartbeat: true,
              persistState: true,
              networkMonitoring: true,
              maxReconnectAttempts: state.autoReconnect ? 10 : 1,
            });

            // Set up event listeners
            const unsubscribeFunctions: (() => void)[] = [];

            // Connection state listener
            const connectionCleanup = client.onConnectionChange((connectionState) => {
              set(draft => {
                draft.connectionState = connectionState;
                draft.isConnecting = connectionState === 'connecting';
                draft.isReconnecting = connectionState === 'reconnecting';

                if (connectionState === 'connected') {
                  draft.lastError = null;
                }
              });

              // Update metrics when connected
              if (connectionState === 'connected') {
                get().updateMetrics();
              }
            });
            unsubscribeFunctions.push(connectionCleanup);

            // Error listener
            const errorCleanup = client.onError((error) => {
              set(draft => {
                draft.lastError = error.message;
                draft.isConnecting = false;
              });

              if (state.debugMode) {
                console.error('WebSocket error:', error);
              }
            });
            unsubscribeFunctions.push(errorCleanup);

            // Network status listener
            const networkCleanup = client.onNetworkChange((isOnline) => {
              if (state.debugMode) {
                console.log('Network status changed:', isOnline ? 'online' : 'offline');
              }
            });
            unsubscribeFunctions.push(networkCleanup);

            // Sequence gap listener
            const sequenceCleanup = client.onSequenceGap((gaps) => {
              set(draft => {
                draft.sequenceGaps.push(...gaps);
                // Keep only recent gaps (last 100)
                if (draft.sequenceGaps.length > 100) {
                  draft.sequenceGaps = draft.sequenceGaps.slice(-100);
                }
              });

              if (state.debugMode) {
                console.warn('Sequence gaps detected:', gaps);
              }
            });
            unsubscribeFunctions.push(sequenceCleanup);

            // Set up message listeners for each event type
            Object.keys(createDefaultMessages()).forEach(eventType => {
              const et = eventType as EventType;
              const messageCleanup = client.subscribe(et, (message) => {
                const currentState = get();

                // Skip if event type is paused
                if (currentState.pausedEventTypes.has(et)) {
                  return;
                }

                set(draft => {
                  // Add to specific event type array
                  draft.messages[et].push(message);

                  // Maintain size limits
                  if (draft.messages[et].length > 100) {
                    draft.messages[et] = draft.messages[et].slice(-100);
                  }

                  // Add to global message history
                  draft.messageHistory.push(message);
                  if (draft.messageHistory.length > draft.maxHistorySize) {
                    draft.messageHistory = draft.messageHistory.slice(-draft.maxHistorySize);
                  }
                });

                if (currentState.debugMode) {
                  console.log(`Received ${et} message:`, message);
                }
              });
              unsubscribeFunctions.push(messageCleanup);
            });

            // Store client and cleanup functions
            set(draft => {
              draft.client = client;
              // Store cleanup functions for later use
              (client as any).__cleanupFunctions = unsubscribeFunctions;
            });

          } catch (error) {
            set(draft => {
              draft.isConnecting = false;
              draft.lastError = error instanceof Error ? error.message : 'Connection failed';
            });

            console.error('Failed to connect WebSocket:', error);
          }
        },

        disconnect: () => {
          const state = get();

          if (state.client) {
            // Clean up listeners
            const cleanupFunctions = (state.client as any).__cleanupFunctions as (() => void)[] | undefined;
            if (cleanupFunctions) {
              cleanupFunctions.forEach(cleanup => cleanup());
            }

            state.client.disconnect();

            set(draft => {
              draft.client = null;
              draft.connectionState = 'disconnected';
              draft.isConnecting = false;
              draft.isReconnecting = false;
              draft.activeSubscriptions.clear();
            });
          }
        },

        reconnect: () => {
          const state = get();
          if (state.client) {
            state.client.reconnect();
          } else {
            state.connect();
          }
        },

        subscribe: (eventType: EventType) => {
          set(draft => {
            draft.activeSubscriptions.add(eventType);
            draft.pausedEventTypes.delete(eventType);
          });
        },

        unsubscribe: (eventType: EventType) => {
          set(draft => {
            draft.activeSubscriptions.delete(eventType);
          });
        },

        pauseEventType: (eventType: EventType) => {
          set(draft => {
            draft.pausedEventTypes.add(eventType);
          });
        },

        resumeEventType: (eventType: EventType) => {
          set(draft => {
            draft.pausedEventTypes.delete(eventType);
          });
        },

        clearMessages: (eventType?: EventType) => {
          set(draft => {
            if (eventType) {
              draft.messages[eventType] = [];
            } else {
              draft.messages = createDefaultMessages();
              draft.messageHistory = [];
            }
          });
        },

        clearErrors: () => {
          set(draft => {
            draft.lastError = null;
            draft.sequenceGaps = [];
          });
        },

        requestMissingMessages: (gaps: SequenceGap[]) => {
          const state = get();
          if (state.client) {
            state.client.requestMissingMessages(gaps);
          }
        },

        updateSettings: (settings) => {
          set(draft => {
            Object.assign(draft, settings);
          });
        },

        // Helper methods
        getLatestMessage: (eventType: EventType) => {
          const messages = get().messages[eventType];
          return messages.length > 0 ? messages[messages.length - 1] : null;
        },

        getMessages: (eventType: EventType, limit?: number) => {
          const messages = get().messages[eventType];
          return limit ? messages.slice(-limit) : messages;
        },

        getConnectionQuality: () => {
          const state = get();
          if (!state.client) return 'unknown';
          return state.client.getConnectionQuality();
        },

        isHealthy: () => {
          const state = get();
          return state.connectionState === 'connected' &&
                 state.lastError === null &&
                 state.getConnectionQuality() !== 'critical';
        },

        // Internal method to update metrics
        updateMetrics: () => {
          const state = get();
          if (state.client) {
            set(draft => {
              draft.connectionStats = state.client!.getConnectionStats();
              draft.sequenceStats = state.client!.getSequenceStats();
              const metrics = state.client!.getMetrics();
              draft.queueStats = metrics.queueStats;
            });
          }
        },
      })),
      {
        name: 'websocket-store',
        partialize: (state) => ({
          // Only persist settings and configuration
          autoReconnect: state.autoReconnect,
          debugMode: state.debugMode,
          maxHistorySize: state.maxHistorySize,
          pausedEventTypes: Array.from(state.pausedEventTypes),
        }),
        onRehydrate: (state) => {
          if (state) {
            // Convert array back to Set
            state.pausedEventTypes = new Set(state.pausedEventTypes as any);
          }
        },
      }
    )
  )
);

// Selector hooks for optimized re-renders
export const useConnectionState = () =>
  useWebSocketStore(state => state.connectionState);

export const useConnectionHealth = () =>
  useWebSocketStore(state => ({
    isHealthy: state.isHealthy(),
    quality: state.getConnectionQuality(),
    lastError: state.lastError,
  }));

export const useEventMessages = (eventType: EventType, limit?: number) =>
  useWebSocketStore(state => state.getMessages(eventType, limit));

export const useLatestMessage = (eventType: EventType) =>
  useWebSocketStore(state => state.getLatestMessage(eventType));

export const useWebSocketMetrics = () =>
  useWebSocketStore(state => ({
    connectionStats: state.connectionStats,
    sequenceStats: state.sequenceStats,
    queueStats: state.queueStats,
    sequenceGaps: state.sequenceGaps,
  }));

// Auto-connect hook for components
export const useAutoConnect = (endpoint?: keyof WebSocketState['endpoints']) => {
  const { connect, connectionState, autoReconnect } = useWebSocketStore(
    state => ({
      connect: state.connect,
      connectionState: state.connectionState,
      autoReconnect: state.autoReconnect,
    })
  );

  // Auto-connect when component mounts if not connected
  React.useEffect(() => {
    if (autoReconnect && connectionState === 'disconnected') {
      connect(endpoint);
    }
  }, [connect, connectionState, autoReconnect, endpoint]);
};

// Cleanup hook for unmounting
export const useWebSocketCleanup = () => {
  const disconnect = useWebSocketStore(state => state.disconnect);

  React.useEffect(() => {
    return () => {
      // Cleanup on unmount
      disconnect();
    };
  }, [disconnect]);
};

export default useWebSocketStore;
