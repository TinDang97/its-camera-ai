# Live Event Components Documentation

## Overview

The ITS Camera AI dashboard provides real-time event monitoring through Server-Sent Events (SSE), delivering instant updates for traffic incidents, camera status changes, system alerts, and analytics data. SSE provides a reliable, efficient streaming solution with automatic reconnection and built-in browser support.

## Core Architecture

### SSE Event Manager

```tsx
// lib/events/EventManager.ts
'use client';

import { env } from '@/lib/env';

export type EventType = 
  | 'traffic_incident' 
  | 'camera_status' 
  | 'system_alert' 
  | 'analytics_update'
  | 'user_action'
  | 'health_check';

export interface LiveEvent {
  id: string;
  type: EventType;
  timestamp: string;
  source: string;
  data: Record<string, any>;
  priority: 'low' | 'medium' | 'high' | 'critical';
  location?: {
    cameraId: string;
    coordinates: { lat: number; lng: number };
    address: string;
  };
}

export class SSEEventManager {
  private eventSource: EventSource | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners = new Map<EventType, Set<(event: LiveEvent) => void>>();
  private connectionState: 'connecting' | 'connected' | 'disconnected' | 'error' = 'disconnected';
  private lastEventId: string | null = null;
  private isOnline = true;

  constructor() {
    // Monitor online status
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => {
        this.isOnline = true;
        this.connect();
      });
      
      window.addEventListener('offline', () => {
        this.isOnline = false;
        this.disconnect();
      });
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.isOnline) {
        reject(new Error('Device is offline'));
        return;
      }

      if (this.eventSource?.readyState === EventSource.OPEN) {
        resolve();
        return;
      }

      this.connectionState = 'connecting';
      
      try {
        // Construct SSE URL with Last-Event-ID for resume functionality
        const url = new URL(env.NEXT_PUBLIC_SSE_URL);
        if (this.lastEventId) {
          url.searchParams.set('lastEventId', this.lastEventId);
        }
        
        this.eventSource = new EventSource(url.toString());
        
        this.eventSource.onopen = () => {
          console.log('📡 SSE connected');
          this.connectionState = 'connected';
          this.reconnectAttempts = 0;
          resolve();
        };

        this.eventSource.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.eventSource.onerror = (error) => {
          console.error('📡 SSE error:', error);
          this.connectionState = 'error';
          
          if (this.eventSource?.readyState === EventSource.CLOSED) {
            this.connectionState = 'disconnected';
            if (this.isOnline && this.reconnectAttempts < this.maxReconnectAttempts) {
              this.scheduleReconnect();
            }
          }
          
          reject(error);
        };
        
        // Set up custom event listeners for typed events
        this.setupCustomEventListeners();
        
      } catch (error) {
        this.connectionState = 'error';
        reject(error);
      }
    });
  }

  private setupCustomEventListeners(): void {
    if (!this.eventSource) return;

    // Listen for specific event types
    ['traffic_incident', 'camera_status', 'system_alert', 'analytics_update', 'user_action'].forEach(eventType => {
      this.eventSource!.addEventListener(eventType, (event: MessageEvent) => {
        this.handleTypedEvent(eventType as EventType, event);
      });
    });
  }

  private handleMessage(event: MessageEvent) {
    try {
      const liveEvent: LiveEvent = JSON.parse(event.data);
      this.lastEventId = event.lastEventId || liveEvent.id;
      this.processEvent(liveEvent);
    } catch (error) {
      console.error('Error parsing SSE message:', error);
    }
  }
  
  private handleTypedEvent(eventType: EventType, event: MessageEvent) {
    try {
      const liveEvent: LiveEvent = JSON.parse(event.data);
      this.lastEventId = event.lastEventId || liveEvent.id;
      this.processEvent({ ...liveEvent, type: eventType });
    } catch (error) {
      console.error(`Error parsing ${eventType} event:`, error);
    }
  }

  private processEvent(liveEvent: LiveEvent) {
    // Validate event structure
    if (!this.isValidEvent(liveEvent)) {
      console.warn('Invalid event received:', liveEvent);
      return;
    }

    // Only process if online
    if (!this.isOnline) {
      return;
    }

    // Emit to type-specific listeners
    const typeListeners = this.listeners.get(liveEvent.type);
    if (typeListeners) {
      typeListeners.forEach(listener => {
        try {
          listener(liveEvent);
        } catch (error) {
          console.error('Error in event listener:', error);
        }
      });
    }

    // Emit to global listeners
    const globalListeners = this.listeners.get('*' as EventType);
    if (globalListeners) {
      globalListeners.forEach(listener => {
        try {
          listener(liveEvent);
        } catch (error) {
          console.error('Error in global listener:', error);
        }
      });
    }
  }

  private isValidEvent(event: any): event is LiveEvent {
    return (
      event &&
      typeof event.id === 'string' &&
      typeof event.type === 'string' &&
      typeof event.timestamp === 'string' &&
      typeof event.source === 'string' &&
      typeof event.data === 'object' &&
      ['low', 'medium', 'high', 'critical'].includes(event.priority)
    );
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    
    setTimeout(() => {
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.connect().catch(() => {
        // Reconnection failed, will try again
      });
    }, delay);
  }

  // SSE doesn't need manual heartbeat - it's handled by the browser
  // and the server can send keep-alive messages automatically

  subscribe(eventType: EventType | '*', listener: (event: LiveEvent) => void): () => void {
    if (!this.listeners.has(eventType as EventType)) {
      this.listeners.set(eventType as EventType, new Set());
    }
    
    this.listeners.get(eventType as EventType)!.add(listener);

    // Return unsubscribe function
    return () => {
      this.listeners.get(eventType as EventType)?.delete(listener);
    };
  }

  // SSE is read-only, so no send method needed for client-to-server communication
  // Use regular HTTP POST requests for client actions
  
  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.connectionState = 'disconnected';
  }
  
  reconnect() {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connect();
  }
  
  getLastEventId(): string | null {
    return this.lastEventId;
  }

  getConnectionState() {
    return this.connectionState;
  }
}

// Singleton instance
export const eventManager = new SSEEventManager();
```

### React Hooks for Live Events

```tsx
// hooks/useLiveEvents.ts
'use client';

import { useEffect, useCallback, useRef, useState } from 'react';
import { eventManager, EventType, LiveEvent } from '@/lib/events/EventManager';

export function useLiveEvents(eventType?: EventType) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<LiveEvent | null>(null);
  const [eventHistory, setEventHistory] = useState<LiveEvent[]>([]);
  const maxHistorySize = 100;

  const handleEvent = useCallback((event: LiveEvent) => {
    setLastEvent(event);
    setEventHistory(prev => {
      const newHistory = [event, ...prev].slice(0, maxHistorySize);
      return newHistory;
    });
  }, []);

  useEffect(() => {
    const connectAndSubscribe = async () => {
      try {
        await eventManager.connect();
        setIsConnected(true);
        
        const unsubscribe = eventManager.subscribe(
          eventType || '*',
          handleEvent
        );

        return unsubscribe;
      } catch (error) {
        console.error('Failed to connect to SSE:', error);
        setIsConnected(false);
        return () => {};
      }
    };

    const unsubscribePromise = connectAndSubscribe();

    return () => {
      unsubscribePromise.then(unsubscribe => unsubscribe());
    };
  }, [eventType, handleEvent]);

  useEffect(() => {
    const handleConnectionChange = () => {
      setIsConnected(eventManager.getConnectionState() === 'connected');
    };

    // Check connection state periodically
    const interval = setInterval(handleConnectionChange, 1000);

    return () => clearInterval(interval);
  }, []);

  // SSE is read-only - use HTTP requests for client actions
  const sendAction = useCallback(async (action: any) => {
    try {
      const response = await fetch('/api/events/actions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(action),
      });
      if (!response.ok) {
        throw new Error('Failed to send action');
      }
    } catch (error) {
      console.error('Error sending action:', error);
    }
  }, []);

  return {
    isConnected,
    lastEvent,
    eventHistory,
    sendAction,
  };
}

export function useTrafficIncidents() {
  const { lastEvent, eventHistory, isConnected } = useLiveEvents('traffic_incident');
  
  const incidents = eventHistory.filter(event => event.type === 'traffic_incident');
  
  return {
    latestIncident: lastEvent?.type === 'traffic_incident' ? lastEvent : null,
    incidents,
    isConnected,
  };
}

export function useCameraStatus() {
  const { lastEvent, eventHistory, isConnected } = useLiveEvents('camera_status');
  
  const [cameraStatuses, setCameraStatuses] = useState<Map<string, any>>(new Map());

  useEffect(() => {
    if (lastEvent?.type === 'camera_status') {
      setCameraStatuses(prev => {
        const newMap = new Map(prev);
        newMap.set(lastEvent.data.cameraId, lastEvent.data);
        return newMap;
      });
    }
  }, [lastEvent]);

  return {
    cameraStatuses: Object.fromEntries(cameraStatuses),
    latestUpdate: lastEvent?.type === 'camera_status' ? lastEvent : null,
    isConnected,
  };
}
```

## Live Event Components

### 1. Live Event Feed

```tsx
// components/live/LiveEventFeed.tsx
'use client';

import { memo, useMemo, useState, useCallback } from 'react';
import { useLiveEvents } from '@/hooks/useLiveEvents';
import { LiveEvent } from '@/lib/events/EventManager';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  AlertTriangle, 
  Camera, 
  Activity, 
  TrendingUp,
  Users,
  Filter,
  Pause,
  Play
} from 'lucide-react';

const EVENT_ICONS = {
  traffic_incident: AlertTriangle,
  camera_status: Camera,
  system_alert: Activity,
  analytics_update: TrendingUp,
  user_action: Users,
} as const;

const EVENT_COLORS = {
  critical: 'bg-red-500 text-white',
  high: 'bg-orange-500 text-white',
  medium: 'bg-yellow-500 text-black',
  low: 'bg-blue-500 text-white',
} as const;

interface LiveEventFeedProps {
  maxEvents?: number;
  autoScroll?: boolean;
  showFilters?: boolean;
  compact?: boolean;
}

export const LiveEventFeed = memo<LiveEventFeedProps>(({
  maxEvents = 50,
  autoScroll = true,
  showFilters = true,
  compact = false,
}) => {
  const { eventHistory, isConnected } = useLiveEvents();
  const [isPaused, setIsPaused] = useState(false);
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [selectedPriorities, setSelectedPriorities] = useState<Set<string>>(new Set());

  const filteredEvents = useMemo(() => {
    let filtered = eventHistory.slice(0, maxEvents);

    if (selectedTypes.size > 0) {
      filtered = filtered.filter(event => selectedTypes.has(event.type));
    }

    if (selectedPriorities.size > 0) {
      filtered = filtered.filter(event => selectedPriorities.has(event.priority));
    }

    return filtered;
  }, [eventHistory, maxEvents, selectedTypes, selectedPriorities]);

  const displayEvents = isPaused ? filteredEvents : filteredEvents;

  const togglePause = useCallback(() => {
    setIsPaused(!isPaused);
  }, [isPaused]);

  const toggleTypeFilter = useCallback((type: string) => {
    setSelectedTypes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(type)) {
        newSet.delete(type);
      } else {
        newSet.add(type);
      }
      return newSet;
    });
  }, []);

  const togglePriorityFilter = useCallback((priority: string) => {
    setSelectedPriorities(prev => {
      const newSet = new Set(prev);
      if (newSet.has(priority)) {
        newSet.delete(priority);
      } else {
        newSet.add(priority);
      }
      return newSet;
    });
  }, []);

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-green-500" />
            <CardTitle className="text-lg font-semibold">Live Events</CardTitle>
            <Badge
              variant={isConnected ? 'default' : 'destructive'}
              className="text-xs"
            >
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={togglePause}
              className="h-8 w-8 p-0"
            >
              {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
            </Button>
            
            {showFilters && (
              <Button variant="ghost" size="sm">
                <Filter className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {showFilters && (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-1">
              {Object.keys(EVENT_ICONS).map(type => (
                <Button
                  key={type}
                  variant={selectedTypes.has(type) ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => toggleTypeFilter(type)}
                  className="h-6 px-2 text-xs"
                >
                  {type.replace('_', ' ')}
                </Button>
              ))}
            </div>
            
            <div className="flex flex-wrap gap-1">
              {Object.keys(EVENT_COLORS).map(priority => (
                <Button
                  key={priority}
                  variant={selectedPriorities.has(priority) ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => togglePriorityFilter(priority)}
                  className="h-6 px-2 text-xs capitalize"
                >
                  {priority}
                </Button>
              ))}
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="p-0">
        <ScrollArea className="h-96">
          <div className="space-y-2 p-4 pt-0">
            {displayEvents.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-sm">No events to display</p>
                <p className="text-xs">Events will appear here in real-time</p>
              </div>
            ) : (
              displayEvents.map(event => (
                <LiveEventItem
                  key={event.id}
                  event={event}
                  compact={compact}
                />
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
});

const LiveEventItem = memo<{
  event: LiveEvent;
  compact: boolean;
}>(({ event, compact }) => {
  const Icon = EVENT_ICONS[event.type as keyof typeof EVENT_ICONS] || Activity;
  const timeAgo = useMemo(() => {
    const diff = Date.now() - new Date(event.timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    
    if (minutes > 0) return `${minutes}m ago`;
    return `${seconds}s ago`;
  }, [event.timestamp]);

  if (compact) {
    return (
      <div className="flex items-center gap-2 py-1">
        <Icon className="h-3 w-3 text-gray-500 flex-shrink-0" />
        <span className="text-xs truncate flex-1">{event.data.message || event.type}</span>
        <span className="text-xs text-gray-400 flex-shrink-0">{timeAgo}</span>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
      <div className="flex-shrink-0 mt-0.5">
        <div className={`p-1.5 rounded-full ${EVENT_COLORS[event.priority]}`}>
          <Icon className="h-3 w-3" />
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <p className="font-medium text-sm truncate">
            {event.data.title || event.type.replace('_', ' ')}
          </p>
          <Badge variant="outline" className="text-xs">
            {event.source}
          </Badge>
        </div>

        <p className="text-xs text-gray-600 mb-1">
          {event.data.message || event.data.description || 'No details available'}
        </p>

        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>{timeAgo}</span>
          {event.location && (
            <span className="truncate ml-2">
              📍 {event.location.address || event.location.cameraId}
            </span>
          )}
        </div>
      </div>
    </div>
  );
});

LiveEventFeed.displayName = 'LiveEventFeed';
LiveEventItem.displayName = 'LiveEventItem';
```

### 2. Real-Time Alert Banner

```tsx
// components/live/RealtimeAlertBanner.tsx
'use client';

import { memo, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTrafficIncidents } from '@/hooks/useLiveEvents';
import { Button } from '@/components/ui/button';
import { X, AlertTriangle, Navigation, Clock } from 'lucide-react';

export const RealtimeAlertBanner = memo(() => {
  const { latestIncident, isConnected } = useTrafficIncidents();
  const [isVisible, setIsVisible] = useState(false);
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (latestIncident && !dismissedIds.has(latestIncident.id)) {
      setIsVisible(true);
      
      // Auto-dismiss after 30 seconds for non-critical incidents
      if (latestIncident.priority !== 'critical') {
        const timer = setTimeout(() => {
          setIsVisible(false);
        }, 30000);
        
        return () => clearTimeout(timer);
      }
    }
  }, [latestIncident, dismissedIds]);

  const handleDismiss = useCallback(() => {
    if (latestIncident) {
      setDismissedIds(prev => new Set(prev).add(latestIncident.id));
    }
    setIsVisible(false);
  }, [latestIncident]);

  const handleViewDetails = useCallback(() => {
    if (latestIncident?.location?.coordinates) {
      // Navigate to map view or incident details
      window.open(`/incidents/${latestIncident.id}`, '_blank');
    }
  }, [latestIncident]);

  if (!isConnected || !latestIncident || !isVisible) {
    return null;
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-600 border-red-700';
      case 'high': return 'bg-orange-600 border-orange-700';
      case 'medium': return 'bg-yellow-600 border-yellow-700';
      default: return 'bg-blue-600 border-blue-700';
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const diff = Date.now() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    return minutes < 1 ? 'Just now' : `${minutes}m ago`;
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: -100, opacity: 0 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className={`fixed top-0 left-0 right-0 z-50 ${getPriorityColor(latestIncident.priority)} text-white shadow-lg`}
      >
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <motion.div
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ repeat: Infinity, duration: 2 }}
              >
                <AlertTriangle className="h-5 w-5 flex-shrink-0" />
              </motion.div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-sm">
                    LIVE: {latestIncident.data.title || 'Traffic Incident'}
                  </span>
                  <span className="px-2 py-0.5 bg-white/20 rounded text-xs font-medium">
                    {latestIncident.priority.toUpperCase()}
                  </span>
                </div>
                
                <div className="flex items-center gap-4 text-xs">
                  <span>{latestIncident.data.description || 'No details available'}</span>
                  
                  {latestIncident.location && (
                    <div className="flex items-center gap-1">
                      <Navigation className="h-3 w-3" />
                      <span>{latestIncident.location.address || latestIncident.location.cameraId}</span>
                    </div>
                  )}
                  
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    <span>{formatTimeAgo(latestIncident.timestamp)}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleViewDetails}
                className="text-white hover:bg-white/20 h-8 text-xs"
              >
                View Details
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDismiss}
                className="text-white hover:bg-white/20 h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
});

RealtimeAlertBanner.displayName = 'RealtimeAlertBanner';
```

### 3. Live Status Indicators

```tsx
// components/live/LiveStatusIndicators.tsx
'use client';

import { memo } from 'react';
import { useCameraStatus, useLiveEvents } from '@/hooks/useLiveEvents';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  Camera, 
  Wifi, 
  WifiOff, 
  Activity, 
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';

export const LiveStatusIndicators = memo(() => {
  const { isConnected } = useLiveEvents();
  const { cameraStatuses } = useCameraStatus();

  const cameraStats = Object.values(cameraStatuses).reduce(
    (acc, status) => {
      acc.total++;
      if (status.online) acc.online++;
      if (status.recording) acc.recording++;
      if (status.hasIssues) acc.issues++;
      return acc;
    },
    { total: 0, online: 0, recording: 0, issues: 0 }
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <StatusCard
        title="Connection Status"
        value={isConnected ? 'Connected' : 'Disconnected'}
        icon={isConnected ? Wifi : WifiOff}
        color={isConnected ? 'green' : 'red'}
        pulse={isConnected}
      />
      
      <StatusCard
        title="Active Cameras"
        value={`${cameraStats.online}/${cameraStats.total}`}
        icon={Camera}
        color={cameraStats.online > 0 ? 'green' : 'gray'}
        subtitle={`${cameraStats.recording} recording`}
      />
      
      <StatusCard
        title="System Health"
        value={cameraStats.issues === 0 ? 'Healthy' : 'Issues Detected'}
        icon={cameraStats.issues === 0 ? CheckCircle : AlertTriangle}
        color={cameraStats.issues === 0 ? 'green' : 'orange'}
        subtitle={cameraStats.issues > 0 ? `${cameraStats.issues} cameras` : 'All systems operational'}
      />
      
      <StatusCard
        title="Live Updates"
        value="Real-time"
        icon={Activity}
        color="blue"
        pulse={isConnected}
        subtitle="Event monitoring active"
      />
    </div>
  );
});

interface StatusCardProps {
  title: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
  color: 'green' | 'red' | 'orange' | 'blue' | 'gray';
  pulse?: boolean;
  subtitle?: string;
}

const StatusCard = memo<StatusCardProps>(({
  title,
  value,
  icon: Icon,
  color,
  pulse = false,
  subtitle,
}) => {
  const colorClasses = {
    green: 'text-green-600 bg-green-100',
    red: 'text-red-600 bg-red-100',
    orange: 'text-orange-600 bg-orange-100',
    blue: 'text-blue-600 bg-blue-100',
    gray: 'text-gray-600 bg-gray-100',
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-600">
          {title}
        </CardTitle>
        <div className={`p-2 rounded-full ${colorClasses[color]} ${pulse ? 'animate-pulse' : ''}`}>
          <Icon className="h-4 w-4" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold mb-1">{value}</div>
        {subtitle && (
          <p className="text-xs text-gray-500">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
});

LiveStatusIndicators.displayName = 'LiveStatusIndicators';
StatusCard.displayName = 'StatusCard';
```

### 4. Live Camera Feed Grid

```tsx
// components/live/LiveCameraGrid.tsx
'use client';

import { memo, useState, useEffect, useCallback } from 'react';
import { useCameraStatus } from '@/hooks/useLiveEvents';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Camera, 
  Play, 
  Pause, 
  Maximize2, 
  Volume2, 
  VolumeX,
  Wifi,
  WifiOff,
  Record,
  AlertCircle
} from 'lucide-react';

interface CameraData {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'maintenance';
  isRecording: boolean;
  hasAudio: boolean;
  resolution: string;
  fps: number;
  lastUpdate: string;
  streamUrl?: string;
}

export const LiveCameraGrid = memo<{ maxCameras?: number }>(({ 
  maxCameras = 6 
}) => {
  const { cameraStatuses, isConnected } = useCameraStatus();
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  const activeCameras = Object.entries(cameraStatuses)
    .filter(([_, status]) => status.online)
    .slice(0, maxCameras)
    .map(([id, status]) => ({
      id,
      name: status.name || `Camera ${id}`,
      location: status.location || 'Unknown Location',
      status: status.online ? 'online' : 'offline',
      isRecording: status.recording || false,
      hasAudio: status.audio || false,
      resolution: status.resolution || '1080p',
      fps: status.fps || 30,
      lastUpdate: status.lastUpdate,
      streamUrl: status.streamUrl,
    })) as CameraData[];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Live Camera Feeds</h3>
        <Badge variant={isConnected ? 'default' : 'destructive'}>
          {isConnected ? 'Live' : 'Offline'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {activeCameras.map(camera => (
          <CameraFeedCard
            key={camera.id}
            camera={camera}
            isSelected={selectedCamera === camera.id}
            onSelect={() => setSelectedCamera(camera.id)}
          />
        ))}
      </div>

      {activeCameras.length === 0 && (
        <div className="text-center py-12">
          <Camera className="h-12 w-12 mx-auto mb-4 text-gray-400" />
          <p className="text-gray-500">No active camera feeds available</p>
          <p className="text-sm text-gray-400">Cameras will appear here when online</p>
        </div>
      )}
    </div>
  );
});

const CameraFeedCard = memo<{
  camera: CameraData;
  isSelected: boolean;
  onSelect: () => void;
}>(({ camera, isSelected, onSelect }) => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(true);
  const [showControls, setShowControls] = useState(false);

  const togglePlay = useCallback(() => {
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const toggleMute = useCallback(() => {
    setIsMuted(!isMuted);
  }, [isMuted]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      case 'maintenance': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <Card 
      className={`cursor-pointer transition-all duration-200 ${
        isSelected ? 'ring-2 ring-blue-500' : ''
      }`}
      onClick={onSelect}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium truncate">
            {camera.name}
          </CardTitle>
          <div className="flex items-center gap-1">
            {camera.isRecording && (
              <Record className="h-3 w-3 text-red-500 animate-pulse" />
            )}
            <div className={`w-2 h-2 rounded-full ${getStatusColor(camera.status)}`} />
          </div>
        </div>
        <p className="text-xs text-gray-500 truncate">{camera.location}</p>
      </CardHeader>

      <CardContent 
        className="p-0"
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
          {/* Video Stream Placeholder */}
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
            {camera.status === 'online' ? (
              <div className="text-center text-white">
                <Camera className="h-8 w-8 mx-auto mb-2 opacity-75" />
                <p className="text-xs opacity-75">Live Stream</p>
                <p className="text-xs opacity-50">{camera.resolution} • {camera.fps}fps</p>
              </div>
            ) : (
              <div className="text-center text-gray-500">
                <WifiOff className="h-8 w-8 mx-auto mb-2" />
                <p className="text-xs">Camera Offline</p>
              </div>
            )}
          </div>

          {/* Live Indicator */}
          {camera.status === 'online' && (
            <div className="absolute top-2 left-2">
              <Badge className="bg-red-600 text-white text-xs animate-pulse">
                ● LIVE
              </Badge>
            </div>
          )}

          {/* Controls Overlay */}
          {showControls && camera.status === 'online' && (
            <div className="absolute inset-0 bg-black/30 flex items-center justify-center">
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    togglePlay();
                  }}
                  className="h-8 w-8 p-0 text-white hover:bg-white/20"
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>

                {camera.hasAudio && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleMute();
                    }}
                    className="h-8 w-8 p-0 text-white hover:bg-white/20"
                  >
                    {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                  </Button>
                )}

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    // Handle fullscreen
                  }}
                  className="h-8 w-8 p-0 text-white hover:bg-white/20"
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          {/* Error States */}
          {camera.status === 'maintenance' && (
            <div className="absolute inset-0 bg-yellow-500/20 flex items-center justify-center">
              <div className="text-center text-yellow-600">
                <AlertCircle className="h-8 w-8 mx-auto mb-2" />
                <p className="text-xs font-medium">Under Maintenance</p>
              </div>
            </div>
          )}
        </div>

        {/* Camera Info */}
        <div className="p-3 text-xs text-gray-500 space-y-1">
          <div className="flex justify-between">
            <span>Status:</span>
            <span className="capitalize">{camera.status}</span>
          </div>
          <div className="flex justify-between">
            <span>Last Update:</span>
            <span>{new Date(camera.lastUpdate).toLocaleTimeString()}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

LiveCameraGrid.displayName = 'LiveCameraGrid';
CameraFeedCard.displayName = 'CameraFeedCard';
```

### 5. Real-Time Analytics Dashboard

```tsx
// components/live/RealtimeAnalyticsDashboard.tsx
'use client';

import { memo, useState, useEffect } from 'react';
import { useLiveEvents } from '@/hooks/useLiveEvents';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  TrendingDown,
  Users,
  Car,
  Clock,
  AlertTriangle
} from 'lucide-react';

interface AnalyticsData {
  vehicleCount: number;
  averageSpeed: number;
  congestionLevel: 'low' | 'medium' | 'high';
  incidentCount: number;
  timestamp: string;
}

export const RealtimeAnalyticsDashboard = memo(() => {
  const { eventHistory, isConnected } = useLiveEvents('analytics_update');
  const [currentData, setCurrentData] = useState<AnalyticsData>({
    vehicleCount: 0,
    averageSpeed: 0,
    congestionLevel: 'low',
    incidentCount: 0,
    timestamp: new Date().toISOString(),
  });

  const [previousData, setPreviousData] = useState<AnalyticsData | null>(null);

  useEffect(() => {
    const latestAnalytics = eventHistory.find(event => 
      event.type === 'analytics_update'
    );

    if (latestAnalytics) {
      setPreviousData(currentData);
      setCurrentData({
        vehicleCount: latestAnalytics.data.vehicleCount || 0,
        averageSpeed: latestAnalytics.data.averageSpeed || 0,
        congestionLevel: latestAnalytics.data.congestionLevel || 'low',
        incidentCount: latestAnalytics.data.incidentCount || 0,
        timestamp: latestAnalytics.timestamp,
      });
    }
  }, [eventHistory, currentData]);

  const getChange = (current: number, previous: number | undefined) => {
    if (previous === undefined) return 0;
    return ((current - previous) / previous) * 100;
  };

  const getCongestionColor = (level: string) => {
    switch (level) {
      case 'high': return 'destructive';
      case 'medium': return 'warning';
      default: return 'secondary';
    }
  };

  const metrics = [
    {
      title: 'Active Vehicles',
      value: currentData.vehicleCount,
      change: getChange(currentData.vehicleCount, previousData?.vehicleCount),
      icon: Car,
      suffix: '',
    },
    {
      title: 'Average Speed',
      value: currentData.averageSpeed,
      change: getChange(currentData.averageSpeed, previousData?.averageSpeed),
      icon: TrendingUp,
      suffix: ' km/h',
    },
    {
      title: 'Active Incidents',
      value: currentData.incidentCount,
      change: getChange(currentData.incidentCount, previousData?.incidentCount),
      icon: AlertTriangle,
      suffix: '',
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Real-Time Analytics</h3>
        <div className="flex items-center gap-2">
          <Badge variant={isConnected ? 'default' : 'destructive'}>
            {isConnected ? 'Live' : 'Offline'}
          </Badge>
          <span className="text-xs text-gray-500">
            Updated: {new Date(currentData.timestamp).toLocaleTimeString()}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {metrics.map((metric, index) => (
          <AnalyticsMetricCard
            key={index}
            title={metric.title}
            value={metric.value}
            change={metric.change}
            icon={metric.icon}
            suffix={metric.suffix}
          />
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center justify-between">
            Traffic Congestion Level
            <Badge variant={getCongestionColor(currentData.congestionLevel)}>
              {currentData.congestionLevel.toUpperCase()}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1 bg-gray-200 rounded-full h-4">
              <div 
                className={`h-4 rounded-full transition-all duration-500 ${
                  currentData.congestionLevel === 'high' ? 'bg-red-500' :
                  currentData.congestionLevel === 'medium' ? 'bg-yellow-500' :
                  'bg-green-500'
                }`}
                style={{ 
                  width: `${
                    currentData.congestionLevel === 'high' ? 100 :
                    currentData.congestionLevel === 'medium' ? 60 : 30
                  }%` 
                }}
              />
            </div>
            <span className="text-sm font-medium min-w-[60px]">
              {currentData.congestionLevel === 'high' ? '80-100%' :
               currentData.congestionLevel === 'medium' ? '40-79%' : '0-39%'}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Based on vehicle density and average speed analysis
          </p>
        </CardContent>
      </Card>
    </div>
  );
});

const AnalyticsMetricCard = memo<{
  title: string;
  value: number;
  change: number;
  icon: React.ComponentType<{ className?: string }>;
  suffix: string;
}>(({ title, value, change, icon: Icon, suffix }) => {
  const isPositive = change > 0;
  const isNeutral = change === 0;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-600">
          {title}
        </CardTitle>
        <Icon className="h-4 w-4 text-gray-600" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold mb-1">
          {value}{suffix}
        </div>
        {!isNeutral && (
          <div className={`flex items-center gap-1 text-xs ${
            isPositive ? 'text-green-600' : 'text-red-600'
          }`}>
            {isPositive ? (
              <TrendingUp className="h-3 w-3" />
            ) : (
              <TrendingDown className="h-3 w-3" />
            )}
            <span>{Math.abs(change).toFixed(1)}%</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
});

RealtimeAnalyticsDashboard.displayName = 'RealtimeAnalyticsDashboard';
AnalyticsMetricCard.displayName = 'AnalyticsMetricCard';
```

## Integration Example

### Complete Live Dashboard Setup

```tsx
// app/dashboard/live/page.tsx
'use client';

import { Suspense } from 'react';
import { LiveEventFeed } from '@/components/live/LiveEventFeed';
import { RealtimeAlertBanner } from '@/components/live/RealtimeAlertBanner';
import { LiveStatusIndicators } from '@/components/live/LiveStatusIndicators';
import { LiveCameraGrid } from '@/components/live/LiveCameraGrid';
import { RealtimeAnalyticsDashboard } from '@/components/live/RealtimeAnalyticsDashboard';
import { Skeleton } from '@/components/ui/skeleton/Skeleton';

export default function LiveDashboardPage() {
  return (
    <div className="space-y-6">
      <RealtimeAlertBanner />
      
      <div className="pt-4"> {/* Space for alert banner */}
        <h1 className="text-2xl font-bold mb-6">Live Dashboard</h1>
        
        <Suspense fallback={<Skeleton height={120} />}>
          <LiveStatusIndicators />
        </Suspense>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <div className="lg:col-span-2 space-y-6">
            <Suspense fallback={<Skeleton height={400} />}>
              <LiveCameraGrid maxCameras={6} />
            </Suspense>
            
            <Suspense fallback={<Skeleton height={300} />}>
              <RealtimeAnalyticsDashboard />
            </Suspense>
          </div>

          <div className="space-y-6">
            <Suspense fallback={<Skeleton height={400} />}>
              <LiveEventFeed maxEvents={20} showFilters />
            </Suspense>
          </div>
        </div>
      </div>
    </div>
  );
}
```

## Best Practices

### 1. Performance Optimization
- Use React.memo for all live components
- Implement proper cleanup for SSE connections
- Debounce frequent updates
- Limit event history size
- Use virtual scrolling for large lists
- Leverage SSE's built-in reconnection handling

### 2. Error Handling
- Implement connection retry logic with exponential backoff
- Handle network failures gracefully with offline detection
- Use Last-Event-ID for resume functionality
- Provide clear error states to users
- Handle SSE connection limits (browser typically allows 6 per domain)

### 3. User Experience
- Show connection status clearly
- Implement progressive loading with skeleton components
- Use animations for state transitions
- Allow users to pause/resume updates
- Provide filtering and search capabilities
- Display connection health and last update times

### 4. Security Considerations
- Validate all incoming SSE messages
- Implement proper authentication for SSE endpoints
- Rate limit message processing
- Sanitize displayed data
- Use HTTPS for SSE connections in production
- Implement proper CORS headers on SSE endpoints

### 5. SSE-Specific Best Practices
- Use typed events for different data streams
- Implement proper Last-Event-ID handling for resume capability
- Set appropriate retry delays on the server side
- Use compression for SSE responses when possible
- Consider using multiple SSE connections for different data types to avoid browser connection limits

### 6. Server-Side Considerations
- Implement keep-alive messages to prevent connection timeout
- Use proper Content-Type: text/event-stream headers
- Handle client disconnections gracefully
- Implement event buffering for missed events
- Use efficient serialization for event data

This comprehensive SSE-based live event system provides reliable real-time monitoring capabilities for the ITS Camera AI dashboard, with automatic reconnection, resume functionality, and efficient server-to-client streaming.