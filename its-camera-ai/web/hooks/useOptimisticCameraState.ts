'use client';

import { useOptimistic, useState, useCallback, useMemo, useEffect } from 'react';
import { camerasAPI, Camera as APICamera, CameraStatus, cameraUtils } from '@/lib/api/cameras';
import {
  AnalyticsWebSocketClient,
  createWebSocketClient,
  buildWebSocketURL,
  WS_ENDPOINTS,
  WebSocketMessage
} from '@/lib/websocket/client';

// Enhanced camera interface for UI
export interface Camera {
  id: string;
  name: string;
  description?: string;
  location: string;
  coordinates?: { lat: number; lng: number; altitude?: number };
  camera_type: string;
  stream_url: string;
  stream_protocol: string;
  backup_stream_url?: string;
  status: CameraStatus;
  config: any;
  health?: {
    is_healthy: boolean;
    status: string;
    last_checked: string;
    response_time_ms: number;
    error_message?: string;
  };
  zone_id?: string;
  tags: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_seen_at?: string;
  // UI-specific properties
  isRecording?: boolean;
  alerts?: number;
  fps?: number;
  resolution?: string;
}

interface CameraStats {
  total: number;
  online: number;
  offline: number;
  maintenance: number;
  streaming: number;
  stopped: number;
  recording: number;
  totalAlerts: number;
  onlinePercentage: string;
  healthyPercentage: string;
}

type OptimisticUpdate =
  | { type: 'updateStatus'; id: string; status: CameraStatus }
  | { type: 'toggleRecording'; id: string }
  | { type: 'updateHealth'; id: string; health: Camera['health'] }
  | { type: 'addCamera'; camera: Camera }
  | { type: 'removeCamera'; id: string }
  | { type: 'updateCamera'; id: string; updates: Partial<Camera> };

function applyOptimisticUpdate(cameras: Camera[], update: OptimisticUpdate): Camera[] {
  switch (update.type) {
    case 'updateStatus':
      return cameras.map(camera =>
        camera.id === update.id ? { ...camera, status: update.status } : camera
      );

    case 'toggleRecording':
      return cameras.map(camera =>
        camera.id === update.id ? { ...camera, isRecording: !camera.isRecording } : camera
      );

    case 'updateHealth':
      return cameras.map(camera =>
        camera.id === update.id ? { ...camera, health: update.health } : camera
      );

    case 'addCamera':
      return [...cameras, update.camera];

    case 'removeCamera':
      return cameras.filter(camera => camera.id !== update.id);

    case 'updateCamera':
      return cameras.map(camera =>
        camera.id === update.id ? { ...camera, ...update.updates } : camera
      );

    default:
      return cameras;
  }
}

// Convert API camera to UI camera format
function convertAPICamera(apiCamera: APICamera): Camera {
  return {
    ...apiCamera,
    // Extract UI-specific properties from config or health
    isRecording: apiCamera.config?.recording_enabled || false,
    alerts: 0, // Will be populated from real-time data
    fps: apiCamera.config?.quality_settings?.fps || 30,
    resolution: apiCamera.config?.quality_settings?.resolution || apiCamera.health?.resolution || '1920x1080',
  };
}

export function useOptimisticCameraState(options?: {
  autoFetch?: boolean;
  enableWebSocket?: boolean;
  pollingInterval?: number;
}) {
  const { autoFetch = true, enableWebSocket = true, pollingInterval = 30000 } = options || {};

  const [cameras, setCameras] = useState<Camera[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wsClient, setWsClient] = useState<AnalyticsWebSocketClient | null>(null);

  const [optimisticCameras, addOptimisticUpdate] = useOptimistic(
    cameras,
    applyOptimisticUpdate
  );

  // Enhanced camera stats calculation
  const cameraStats = useMemo((): CameraStats => {
    const stats = optimisticCameras.reduce(
      (acc, camera) => {
        acc.total += 1;

        // Count by status
        if (camera.status === 'online' || camera.status === 'streaming') {
          acc.online += 1;
        } else if (camera.status === 'offline' || camera.status === 'stopped') {
          acc.offline += 1;
        } else if (camera.status === 'maintenance') {
          acc.maintenance += 1;
        }

        if (camera.status === 'streaming') acc.streaming += 1;
        if (camera.status === 'stopped') acc.stopped += 1;
        if (camera.isRecording) acc.recording += 1;
        acc.totalAlerts += camera.alerts || 0;

        return acc;
      },
      {
        total: 0,
        online: 0,
        offline: 0,
        maintenance: 0,
        streaming: 0,
        stopped: 0,
        recording: 0,
        totalAlerts: 0
      }
    );

    const healthy = optimisticCameras.filter(c => c.health?.is_healthy).length;

    return {
      ...stats,
      onlinePercentage: stats.total > 0 ? ((stats.online / stats.total) * 100).toFixed(1) : '0',
      healthyPercentage: stats.total > 0 ? ((healthy / stats.total) * 100).toFixed(1) : '0',
    };
  }, [optimisticCameras]);

  // Fetch cameras from API
  const fetchCameras = useCallback(async (params?: {
    page?: number;
    size?: number;
    status?: CameraStatus;
    search?: string;
  }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await camerasAPI.list(params);
      const convertedCameras = response.items.map(convertAPICamera);
      setCameras(convertedCameras);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch cameras';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    if (autoFetch) {
      fetchCameras().catch(console.error);
    }
  }, [autoFetch, fetchCameras]);

  // Periodic refresh
  useEffect(() => {
    if (!autoFetch || pollingInterval <= 0) return;

    const interval = setInterval(() => {
      fetchCameras().catch(console.error);
    }, pollingInterval);

    return () => clearInterval(interval);
  }, [autoFetch, pollingInterval, fetchCameras]);

  // WebSocket integration
  useEffect(() => {
    if (!enableWebSocket) return;

    try {
      const client = createWebSocketClient({
        url: buildWebSocketURL(WS_ENDPOINTS.CAMERA_ANALYTICS),
        maxReconnectAttempts: 5,
        reconnectInterval: 1000,
        heartbeatInterval: 30000,
      });

      // Subscribe to camera status updates
      client.subscribe('camera_status', (message: WebSocketMessage) => {
        if (message.camera_id && message.data.status) {
          addOptimisticUpdate({
            type: 'updateStatus',
            id: message.camera_id,
            status: message.data.status as CameraStatus,
          });
        }
      });

      // Subscribe to health updates
      client.subscribe('system_health', (message: WebSocketMessage) => {
        if (message.camera_id && message.data.health) {
          addOptimisticUpdate({
            type: 'updateHealth',
            id: message.camera_id,
            health: message.data.health,
          });
        }
      });

      // Subscribe to metrics for alert counts
      client.subscribe('metrics', (message: WebSocketMessage) => {
        if (message.camera_id && message.data.alerts !== undefined) {
          addOptimisticUpdate({
            type: 'updateCamera',
            id: message.camera_id,
            updates: { alerts: message.data.alerts },
          });
        }
      });

      setWsClient(client);

      return () => {
        client.disconnect();
      };
    } catch (error) {
      console.error('Failed to setup WebSocket:', error);
    }
  }, [enableWebSocket, addOptimisticUpdate]);

  // Camera operations
  const updateCameraStatus = useCallback(async (
    id: string,
    status: CameraStatus
  ) => {
    // Apply optimistic update
    addOptimisticUpdate({ type: 'updateStatus', id, status });

    try {
      // Use the appropriate API call based on status
      if (status === 'streaming') {
        await camerasAPI.controlStream(id, 'start');
      } else if (status === 'stopped') {
        await camerasAPI.controlStream(id, 'stop');
      } else {
        // For other status changes, update the camera directly
        await camerasAPI.update(id, { status });
      }

      // Refresh camera data to get updated state
      const updatedCamera = await camerasAPI.getById(id);
      const convertedCamera = convertAPICamera(updatedCamera);

      setCameras(prev => prev.map(camera =>
        camera.id === id ? convertedCamera : camera
      ));
    } catch (error) {
      // Optimistic update will be reverted automatically
      console.error('Failed to update camera status:', error);
      throw error;
    }
  }, [addOptimisticUpdate]);

  const toggleRecording = useCallback(async (id: string) => {
    // Apply optimistic update
    addOptimisticUpdate({ type: 'toggleRecording', id });

    try {
      // Get current camera state to determine action
      const camera = cameras.find(c => c.id === id);
      const action = camera?.isRecording ? 'stop' : 'start';

      // Update camera config to toggle recording
      await camerasAPI.update(id, {
        config: {
          ...camera?.config,
          recording_enabled: !camera?.isRecording,
        },
      });

      // Also control the stream if needed
      if (action === 'start' && camera?.status !== 'streaming') {
        await camerasAPI.controlStream(id, 'start');
      }

      // Update actual state
      setCameras(prev => prev.map(camera =>
        camera.id === id ? {
          ...camera,
          isRecording: !camera.isRecording,
          status: action === 'start' ? 'streaming' : camera.status,
        } : camera
      ));
    } catch (error) {
      console.error('Failed to toggle recording:', error);
      throw error;
    }
  }, [addOptimisticUpdate, cameras]);

  const createCamera = useCallback(async (cameraData: any) => {
    try {
      const newCamera = await camerasAPI.create(cameraData);
      const convertedCamera = convertAPICamera(newCamera);

      addOptimisticUpdate({ type: 'addCamera', camera: convertedCamera });
      setCameras(prev => [...prev, convertedCamera]);

      return convertedCamera;
    } catch (error) {
      console.error('Failed to create camera:', error);
      throw error;
    }
  }, [addOptimisticUpdate]);

  const deleteCamera = useCallback(async (id: string) => {
    addOptimisticUpdate({ type: 'removeCamera', id });

    try {
      await camerasAPI.delete(id);
      setCameras(prev => prev.filter(camera => camera.id !== id));
    } catch (error) {
      console.error('Failed to delete camera:', error);
      // Revert optimistic update by refetching
      fetchCameras().catch(console.error);
      throw error;
    }
  }, [addOptimisticUpdate, fetchCameras]);

  const refreshCamera = useCallback(async (id: string) => {
    try {
      const updatedCamera = await camerasAPI.getById(id);
      const convertedCamera = convertAPICamera(updatedCamera);

      setCameras(prev => prev.map(camera =>
        camera.id === id ? convertedCamera : camera
      ));

      return convertedCamera;
    } catch (error) {
      console.error('Failed to refresh camera:', error);
      throw error;
    }
  }, []);

  const getCameraStats = useCallback(async (id: string, days: number = 7) => {
    try {
      return await camerasAPI.getStats(id, days);
    } catch (error) {
      console.error('Failed to get camera stats:', error);
      throw error;
    }
  }, []);

  return {
    cameras: optimisticCameras,
    cameraStats,
    isLoading,
    error,
    wsClient,

    // Actions
    fetchCameras,
    updateCameraStatus,
    toggleRecording,
    createCamera,
    deleteCamera,
    refreshCamera,
    getCameraStats,

    // Utilities
    clearError: () => setError(null),
  };
}
