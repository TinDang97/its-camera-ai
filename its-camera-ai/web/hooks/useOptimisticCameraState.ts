'use client';

import { useOptimistic, useState, useCallback, useMemo } from 'react';

export interface Camera {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'maintenance';
  lastSeen: string;
  resolution: string;
  fps: number;
  alerts: number;
  isRecording: boolean;
}

interface CameraStats {
  total: number;
  online: number;
  offline: number;
  maintenance: number;
  recording: number;
  totalAlerts: number;
  onlinePercentage: string;
}

type OptimisticUpdate =
  | { type: 'updateStatus'; id: string; status: Camera['status'] }
  | { type: 'toggleRecording'; id: string };

function applyOptimisticUpdate(cameras: Camera[], update: OptimisticUpdate): Camera[] {
  return cameras.map(camera => {
    if (camera.id !== update.id) return camera;

    switch (update.type) {
      case 'updateStatus':
        return { ...camera, status: update.status };
      case 'toggleRecording':
        return { ...camera, isRecording: !camera.isRecording };
      default:
        return camera;
    }
  });
}

export function useOptimisticCameraState(initialCameras: Camera[]) {
  const [cameras, setCameras] = useState<Camera[]>(initialCameras);
  const [optimisticCameras, addOptimisticUpdate] = useOptimistic(
    cameras,
    applyOptimisticUpdate
  );

  const cameraStats = useMemo((): CameraStats => {
    const stats = optimisticCameras.reduce(
      (acc, camera) => {
        acc.total += 1;
        acc[camera.status] += 1;
        if (camera.isRecording) acc.recording += 1;
        acc.totalAlerts += camera.alerts;
        return acc;
      },
      { total: 0, online: 0, offline: 0, maintenance: 0, recording: 0, totalAlerts: 0 }
    );

    return {
      ...stats,
      onlinePercentage: ((stats.online / stats.total) * 100).toFixed(1),
    };
  }, [optimisticCameras]);

  const updateCameraStatus = useCallback(async (
    id: string,
    status: Camera['status'],
    apiCall: () => Promise<void>
  ) => {
    // Apply optimistic update
    addOptimisticUpdate({ type: 'updateStatus', id, status });

    try {
      // Execute API call
      await apiCall();

      // Update actual state on success
      setCameras(prev => prev.map(camera =>
        camera.id === id ? { ...camera, status } : camera
      ));
    } catch (error) {
      // Optimistic update will be reverted automatically
      console.error('Failed to update camera status:', error);
      throw error;
    }
  }, [addOptimisticUpdate]);

  const toggleRecording = useCallback(async (
    id: string,
    apiCall: () => Promise<void>
  ) => {
    // Apply optimistic update
    addOptimisticUpdate({ type: 'toggleRecording', id });

    try {
      // Execute API call
      await apiCall();

      // Update actual state on success
      setCameras(prev => prev.map(camera =>
        camera.id === id ? { ...camera, isRecording: !camera.isRecording } : camera
      ));
    } catch (error) {
      // Optimistic update will be reverted automatically
      console.error('Failed to toggle recording:', error);
      throw error;
    }
  }, [addOptimisticUpdate]);

  return {
    cameras: optimisticCameras,
    cameraStats,
    updateCameraStatus,
    toggleRecording,
  };
}
