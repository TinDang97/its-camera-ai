/**
 * Analytics State Management with Zustand
 *
 * Centralized store for managing real-time analytics data, metrics processing,
 * and dashboard state for the ITS Camera AI Dashboard.
 */

import { create } from 'zustand';
import { subscribeWithSelector, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { WebSocketMessage, EventType } from '@/lib/websocket/client';

// Analytics data interfaces
export interface MetricData {
  timestamp: number;
  value: number;
  cameraId?: string;
  metadata?: Record<string, any>;
}

export interface TrafficMetrics {
  vehicleCount: number;
  avgSpeed: number;
  trafficFlow: number;
  congestionLevel: 'free' | 'moderate' | 'heavy' | 'congested';
  timestamp: number;
}

export interface IncidentData {
  id: string;
  type: 'accident' | 'congestion' | 'roadwork' | 'weather' | 'other';
  severity: 'low' | 'medium' | 'high' | 'critical';
  cameraId: string;
  location: string;
  description: string;
  timestamp: number;
  resolved: boolean;
  resolvedAt?: number;
}

export interface PredictionData {
  type: 'traffic_flow' | 'congestion' | 'incident_risk';
  prediction: number;
  confidence: number;
  timeHorizon: number; // minutes
  cameraId: string;
  timestamp: number;
}

export interface CameraHealthMetrics {
  cameraId: string;
  isOnline: boolean;
  responseTime: number;
  frameRate: number;
  quality: 'excellent' | 'good' | 'poor' | 'critical';
  lastUpdate: number;
  errorCount: number;
}

export interface SystemHealthMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
  activeConnections: number;
  processingQueue: number;
  timestamp: number;
}

// Time series data for charts
export interface TimeSeriesData {
  timestamp: number;
  values: Record<string, number>;
}

// Analytics aggregations
export interface AnalyticsAggregation {
  hourly: TimeSeriesData[];
  daily: TimeSeriesData[];
  weekly: TimeSeriesData[];
  monthly: TimeSeriesData[];
}

// Analytics store state
export interface AnalyticsState {
  // Real-time metrics
  currentMetrics: {
    traffic: TrafficMetrics | null;
    systemHealth: SystemHealthMetrics | null;
    totalVehicles: number;
    totalIncidents: number;
    avgProcessingTime: number;
    lastUpdate: number;
  };

  // Time series data
  timeSeries: {
    vehicleCount: MetricData[];
    trafficFlow: MetricData[];
    speed: MetricData[];
    incidents: MetricData[];
    systemMetrics: MetricData[];
  };

  // Incidents and alerts
  incidents: IncidentData[];
  activeIncidents: IncidentData[];
  resolvedIncidents: IncidentData[];

  // Predictions
  predictions: PredictionData[];
  latestPredictions: Record<string, PredictionData>; // by camera ID

  // Camera health
  cameraHealth: Record<string, CameraHealthMetrics>;

  // Aggregated data for charts
  aggregations: {
    hourly: AnalyticsAggregation;
    daily: AnalyticsAggregation;
    weekly: AnalyticsAggregation;
  };

  // Dashboard settings
  settings: {
    autoRefresh: boolean;
    refreshInterval: number;
    maxDataPoints: number;
    alertThresholds: {
      highTraffic: number;
      lowFrameRate: number;
      highResponseTime: number;
      criticalIncidents: number;
    };
  };

  // Processing state
  isProcessing: boolean;
  lastProcessedMessage: number;
  processingStats: {
    messagesProcessed: number;
    processingErrors: number;
    averageProcessingTime: number;
  };

  // Actions
  processMessage: (message: WebSocketMessage) => void;
  addIncident: (incident: Omit<IncidentData, 'id' | 'timestamp'>) => void;
  resolveIncident: (incidentId: string) => void;
  updateCameraHealth: (cameraId: string, health: Partial<CameraHealthMetrics>) => void;
  clearOldData: (maxAge?: number) => void;
  exportData: (timeRange: 'hour' | 'day' | 'week' | 'month') => any;
  updateSettings: (settings: Partial<AnalyticsState['settings']>) => void;

  // Data retrieval helpers
  getMetricHistory: (metric: keyof AnalyticsState['timeSeries'], cameraId?: string, limit?: number) => MetricData[];
  getAggregatedData: (metric: string, period: 'hourly' | 'daily' | 'weekly') => TimeSeriesData[];
  getIncidentsByTimeRange: (startTime: number, endTime: number) => IncidentData[];
  getCameraHealthSummary: () => {
    online: number;
    offline: number;
    total: number;
    averageResponseTime: number;
    averageFrameRate: number;
  };
  getSystemHealthTrend: () => 'improving' | 'stable' | 'degrading';
}

// Helper function to calculate congestion level
const calculateCongestionLevel = (vehicleCount: number, speed: number): TrafficMetrics['congestionLevel'] => {
  if (speed > 50 && vehicleCount < 20) return 'free';
  if (speed > 30 && vehicleCount < 50) return 'moderate';
  if (speed > 15 && vehicleCount < 80) return 'heavy';
  return 'congested';
};

// Helper function to create time series data point
const createMetricData = (value: number, cameraId?: string, metadata?: Record<string, any>): MetricData => ({
  timestamp: Date.now(),
  value,
  cameraId,
  metadata,
});

// Default settings
const defaultSettings: AnalyticsState['settings'] = {
  autoRefresh: true,
  refreshInterval: 5000,
  maxDataPoints: 1000,
  alertThresholds: {
    highTraffic: 80,
    lowFrameRate: 15,
    highResponseTime: 1000,
    criticalIncidents: 5,
  },
};

// Analytics store implementation
export const useAnalyticsStore = create<AnalyticsState>()(
  subscribeWithSelector(
    persist(
      immer((set, get) => ({
        // Initial state
        currentMetrics: {
          traffic: null,
          systemHealth: null,
          totalVehicles: 0,
          totalIncidents: 0,
          avgProcessingTime: 0,
          lastUpdate: 0,
        },

        timeSeries: {
          vehicleCount: [],
          trafficFlow: [],
          speed: [],
          incidents: [],
          systemMetrics: [],
        },

        incidents: [],
        activeIncidents: [],
        resolvedIncidents: [],

        predictions: [],
        latestPredictions: {},

        cameraHealth: {},

        aggregations: {
          hourly: { hourly: [], daily: [], weekly: [], monthly: [] },
          daily: { hourly: [], daily: [], weekly: [], monthly: [] },
          weekly: { hourly: [], daily: [], weekly: [], monthly: [] },
        },

        settings: defaultSettings,

        isProcessing: false,
        lastProcessedMessage: 0,
        processingStats: {
          messagesProcessed: 0,
          processingErrors: 0,
          averageProcessingTime: 0,
        },

        // Actions
        processMessage: (message: WebSocketMessage) => {
          const startTime = performance.now();

          set(draft => {
            draft.isProcessing = true;
            draft.lastProcessedMessage = Date.now();
          });

          try {
            const state = get();

            set(draft => {
              switch (message.event_type) {
                case 'vehicle_count':
                  {
                    const count = message.data.count || 0;
                    const speed = message.data.average_speed || 0;

                    // Update current metrics
                    draft.currentMetrics.totalVehicles = count;
                    draft.currentMetrics.lastUpdate = Date.now();

                    // Update traffic metrics
                    draft.currentMetrics.traffic = {
                      vehicleCount: count,
                      avgSpeed: speed,
                      trafficFlow: count * speed * 0.01, // Simple flow calculation
                      congestionLevel: calculateCongestionLevel(count, speed),
                      timestamp: Date.now(),
                    };

                    // Add to time series
                    draft.timeSeries.vehicleCount.push(createMetricData(count, message.camera_id));
                    draft.timeSeries.speed.push(createMetricData(speed, message.camera_id));
                    draft.timeSeries.trafficFlow.push(createMetricData(count * speed * 0.01, message.camera_id));

                    // Maintain data limits
                    Object.keys(draft.timeSeries).forEach(key => {
                      const series = draft.timeSeries[key as keyof typeof draft.timeSeries];
                      if (series.length > state.settings.maxDataPoints) {
                        series.splice(0, series.length - state.settings.maxDataPoints);
                      }
                    });
                  }
                  break;

                case 'incident':
                  {
                    const incident: IncidentData = {
                      id: `incident_${Date.now()}_${Math.random().toString(36).slice(2)}`,
                      type: message.data.type || 'other',
                      severity: message.data.severity || 'medium',
                      cameraId: message.camera_id || 'unknown',
                      location: message.data.location || 'Unknown location',
                      description: message.data.description || 'No description',
                      timestamp: Date.now(),
                      resolved: false,
                    };

                    draft.incidents.push(incident);
                    draft.activeIncidents.push(incident);
                    draft.currentMetrics.totalIncidents++;

                    // Add to time series
                    draft.timeSeries.incidents.push(createMetricData(1, message.camera_id, {
                      severity: incident.severity,
                      type: incident.type
                    }));
                  }
                  break;

                case 'prediction':
                  {
                    const prediction: PredictionData = {
                      type: message.data.prediction_type || 'traffic_flow',
                      prediction: message.data.prediction || 0,
                      confidence: message.confidence_score || 0.5,
                      timeHorizon: message.data.time_horizon || 30,
                      cameraId: message.camera_id || 'unknown',
                      timestamp: Date.now(),
                    };

                    draft.predictions.push(prediction);

                    // Update latest predictions by camera
                    if (message.camera_id) {
                      draft.latestPredictions[message.camera_id] = prediction;
                    }

                    // Keep only recent predictions
                    if (draft.predictions.length > 100) {
                      draft.predictions = draft.predictions.slice(-100);
                    }
                  }
                  break;

                case 'camera_status':
                  {
                    const cameraId = message.camera_id || 'unknown';
                    const isOnline = message.data.status === 'online';

                    if (!draft.cameraHealth[cameraId]) {
                      draft.cameraHealth[cameraId] = {
                        cameraId,
                        isOnline,
                        responseTime: 0,
                        frameRate: 0,
                        quality: 'good',
                        lastUpdate: Date.now(),
                        errorCount: 0,
                      };
                    }

                    const health = draft.cameraHealth[cameraId];
                    health.isOnline = isOnline;
                    health.lastUpdate = Date.now();
                    health.responseTime = message.data.response_time || health.responseTime;
                    health.frameRate = message.data.frame_rate || health.frameRate;

                    // Calculate quality based on metrics
                    if (health.responseTime < 100 && health.frameRate > 25) {
                      health.quality = 'excellent';
                    } else if (health.responseTime < 500 && health.frameRate > 15) {
                      health.quality = 'good';
                    } else if (health.responseTime < 1000 && health.frameRate > 10) {
                      health.quality = 'poor';
                    } else {
                      health.quality = 'critical';
                    }
                  }
                  break;

                case 'system_health':
                  {
                    const systemHealth: SystemHealthMetrics = {
                      cpuUsage: message.data.cpu_usage || 0,
                      memoryUsage: message.data.memory_usage || 0,
                      diskUsage: message.data.disk_usage || 0,
                      networkLatency: message.data.network_latency || 0,
                      activeConnections: message.data.active_connections || 0,
                      processingQueue: message.data.processing_queue || 0,
                      timestamp: Date.now(),
                    };

                    draft.currentMetrics.systemHealth = systemHealth;

                    // Add to time series
                    draft.timeSeries.systemMetrics.push(createMetricData(
                      systemHealth.cpuUsage,
                      'system',
                      {
                        memory: systemHealth.memoryUsage,
                        disk: systemHealth.diskUsage,
                        latency: systemHealth.networkLatency
                      }
                    ));
                  }
                  break;

                case 'metrics':
                  {
                    // Process general metrics
                    if (message.processing_latency_ms) {
                      const processingTime = message.processing_latency_ms;
                      draft.currentMetrics.avgProcessingTime = (
                        draft.currentMetrics.avgProcessingTime * 0.9 + processingTime * 0.1
                      );
                    }
                  }
                  break;
              }

              // Update processing stats
              const processingTime = performance.now() - startTime;
              draft.processingStats.messagesProcessed++;
              draft.processingStats.averageProcessingTime = (
                draft.processingStats.averageProcessingTime * 0.9 + processingTime * 0.1
              );

              draft.isProcessing = false;
            });

          } catch (error) {
            console.error('Error processing analytics message:', error);

            set(draft => {
              draft.processingStats.processingErrors++;
              draft.isProcessing = false;
            });
          }
        },

        addIncident: (incidentData) => {
          set(draft => {
            const incident: IncidentData = {
              ...incidentData,
              id: `manual_${Date.now()}_${Math.random().toString(36).slice(2)}`,
              timestamp: Date.now(),
            };

            draft.incidents.push(incident);
            if (!incident.resolved) {
              draft.activeIncidents.push(incident);
            }
            draft.currentMetrics.totalIncidents++;
          });
        },

        resolveIncident: (incidentId: string) => {
          set(draft => {
            const incidentIndex = draft.incidents.findIndex(i => i.id === incidentId);
            if (incidentIndex !== -1) {
              draft.incidents[incidentIndex].resolved = true;
              draft.incidents[incidentIndex].resolvedAt = Date.now();

              // Move to resolved incidents
              const incident = draft.incidents[incidentIndex];
              draft.resolvedIncidents.push(incident);

              // Remove from active incidents
              const activeIndex = draft.activeIncidents.findIndex(i => i.id === incidentId);
              if (activeIndex !== -1) {
                draft.activeIncidents.splice(activeIndex, 1);
              }
            }
          });
        },

        updateCameraHealth: (cameraId: string, healthUpdate: Partial<CameraHealthMetrics>) => {
          set(draft => {
            if (!draft.cameraHealth[cameraId]) {
              draft.cameraHealth[cameraId] = {
                cameraId,
                isOnline: false,
                responseTime: 0,
                frameRate: 0,
                quality: 'poor',
                lastUpdate: Date.now(),
                errorCount: 0,
              };
            }

            Object.assign(draft.cameraHealth[cameraId], healthUpdate, {
              lastUpdate: Date.now()
            });
          });
        },

        clearOldData: (maxAge = 24 * 60 * 60 * 1000) => { // Default 24 hours
          const cutoff = Date.now() - maxAge;

          set(draft => {
            // Clean time series data
            Object.keys(draft.timeSeries).forEach(key => {
              const series = draft.timeSeries[key as keyof typeof draft.timeSeries];
              const filtered = series.filter(data => data.timestamp > cutoff);
              draft.timeSeries[key as keyof typeof draft.timeSeries] = filtered as any;
            });

            // Clean incidents
            draft.incidents = draft.incidents.filter(incident => incident.timestamp > cutoff);
            draft.resolvedIncidents = draft.resolvedIncidents.filter(incident => incident.timestamp > cutoff);

            // Clean predictions
            draft.predictions = draft.predictions.filter(prediction => prediction.timestamp > cutoff);
          });
        },

        exportData: (timeRange: 'hour' | 'day' | 'week' | 'month') => {
          const state = get();
          const now = Date.now();
          const timeRangeMs = {
            hour: 60 * 60 * 1000,
            day: 24 * 60 * 60 * 1000,
            week: 7 * 24 * 60 * 60 * 1000,
            month: 30 * 24 * 60 * 60 * 1000,
          }[timeRange];

          const cutoff = now - timeRangeMs;

          return {
            timeRange,
            exportTime: now,
            currentMetrics: state.currentMetrics,
            timeSeries: Object.fromEntries(
              Object.entries(state.timeSeries).map(([key, data]) => [
                key,
                data.filter(item => item.timestamp > cutoff)
              ])
            ),
            incidents: state.incidents.filter(incident => incident.timestamp > cutoff),
            predictions: state.predictions.filter(prediction => prediction.timestamp > cutoff),
            cameraHealth: state.cameraHealth,
            processingStats: state.processingStats,
          };
        },

        updateSettings: (settingsUpdate) => {
          set(draft => {
            Object.assign(draft.settings, settingsUpdate);
          });
        },

        // Helper methods
        getMetricHistory: (metric, cameraId, limit) => {
          const data = get().timeSeries[metric];
          let filtered = cameraId ? data.filter(d => d.cameraId === cameraId) : data;
          return limit ? filtered.slice(-limit) : filtered;
        },

        getAggregatedData: (metric, period) => {
          return get().aggregations[period][period] || [];
        },

        getIncidentsByTimeRange: (startTime, endTime) => {
          return get().incidents.filter(
            incident => incident.timestamp >= startTime && incident.timestamp <= endTime
          );
        },

        getCameraHealthSummary: () => {
          const cameras = Object.values(get().cameraHealth);
          const online = cameras.filter(c => c.isOnline).length;
          const total = cameras.length;

          return {
            online,
            offline: total - online,
            total,
            averageResponseTime: total > 0 ?
              cameras.reduce((sum, c) => sum + c.responseTime, 0) / total : 0,
            averageFrameRate: total > 0 ?
              cameras.reduce((sum, c) => sum + c.frameRate, 0) / total : 0,
          };
        },

        getSystemHealthTrend: () => {
          const metrics = get().timeSeries.systemMetrics.slice(-10);
          if (metrics.length < 2) return 'stable';

          const recent = metrics.slice(-3);
          const older = metrics.slice(-6, -3);

          const recentAvg = recent.reduce((sum, m) => sum + m.value, 0) / recent.length;
          const olderAvg = older.reduce((sum, m) => sum + m.value, 0) / older.length;

          if (recentAvg < olderAvg * 0.95) return 'improving';
          if (recentAvg > olderAvg * 1.05) return 'degrading';
          return 'stable';
        },
      })),
      {
        name: 'analytics-store',
        partialize: (state) => ({
          // Only persist settings and aggregated data
          settings: state.settings,
          processingStats: state.processingStats,
        }),
      }
    )
  )
);

// Selector hooks for optimized re-renders
export const useCurrentMetrics = () =>
  useAnalyticsStore(state => state.currentMetrics);

export const useTrafficMetrics = () =>
  useAnalyticsStore(state => state.currentMetrics.traffic);

export const useSystemHealth = () =>
  useAnalyticsStore(state => state.currentMetrics.systemHealth);

export const useActiveIncidents = () =>
  useAnalyticsStore(state => state.activeIncidents);

export const useTimeSeriesData = (metric: keyof AnalyticsState['timeSeries'], limit?: number) =>
  useAnalyticsStore(state => state.getMetricHistory(metric, undefined, limit));

export const useCameraHealth = () =>
  useAnalyticsStore(state => ({
    cameras: state.cameraHealth,
    summary: state.getCameraHealthSummary(),
  }));

export const useProcessingStats = () =>
  useAnalyticsStore(state => ({
    stats: state.processingStats,
    isProcessing: state.isProcessing,
    lastProcessed: state.lastProcessedMessage,
  }));

export default useAnalyticsStore;
