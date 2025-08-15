'use client';

import React, { Suspense, useDeferredValue, useTransition, useState, useCallback } from 'react';
import {
  IconCamera,
  IconSearch,
  IconFilter,
  IconDownload,
  IconSettings,
  IconRefresh,
  IconTrendingUp,
  IconTrendingDown,
  IconAlertTriangle,
  IconVideo,
  IconPlayerRecord,
  IconPlayerStop,
  IconChartArea,
  IconActivity,
  IconEye
} from '@tabler/icons-react';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { useOptimisticCameraState, Camera as CameraType } from '@/hooks/useOptimisticCameraState';
import { useErrorRecovery } from '@/hooks/useErrorRecovery';
import { useRealTimeAnalytics } from '@/hooks/useRealTimeAnalytics';

// Mock initial camera data
const initialCameras: CameraType[] = [
  {
    id: 'cam-001',
    name: 'Highway Entrance',
    location: 'Main Gate A',
    status: 'online',
    lastSeen: new Date().toISOString(),
    resolution: '1920x1080',
    fps: 30,
    alerts: 12,
    isRecording: true,
  },
  {
    id: 'cam-002',
    name: 'Intersection Alpha',
    location: 'Junction B2',
    status: 'online',
    lastSeen: new Date(Date.now() - 2000).toISOString(),
    resolution: '2560x1440',
    fps: 60,
    alerts: 8,
    isRecording: true,
  },
  {
    id: 'cam-003',
    name: 'Parking Area',
    location: 'Zone C',
    status: 'maintenance',
    lastSeen: new Date(Date.now() - 300000).toISOString(),
    resolution: '1920x1080',
    fps: 30,
    alerts: 0,
    isRecording: false,
  },
  {
    id: 'cam-004',
    name: 'Exit Terminal',
    location: 'Exit D',
    status: 'offline',
    lastSeen: new Date(Date.now() - 600000).toISOString(),
    resolution: '1920x1080',
    fps: 30,
    alerts: 3,
    isRecording: false,
  },
];

// Enhanced Camera Card Component
const CameraCard: React.FC<{
  camera: CameraType;
  onStatusChange: (id: string, status: CameraType['status']) => void;
  onToggleRecording: (id: string) => void;
}> = React.memo(({ camera, onStatusChange, onToggleRecording }) => {
  const [isPending, startTransition] = useTransition();

  const getStatusColor = (status: CameraType['status']) => {
    switch (status) {
      case 'online': return 'bg-green-100 text-green-800 border-green-200';
      case 'offline': return 'bg-red-100 text-red-800 border-red-200';
      case 'maintenance': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusIcon = (status: CameraType['status']) => {
    switch (status) {
      case 'online': return '🟢';
      case 'offline': return '🔴';
      case 'maintenance': return '🟡';
      default: return '⚫';
    }
  };

  const handleStatusChange = (newStatus: CameraType['status']) => {
    startTransition(() => {
      onStatusChange(camera.id, newStatus);
    });
  };

  const handleToggleRecording = () => {
    startTransition(() => {
      onToggleRecording(camera.id);
    });
  };

  const timeSinceLastSeen = React.useMemo(() => {
    const diff = Date.now() - new Date(camera.lastSeen).getTime();
    if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    return `${Math.floor(diff / 3600000)}h ago`;
  }, [camera.lastSeen]);

  return (
    <div className={`border rounded-lg p-4 transition-all duration-300 hover:shadow-lg ${
      isPending ? 'opacity-70' : ''
    }`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <IconCamera className="w-5 h-5 text-orange-500" />
          <h3 className="font-semibold text-gray-900">{camera.name}</h3>
        </div>
        <div className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(camera.status)}`}>
          {getStatusIcon(camera.status)} {camera.status}
        </div>
      </div>

      <div className="space-y-2 text-sm text-gray-600 mb-4">
        <div className="flex justify-between">
          <span>Location:</span>
          <span className="font-medium">{camera.location}</span>
        </div>
        <div className="flex justify-between">
          <span>Resolution:</span>
          <span className="font-medium">{camera.resolution}</span>
        </div>
        <div className="flex justify-between">
          <span>FPS:</span>
          <span className="font-medium">{camera.fps}</span>
        </div>
        <div className="flex justify-between">
          <span>Alerts:</span>
          <span className={`font-medium ${camera.alerts > 0 ? 'text-red-600' : 'text-green-600'}`}>
            {camera.alerts}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Last Seen:</span>
          <span className="font-medium">{timeSinceLastSeen}</span>
        </div>
      </div>

      <div className="flex gap-2">
        <select
          value={camera.status}
          onChange={(e) => handleStatusChange(e.target.value as CameraType['status'])}
          className="flex-1 px-3 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isPending}
        >
          <option value="online">Online</option>
          <option value="offline">Offline</option>
          <option value="maintenance">Maintenance</option>
        </select>

        <button
          onClick={handleToggleRecording}
          disabled={isPending}
          className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
            camera.isRecording
              ? 'bg-red-100 text-red-700 hover:bg-red-200'
              : 'bg-green-100 text-green-700 hover:bg-green-200'
          } disabled:opacity-50`}
        >
          {camera.isRecording ? 'Stop' : 'Record'}
        </button>
      </div>
    </div>
  );
});

CameraCard.displayName = 'CameraCard';

// Analytics Chart Component (Simplified)
const AnalyticsChart: React.FC<{
  title: string;
  data: Array<{ x: string; y: number }>;
  color: string;
  trend?: { increasing: boolean; percentage: number };
}> = ({ title, data, color, trend }) => {
  const maxValue = Math.max(...data.map(d => d.y));
  const minValue = Math.min(...data.map(d => d.y));
  const range = maxValue - minValue || 1;

  return (
    <div className="bg-white p-4 rounded-lg border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-gray-900">{title}</h3>
        {trend && (
          <div className={`flex items-center gap-1 text-sm ${
            trend.increasing ? 'text-green-600' : 'text-red-600'
          }`}>
            {trend.increasing ? <IconTrendingUp className="w-4 h-4" /> : <IconTrendingDown className="w-4 h-4" />}
            {trend.percentage.toFixed(1)}%
          </div>
        )}
      </div>

      <div className="h-24 flex items-end gap-1">
        {data.slice(-20).map((point, index) => {
          const height = ((point.y - minValue) / range) * 100 || 5;
          return (
            <div
              key={index}
              className={`flex-1 ${color} rounded-sm transition-all duration-300`}
              style={{ height: `${height}%`, minHeight: '2px' }}
              title={`${new Date(point.x).toLocaleTimeString()}: ${point.y}`}
            />
          );
        })}
      </div>
    </div>
  );
};

// Main Camera Monitoring Page
const CamerasPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [isPending, startTransition] = useTransition();

  // Use our custom hooks
  const {
    cameras,
    cameraStats,
    updateCameraStatus,
    toggleRecording,
  } = useOptimisticCameraState(initialCameras);

  const { analytics, chartData, lastUpdate } = useRealTimeAnalytics({
    maxDataPoints: 50,
    updateInterval: refreshInterval,
  });

  const { executeWithRecovery, hasError, retryOperation, clearError } = useErrorRecovery();

  // Defer search results for better performance during typing
  const deferredSearchQuery = useDeferredValue(searchQuery);

  // Filter cameras based on search and status
  const filteredCameras = React.useMemo(() => {
    return cameras.filter(camera => {
      const matchesSearch = camera.name.toLowerCase().includes(deferredSearchQuery.toLowerCase()) ||
                           camera.location.toLowerCase().includes(deferredSearchQuery.toLowerCase());
      const matchesStatus = statusFilter === 'all' || camera.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [cameras, deferredSearchQuery, statusFilter]);

  // API simulation functions
  const simulateStatusUpdate = async (id: string, status: CameraType['status']) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    if (Math.random() < 0.1) throw new Error('Network error');
    console.log(`Updated camera ${id} status to ${status}`);
  };

  const simulateToggleRecording = async (id: string) => {
    await new Promise(resolve => setTimeout(resolve, 300));
    if (Math.random() < 0.05) throw new Error('Recording toggle failed');
    console.log(`Toggled recording for camera ${id}`);
  };

  const handleStatusChange = useCallback((id: string, status: CameraType['status']) => {
    updateCameraStatus(id, status, () => simulateStatusUpdate(id, status));
  }, [updateCameraStatus]);

  const handleToggleRecording = useCallback((id: string) => {
    toggleRecording(id, () => simulateToggleRecording(id));
  }, [toggleRecording]);

  const handleExport = useCallback(() => {
    startTransition(async () => {
      await executeWithRecovery(async () => {
        const data = {
          cameras: filteredCameras,
          stats: cameraStats,
          analytics,
          exportTime: new Date().toISOString(),
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `camera-report-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 'export');
    });
  }, [filteredCameras, cameraStats, analytics, executeWithRecovery]);

  return (
    <div className="min-h-screen bg-cyan-50">
      <div id="main-content">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-cyan-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-6 gap-4">
            <div className="min-w-0">
              <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 truncate">Camera Monitoring</h1>
              <p className="text-gray-600 mt-1 text-sm sm:text-base">
                Real-time monitoring and control of {cameraStats.total} cameras
                {lastUpdate && (
                  <span className="hidden sm:inline ml-2 text-sm">
                    Last updated: {lastUpdate.toLocaleTimeString()}
                  </span>
                )}
              </p>
            </div>
            <div className="flex gap-2 shrink-0">
              <button
                onClick={handleExport}
                disabled={isPending}
                className="flex items-center gap-2 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors disabled:opacity-50"
                aria-label="Export camera data and analytics report"
              >
                <IconDownload className="w-4 h-4" />
                Export
              </button>
              <button
                className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                aria-label="Open camera monitoring settings"
              >
                <IconSettings className="w-4 h-4" />
                Settings
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {hasError && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <IconAlertTriangle className="w-5 h-5 text-orange-peel" />
              <span className="text-red-800">An error occurred while processing your request</span>
            </div>
            <button
              onClick={clearError}
              className="text-red-600 hover:text-red-700 font-medium"
              aria-label="Dismiss error message"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Statistics Cards */}
        <section aria-label="Camera system statistics" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-8">
          <div className="bg-white p-4 sm:p-6 rounded-lg border" role="region" aria-labelledby="online-cameras-heading">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <h3 id="online-cameras-heading" className="text-sm text-gray-600">Online Cameras</h3>
                <p className="text-xl sm:text-2xl font-bold text-green-600" aria-label="{cameraStats.online} cameras currently online">{cameraStats.online}</p>
              </div>
              <div className="text-2xl sm:text-3xl shrink-0" aria-hidden="true">🟢</div>
            </div>
            <p className="text-xs text-gray-500 mt-2">{cameraStats.onlinePercentage}% operational</p>
          </div>

          <div className="bg-white p-4 sm:p-6 rounded-lg border" role="region" aria-labelledby="recording-cameras-heading">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <h3 id="recording-cameras-heading" className="text-sm text-gray-600">Recording</h3>
                <p className="text-xl sm:text-2xl font-bold text-blue-600" aria-label="{cameraStats.recording} cameras currently recording">{cameraStats.recording}</p>
              </div>
              <div className="text-2xl sm:text-3xl shrink-0" aria-hidden="true">🎥</div>
            </div>
            <p className="text-xs text-gray-500 mt-2">Active recordings</p>
          </div>

          <div className="bg-white p-4 sm:p-6 rounded-lg border" role="region" aria-labelledby="total-alerts-heading">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <h3 id="total-alerts-heading" className="text-sm text-gray-600">Total Alerts</h3>
                <p className="text-xl sm:text-2xl font-bold text-red-600" aria-label="{cameraStats.totalAlerts} total active alerts">{cameraStats.totalAlerts}</p>
              </div>
              <div className="text-2xl sm:text-3xl shrink-0" aria-hidden="true">🚨</div>
            </div>
            <p className="text-xs text-gray-500 mt-2">Across all cameras</p>
          </div>

          <div className="bg-white p-4 sm:p-6 rounded-lg border" role="region" aria-labelledby="peak-hour-heading">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <h3 id="peak-hour-heading" className="text-sm text-gray-600">Peak Hour</h3>
                <p className="text-xl sm:text-2xl font-bold text-purple-600" aria-label="Peak traffic hour at {analytics.peakHour.hour}:00">{analytics.peakHour.hour}:00</p>
              </div>
              <div className="text-2xl sm:text-3xl shrink-0" aria-hidden="true">📈</div>
            </div>
            <p className="text-xs text-gray-500 mt-2">{analytics.peakHour.count} vehicles</p>
          </div>
        </section>

        {/* Real-time Analytics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 mb-8">
          <AnalyticsChart
            title="Vehicle Count"
            data={chartData.vehicleCount}
            color="bg-blue-500"
            trend={analytics.flowTrends}
          />
          <AnalyticsChart
            title="Traffic Flow"
            data={chartData.trafficFlow}
            color="bg-green-500"
            trend={analytics.flowTrends}
          />
          <AnalyticsChart
            title="Alert Activity"
            data={chartData.alerts}
            color="bg-red-500"
            trend={analytics.alertTrends}
          />
        </div>

        {/* Filters and Search */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="flex-1 relative">
            <IconSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white-300 w-4 h-4" />
            <input
              type="text"
              placeholder="Search cameras..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm sm:text-base"
              aria-label="Search cameras by name or location"
              role="searchbox"
            />
          </div>

          <div className="flex flex-col sm:flex-row gap-2">
            <div className="relative min-w-0 flex-1 sm:flex-none">
              <IconFilter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white-300 w-4 h-4" />
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="w-full sm:w-auto pl-10 pr-8 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm sm:text-base"
                aria-label="Filter cameras by status"
              >
                <option value="all">All Status</option>
                <option value="online">Online</option>
                <option value="offline">Offline</option>
                <option value="maintenance">Maintenance</option>
              </select>
            </div>

            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="w-full sm:w-auto px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm sm:text-base"
              aria-label="Set refresh interval for camera data"
            >
              <option value={1000}>1s refresh</option>
              <option value={5000}>5s refresh</option>
              <option value={10000}>10s refresh</option>
              <option value={30000}>30s refresh</option>
            </select>
          </div>
        </div>

        {/* Camera Grid */}
        <ErrorBoundary level="component">
          <Suspense fallback={
            <div className="flex items-center justify-center py-12">
              <IconRefresh className="w-8 h-8 animate-spin text-orange-peel" />
              <span className="ml-2 text-gray-600">Loading cameras...</span>
            </div>
          }>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 sm:gap-6">
              {filteredCameras.map((camera) => (
                <CameraCard
                  key={camera.id}
                  camera={camera}
                  onStatusChange={handleStatusChange}
                  onToggleRecording={handleToggleRecording}
                />
              ))}
            </div>

            {filteredCameras.length === 0 && (
              <div className="text-center py-12">
                <IconCamera className="w-12 h-12 text-white-300 mx-auto mb-4" />
                <p className="text-gray-600">No cameras found matching your filters</p>
              </div>
            )}
          </Suspense>
        </ErrorBoundary>
      </div>
      </div>
    </div>
  );
};

export default function CamerasPageWrapper() {
  return (
    <ErrorBoundary level="page">
      <CamerasPage />
    </ErrorBoundary>
  );
}
