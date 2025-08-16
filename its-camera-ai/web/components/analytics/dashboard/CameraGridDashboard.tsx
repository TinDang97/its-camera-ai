/**
 * Camera Grid Dashboard Component
 *
 * Grid view of all cameras with live status indicators, performance metrics,
 * and live preview thumbnails for comprehensive camera monitoring.
 */

'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { useDeferredValue, useTransition } from 'react';
import { cn } from '@/lib/utils';
import {
  IconCamera,
  IconCameraOff,
  IconSignal,
  IconAlertTriangle,
  IconCheckCircle,
  IconSettings,
  IconMaximize2,
  IconRefresh,
  IconGrid3x3,
  IconList,
  IconFilter,
  IconDownload,
  IconZoomIn,
  IconPlay,
  IconPause
} from '@tabler/icons-react';
import { MetricCard, MetricCardData } from './MetricCard';
import { DashboardWidget, WidgetConfig } from './DashboardWidget';
import { useAnalyticsStore } from '@/stores/analytics';
import { useRealTimeMetrics } from '@/providers/RealTimeProvider';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  ToggleGroup,
  ToggleGroupItem,
} from '@/components/ui/toggle-group';

export interface CameraInfo {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'maintenance' | 'error';
  streamUrl?: string;
  thumbnailUrl?: string;
  coordinates: { lat: number; lng: number };
  installation: {
    installedAt: number;
    lastMaintenance?: number;
    nextMaintenance?: number;
  };
  performance: {
    uptime: number; // percentage
    fps: number;
    resolution: string;
    bandwidth: number; // Mbps
    latency: number; // ms
    errorRate: number; // percentage
  };
  currentMetrics: {
    vehicleCount: number;
    averageSpeed: number;
    congestionLevel: 'free' | 'light' | 'moderate' | 'heavy' | 'severe';
    lastUpdate: number;
  };
  settings: {
    recordingEnabled: boolean;
    alertsEnabled: boolean;
    qualityProfile: 'low' | 'medium' | 'high' | 'ultra';
    nightVisionEnabled: boolean;
  };
  health: {
    temperature: number; // Celsius
    storage: {
      used: number; // GB
      total: number; // GB
    };
    network: {
      signalStrength: number; // percentage
      packetLoss: number; // percentage
    };
  };
}

export interface CameraGridData {
  cameras: CameraInfo[];
  totalCameras: number;
  onlineCameras: number;
  offlineCameras: number;
  maintenanceCameras: number;
  overallUptime: number;
  averageFps: number;
  totalBandwidth: number;
  alertsCount: number;
}

export interface CameraGridDashboardProps {
  data?: CameraGridData;
  className?: string;
  showControls?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  viewMode?: 'grid' | 'list';
  gridSize?: 'small' | 'medium' | 'large';
  onCameraSelect?: (camera: CameraInfo) => void;
  onCameraSettings?: (cameraId: string) => void;
  onCameraStream?: (cameraId: string, action: 'play' | 'pause' | 'fullscreen') => void;
  onDataExport?: () => void;
}

// Camera status configurations
const statusConfig = {
  online: { color: 'text-success', bgColor: 'bg-success/20', label: 'Online', icon: IconCheckCircle },
  offline: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Offline', icon: IconCameraOff },
  maintenance: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Maintenance', icon: IconSettings },
  error: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Error', icon: IconAlertTriangle },
};

// Congestion level configurations
const congestionConfig = {
  free: { color: 'text-success', label: 'Free Flow' },
  light: { color: 'text-success', label: 'Light' },
  moderate: { color: 'text-warning', label: 'Moderate' },
  heavy: { color: 'text-warning', label: 'Heavy' },
  severe: { color: 'text-destructive', label: 'Severe' },
};

// Generate mock camera data
const generateMockCameraData = (): CameraGridData => {
  const cameras: CameraInfo[] = Array.from({ length: 12 }, (_, i) => ({
    id: `CAM-${String(i + 1).padStart(3, '0')}`,
    name: `Camera ${i + 1}`,
    location: [
      'Main St & 1st Ave',
      'Highway 101 North',
      'Downtown Plaza',
      'Industrial Blvd',
      'Airport Road',
      'City Center',
      'University Ave',
      'Shopping District',
      'Residential Area',
      'Business Park',
      'Transit Hub',
      'Emergency Route'
    ][i],
    status: ['online', 'offline', 'maintenance', 'error'][Math.floor(Math.random() * 4)] as any,
    streamUrl: `rtmp://stream.example.com/cam${i + 1}`,
    thumbnailUrl: `/api/cameras/cam${i + 1}/thumbnail`,
    coordinates: {
      lat: 37.7749 + (Math.random() - 0.5) * 0.1,
      lng: -122.4194 + (Math.random() - 0.5) * 0.1,
    },
    installation: {
      installedAt: Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000,
      lastMaintenance: Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000,
      nextMaintenance: Date.now() + Math.random() * 90 * 24 * 60 * 60 * 1000,
    },
    performance: {
      uptime: Math.random() * 20 + 80, // 80-100%
      fps: Math.floor(Math.random() * 10) + 25, // 25-35 fps
      resolution: ['1080p', '4K', '720p'][Math.floor(Math.random() * 3)],
      bandwidth: Math.random() * 3 + 2, // 2-5 Mbps
      latency: Math.floor(Math.random() * 100) + 50, // 50-150ms
      errorRate: Math.random() * 2, // 0-2%
    },
    currentMetrics: {
      vehicleCount: Math.floor(Math.random() * 100),
      averageSpeed: Math.floor(Math.random() * 40) + 20,
      congestionLevel: ['free', 'light', 'moderate', 'heavy', 'severe'][Math.floor(Math.random() * 5)] as any,
      lastUpdate: Date.now() - Math.random() * 30000,
    },
    settings: {
      recordingEnabled: Math.random() > 0.3,
      alertsEnabled: Math.random() > 0.2,
      qualityProfile: ['low', 'medium', 'high', 'ultra'][Math.floor(Math.random() * 4)] as any,
      nightVisionEnabled: Math.random() > 0.4,
    },
    health: {
      temperature: Math.floor(Math.random() * 20) + 35, // 35-55°C
      storage: {
        used: Math.floor(Math.random() * 80) + 20,
        total: 100,
      },
      network: {
        signalStrength: Math.floor(Math.random() * 20) + 80, // 80-100%
        packetLoss: Math.random() * 2, // 0-2%
      },
    },
  }));

  const onlineCameras = cameras.filter(c => c.status === 'online').length;
  const offlineCameras = cameras.filter(c => c.status === 'offline').length;
  const maintenanceCameras = cameras.filter(c => c.status === 'maintenance').length;

  return {
    cameras,
    totalCameras: cameras.length,
    onlineCameras,
    offlineCameras,
    maintenanceCameras,
    overallUptime: cameras.reduce((sum, c) => sum + c.performance.uptime, 0) / cameras.length,
    averageFps: cameras.reduce((sum, c) => sum + c.performance.fps, 0) / cameras.length,
    totalBandwidth: cameras.reduce((sum, c) => sum + c.performance.bandwidth, 0),
    alertsCount: cameras.filter(c => c.performance.errorRate > 1 || c.performance.uptime < 90).length,
  };
};

export const CameraGridDashboard: React.FC<CameraGridDashboardProps> = ({
  data: externalData,
  className,
  showControls = true,
  autoRefresh = true,
  refreshInterval = 30000,
  viewMode = 'grid',
  gridSize = 'medium',
  onCameraSelect,
  onCameraSettings,
  onCameraStream,
  onDataExport,
}) => {
  const [isPending, startTransition] = useTransition();
  const [lastRefresh, setLastRefresh] = useState(Date.now());
  const [currentViewMode, setCurrentViewMode] = useState<'grid' | 'list'>(viewMode);
  const [currentGridSize, setCurrentGridSize] = useState(gridSize);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [locationFilter, setLocationFilter] = useState<string>('all');

  // Use real analytics data or mock data
  const analyticsData = useAnalyticsStore(state => state.currentMetrics);
  const { isConnected, metrics } = useRealTimeMetrics();

  // Generate or use provided data
  const data = useMemo(() => {
    if (externalData) return externalData;
    return generateMockCameraData();
  }, [externalData, lastRefresh]);

  const deferredData = useDeferredValue(data);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      startTransition(() => {
        setLastRefresh(Date.now());
      });
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  // Manual refresh
  const handleRefresh = () => {
    startTransition(() => {
      setLastRefresh(Date.now());
    });
  };

  // Filter cameras
  const filteredCameras = useMemo(() => {
    return deferredData.cameras.filter(camera => {
      const statusMatch = statusFilter === 'all' || camera.status === statusFilter;
      const locationMatch = locationFilter === 'all' || camera.location.includes(locationFilter);
      return statusMatch && locationMatch;
    });
  }, [deferredData.cameras, statusFilter, locationFilter]);

  // Unique locations for filter
  const locations = useMemo(() => {
    const uniqueLocations = Array.from(new Set(deferredData.cameras.map(c => c.location.split(' ')[0])));
    return uniqueLocations.slice(0, 10); // Limit to 10 for UI
  }, [deferredData.cameras]);

  // Metric card configurations
  const totalCamerasMetric: MetricCardData = {
    value: deferredData.totalCameras,
    unit: 'cameras',
    format: 'number',
    status: 'normal',
    metadata: {
      description: 'Total number of deployed cameras',
    },
  };

  const onlineCamerasMetric: MetricCardData = {
    value: deferredData.onlineCameras,
    previousValue: deferredData.totalCameras,
    unit: 'online',
    format: 'number',
    status: deferredData.onlineCameras / deferredData.totalCameras < 0.8 ? 'critical' :
           deferredData.onlineCameras / deferredData.totalCameras < 0.9 ? 'warning' : 'good',
    metadata: {
      description: 'Cameras currently operational and streaming',
    },
  };

  const uptimeMetric: MetricCardData = {
    value: deferredData.overallUptime,
    target: 99,
    unit: '%',
    format: 'percentage',
    status: deferredData.overallUptime < 95 ? 'critical' :
           deferredData.overallUptime < 98 ? 'warning' : 'good',
    metadata: {
      description: 'Average uptime across all cameras',
    },
  };

  const bandwidthMetric: MetricCardData = {
    value: deferredData.totalBandwidth,
    unit: 'Mbps',
    format: 'number',
    status: deferredData.totalBandwidth > 50 ? 'warning' : 'normal',
    metadata: {
      description: 'Total network bandwidth consumption',
    },
  };

  // Grid size classes
  const gridSizeClasses = {
    small: 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6',
    medium: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
    large: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Camera Grid</h2>
          <p className="text-muted-foreground">
            Live monitoring of all camera installations and performance
          </p>
        </div>

        {showControls && (
          <div className="flex items-center space-x-2">
            {/* Connection status */}
            <Badge variant={isConnected ? 'default' : 'destructive'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>

            {/* View mode toggle */}
            <ToggleGroup
              type="single"
              value={currentViewMode}
              onValueChange={(value) => value && setCurrentViewMode(value as 'grid' | 'list')}
            >
              <ToggleGroupItem value="grid" aria-label="Grid view">
                <IconGrid3x3 className="h-4 w-4" />
              </ToggleGroupItem>
              <ToggleGroupItem value="list" aria-label="List view">
                <IconList className="h-4 w-4" />
              </ToggleGroupItem>
            </ToggleGroup>

            {/* Grid size for grid view */}
            {currentViewMode === 'grid' && (
              <Select value={currentGridSize} onValueChange={setCurrentGridSize}>
                <SelectTrigger className="w-20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="small">Small</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="large">Large</SelectItem>
                </SelectContent>
              </Select>
            )}

            {/* Filters */}
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                {Object.keys(statusConfig).map(status => (
                  <SelectItem key={status} value={status}>
                    {statusConfig[status as keyof typeof statusConfig].label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={locationFilter} onValueChange={setLocationFilter}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Locations</SelectItem>
                {locations.map(location => (
                  <SelectItem key={location} value={location}>
                    {location}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Refresh button */}
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isPending}
            >
              <IconRefresh className={cn(
                'h-4 w-4 mr-2',
                isPending && 'animate-spin'
              )} />
              Refresh
            </Button>

            {/* Export button */}
            {onDataExport && (
              <Button variant="outline" size="sm" onClick={onDataExport}>
                <IconDownload className="h-4 w-4 mr-2" />
                Export
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Cameras"
          subtitle="Deployed installations"
          data={totalCamerasMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="Online Cameras"
          subtitle="Currently operational"
          data={onlineCamerasMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="System Uptime"
          subtitle="Average across all cameras"
          data={uptimeMetric}
          size="medium"
          showSparkline={false}
          showTarget={true}
        />

        <MetricCard
          title="Total Bandwidth"
          subtitle="Current network usage"
          data={bandwidthMetric}
          size="medium"
          showSparkline={false}
        />
      </div>

      {/* Camera Grid/List */}
      <DashboardWidget
        config={{
          id: 'camera-grid',
          title: `Camera ${currentViewMode === 'grid' ? 'Grid' : 'List'}`,
          subtitle: `${filteredCameras.length} cameras shown`,
          size: 'xlarge',
          priority: 'high',
          category: 'cameras',
        }}
        data={{
          timestamp: Date.now(),
          isLoading: false,
          data: filteredCameras,
        }}
      >
        <ScrollArea className="h-[600px]">
          {currentViewMode === 'grid' ? (
            // Grid View
            <div className={cn('grid gap-4', gridSizeClasses[currentGridSize])}>
              {filteredCameras.map((camera) => {
                const statusInfo = statusConfig[camera.status];
                const congestionInfo = congestionConfig[camera.currentMetrics.congestionLevel];
                const StatusIcon = statusInfo.icon;

                return (
                  <Card
                    key={camera.id}
                    className={cn(
                      'relative overflow-hidden cursor-pointer transition-all hover:shadow-lg',
                      'border-l-4',
                      statusInfo.bgColor.replace('/20', '/50')
                    )}
                    onClick={() => onCameraSelect?.(camera)}
                  >
                    {/* Camera thumbnail/preview */}
                    <div className="aspect-video bg-muted relative">
                      {camera.status === 'online' ? (
                        <div className="w-full h-full bg-gradient-to-br from-blue-500/20 to-green-500/20 flex items-center justify-center">
                          <IconCamera className="h-8 w-8 text-muted-foreground" />
                          <div className="absolute top-2 right-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0 bg-black/50 hover:bg-black/70"
                              onClick={(e) => {
                                e.stopPropagation();
                                onCameraStream?.(camera.id, 'fullscreen');
                              }}
                            >
                              <IconMaximize2 className="h-3 w-3 text-white" />
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="w-full h-full bg-muted flex items-center justify-center">
                          <StatusIcon className={cn('h-8 w-8', statusInfo.color)} />
                        </div>
                      )}

                      {/* Live indicator */}
                      {camera.status === 'online' && (
                        <div className="absolute top-2 left-2 flex items-center space-x-1 bg-black/50 rounded px-2 py-1">
                          <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                          <span className="text-xs text-white">LIVE</span>
                        </div>
                      )}

                      {/* FPS indicator */}
                      {camera.status === 'online' && (
                        <div className="absolute bottom-2 left-2 bg-black/50 rounded px-2 py-1">
                          <span className="text-xs text-white">{camera.performance.fps} FPS</span>
                        </div>
                      )}
                    </div>

                    {/* Camera info */}
                    <div className="p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-sm font-semibold truncate">{camera.name}</h4>
                        <Badge
                          variant="outline"
                          className={cn('text-xs', statusInfo.color)}
                        >
                          {statusInfo.label}
                        </Badge>
                      </div>

                      <p className="text-xs text-muted-foreground truncate">
                        {camera.location}
                      </p>

                      {/* Metrics */}
                      {camera.status === 'online' && (
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Traffic:</span>
                            <span className={congestionInfo.color}>
                              {camera.currentMetrics.vehicleCount} vehicles
                            </span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Speed:</span>
                            <span>{camera.currentMetrics.averageSpeed} mph</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Uptime:</span>
                            <span className={camera.performance.uptime > 95 ? 'text-success' : 'text-warning'}>
                              {camera.performance.uptime.toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Quick actions */}
                      <div className="flex items-center justify-between pt-2">
                        <div className="flex space-x-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={(e) => {
                              e.stopPropagation();
                              onCameraStream?.(camera.id, 'play');
                            }}
                            title="View stream"
                          >
                            <IconPlay className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={(e) => {
                              e.stopPropagation();
                              onCameraSettings?.(camera.id);
                            }}
                            title="Camera settings"
                          >
                            <IconSettings className="h-3 w-3" />
                          </Button>
                        </div>

                        {/* Signal strength indicator */}
                        <div className="flex items-center space-x-1">
                          <IconSignal className={cn(
                            'h-3 w-3',
                            camera.health.network.signalStrength > 80 ? 'text-success' :
                            camera.health.network.signalStrength > 60 ? 'text-warning' : 'text-destructive'
                          )} />
                          <span className="text-xs text-muted-foreground">
                            {camera.health.network.signalStrength.toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          ) : (
            // List View
            <div className="space-y-3">
              {filteredCameras.map((camera) => {
                const statusInfo = statusConfig[camera.status];
                const congestionInfo = congestionConfig[camera.currentMetrics.congestionLevel];
                const StatusIcon = statusInfo.icon;

                return (
                  <Card
                    key={camera.id}
                    className={cn(
                      'p-4 cursor-pointer transition-all hover:shadow-md',
                      'border-l-4',
                      statusInfo.bgColor.replace('/20', '/50')
                    )}
                    onClick={() => onCameraSelect?.(camera)}
                  >
                    <div className="flex items-center space-x-4">
                      {/* Camera thumbnail */}
                      <div className="flex-shrink-0 w-20 h-12 bg-muted rounded overflow-hidden">
                        {camera.status === 'online' ? (
                          <div className="w-full h-full bg-gradient-to-br from-blue-500/20 to-green-500/20 flex items-center justify-center">
                            <IconCamera className="h-4 w-4 text-muted-foreground" />
                          </div>
                        ) : (
                          <div className="w-full h-full bg-muted flex items-center justify-center">
                            <StatusIcon className={cn('h-4 w-4', statusInfo.color)} />
                          </div>
                        )}
                      </div>

                      {/* Camera details */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-semibold truncate">{camera.name}</h4>
                          <Badge
                            variant="outline"
                            className={cn('text-xs', statusInfo.color)}
                          >
                            {statusInfo.label}
                          </Badge>
                        </div>

                        <p className="text-xs text-muted-foreground truncate mt-1">
                          {camera.location}
                        </p>

                        {/* Performance metrics */}
                        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mt-2 text-xs">
                          <div>
                            <span className="text-muted-foreground block">Uptime</span>
                            <span className={camera.performance.uptime > 95 ? 'text-success' : 'text-warning'}>
                              {camera.performance.uptime.toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block">FPS</span>
                            <span>{camera.performance.fps}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block">Bandwidth</span>
                            <span>{camera.performance.bandwidth.toFixed(1)} Mbps</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block">Latency</span>
                            <span>{camera.performance.latency}ms</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block">Vehicles</span>
                            <span>{camera.currentMetrics.vehicleCount}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block">Speed</span>
                            <span>{camera.currentMetrics.averageSpeed} mph</span>
                          </div>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex-shrink-0 flex items-center space-x-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                          onClick={(e) => {
                            e.stopPropagation();
                            onCameraStream?.(camera.id, 'play');
                          }}
                          title="View stream"
                        >
                          <IconPlay className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                          onClick={(e) => {
                            e.stopPropagation();
                            onCameraSettings?.(camera.id);
                          }}
                          title="Camera settings"
                        >
                          <IconSettings className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                          onClick={(e) => {
                            e.stopPropagation();
                            onCameraStream?.(camera.id, 'fullscreen');
                          }}
                          title="Fullscreen"
                        >
                          <IconMaximize2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          )}

          {filteredCameras.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <IconCamera className="h-8 w-8 mx-auto mb-2" />
              <p>No cameras match the current filters</p>
            </div>
          )}
        </ScrollArea>
      </DashboardWidget>
    </div>
  );
};

export default CameraGridDashboard;
