/**
 * Camera Performance Dashboard Component
 *
 * Comprehensive camera monitoring dashboard with uptime metrics,
 * frame rate monitoring, health scoring, and network performance analysis.
 */

'use client';

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { useDeferredValue, useTransition } from 'react';
import { cn } from '@/lib/utils';
import {
  IconCamera,
  IconActivity,
  IconSignal,
  IconClock,
  IconRefresh,
  IconTrendingUp,
  IconTrendingDown,
  IconAlertTriangle,
  IconCheck,
  IconX,
  IconWifi,
  IconWifiOff
} from '@tabler/icons-react';
import { MetricCard, MetricCardData } from './MetricCard';
import { DashboardWidget, WidgetConfig } from './DashboardWidget';
import { RealTimeLineChart } from '@/components/charts/RealTimeLineChart';
import { RealTimeAreaChart } from '@/components/charts/RealTimeAreaChart';
import { RealTimeBarChart, BarChartData } from '@/components/charts/RealTimeBarChart';
import { useAnalyticsStore, useCameraHealth, CameraHealthMetrics } from '@/stores/analytics';
import { useRealTimeMetrics } from '@/providers/RealTimeProvider';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';

export interface CameraPerformanceData {
  totalCameras: number;
  onlineCameras: number;
  offlineCameras: number;
  maintenanceCameras: number;
  averageUptime: number;
  averageFrameRate: number;
  averageResponseTime: number;
  totalErrorCount: number;
  cameraDetails: Array<{
    cameraId: string;
    location: string;
    isOnline: boolean;
    uptime: number; // percentage
    frameRate: number;
    responseTime: number;
    quality: 'excellent' | 'good' | 'poor' | 'critical';
    lastUpdate: number;
    errorCount: number;
    networkLatency: number;
    dataUsage: number; // MB per hour
    resolution: string;
  }>;
  healthHistory: Array<{
    timestamp: number;
    healthScore: number;
    onlineCount: number;
    avgResponseTime: number;
    avgFrameRate: number;
  }>;
  alertCounts: {
    critical: number;
    warning: number;
    info: number;
  };
}

export interface CameraPerformanceDashboardProps {
  data?: CameraPerformanceData;
  className?: string;
  showControls?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onDataExport?: () => void;
  onCameraAction?: (cameraId: string, action: 'restart' | 'calibrate' | 'disable') => void;
  selectedTimeRange?: '1h' | '6h' | '24h' | '7d';
}

// Quality level configurations
const qualityConfig = {
  excellent: { color: 'text-success', bgColor: 'bg-success/20', label: 'Excellent', value: 95 },
  good: { color: 'text-success', bgColor: 'bg-success/20', label: 'Good', value: 75 },
  poor: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Poor', value: 50 },
  critical: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Critical', value: 25 },
};

// Generate mock camera performance data
const generateMockCameraData = (): CameraPerformanceData => {
  const now = Date.now();
  const cameraCount = 12;

  const cameraDetails = Array.from({ length: cameraCount }, (_, i) => {
    const isOnline = Math.random() > 0.15; // 85% online rate
    const quality = isOnline ?
      ['excellent', 'good', 'poor', 'critical'][Math.floor(Math.random() * 4)] as any :
      'critical';

    return {
      cameraId: `CAM-${String(i + 1).padStart(3, '0')}`,
      location: [
        'Main St & 1st Ave', 'Highway 101 North', 'Downtown Plaza', 'Industrial Blvd',
        'Airport Road', 'City Center', 'Park Avenue', 'Riverside Drive',
        'Tech District', 'University Campus', 'Shopping Center', 'Business District'
      ][i],
      isOnline,
      uptime: isOnline ? 85 + Math.random() * 15 : Math.random() * 30,
      frameRate: isOnline ? 20 + Math.random() * 10 : 0,
      responseTime: isOnline ? 50 + Math.random() * 200 : 0,
      quality,
      lastUpdate: now - Math.random() * 300000, // Last 5 minutes
      errorCount: Math.floor(Math.random() * 10),
      networkLatency: isOnline ? 10 + Math.random() * 100 : 0,
      dataUsage: isOnline ? 50 + Math.random() * 200 : 0,
      resolution: isOnline ? ['1080p', '720p', '4K'][Math.floor(Math.random() * 3)] : '0p',
    };
  });

  const onlineCameras = cameraDetails.filter(c => c.isOnline).length;
  const offlineCameras = cameraDetails.filter(c => !c.isOnline).length;

  return {
    totalCameras: cameraCount,
    onlineCameras,
    offlineCameras,
    maintenanceCameras: Math.floor(Math.random() * 2),
    averageUptime: onlineCameras > 0 ?
      cameraDetails.filter(c => c.isOnline).reduce((sum, c) => sum + c.uptime, 0) / onlineCameras : 0,
    averageFrameRate: onlineCameras > 0 ?
      cameraDetails.filter(c => c.isOnline).reduce((sum, c) => sum + c.frameRate, 0) / onlineCameras : 0,
    averageResponseTime: onlineCameras > 0 ?
      cameraDetails.filter(c => c.isOnline).reduce((sum, c) => sum + c.responseTime, 0) / onlineCameras : 0,
    totalErrorCount: cameraDetails.reduce((sum, c) => sum + c.errorCount, 0),
    cameraDetails,
    healthHistory: Array.from({ length: 20 }, (_, i) => ({
      timestamp: now - (19 - i) * 60000, // Last 20 minutes
      healthScore: 70 + Math.random() * 25,
      onlineCount: Math.floor(8 + Math.random() * 4),
      avgResponseTime: 80 + Math.random() * 40,
      avgFrameRate: 22 + Math.random() * 8,
    })),
    alertCounts: {
      critical: Math.floor(Math.random() * 3),
      warning: Math.floor(Math.random() * 5),
      info: Math.floor(Math.random() * 8),
    },
  };
};

export const CameraPerformanceDashboard: React.FC<CameraPerformanceDashboardProps> = ({
  data: externalData,
  className,
  showControls = true,
  autoRefresh = true,
  refreshInterval = 30000,
  onDataExport,
  onCameraAction,
  selectedTimeRange = '6h',
}) => {
  const [isPending, startTransition] = useTransition();
  const [lastRefresh, setLastRefresh] = useState(Date.now());
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState(selectedTimeRange);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'name' | 'status' | 'frameRate' | 'responseTime'>('status');
  const [showOfflineOnly, setShowOfflineOnly] = useState(false);

  // Use real camera health data
  const { cameras, summary } = useCameraHealth();
  const { isConnected, metrics } = useRealTimeMetrics();
  const updateCameraHealth = useAnalyticsStore(state => state.updateCameraHealth);

  // Generate or use provided data
  const data = useMemo(() => {
    if (externalData) return externalData;

    // Convert real camera health data to dashboard format
    if (Object.keys(cameras).length > 0) {
      const cameraDetails = Object.values(cameras).map(camera => ({
        cameraId: camera.cameraId,
        location: `Location ${camera.cameraId}`,
        isOnline: camera.isOnline,
        uptime: camera.isOnline ? 85 + Math.random() * 15 : 0,
        frameRate: camera.frameRate,
        responseTime: camera.responseTime,
        quality: camera.quality,
        lastUpdate: camera.lastUpdate,
        errorCount: camera.errorCount,
        networkLatency: camera.responseTime * 0.7, // Estimate
        dataUsage: camera.frameRate * 2, // Estimate based on frame rate
        resolution: camera.frameRate > 25 ? '1080p' : camera.frameRate > 15 ? '720p' : '480p',
      }));

      return {
        totalCameras: summary.total,
        onlineCameras: summary.online,
        offlineCameras: summary.offline,
        maintenanceCameras: 0,
        averageUptime: summary.online > 0 ? 90 : 0,
        averageFrameRate: summary.averageFrameRate,
        averageResponseTime: summary.averageResponseTime,
        totalErrorCount: Object.values(cameras).reduce((sum, c) => sum + c.errorCount, 0),
        cameraDetails,
        ...generateMockCameraData(), // Fill in missing data
        cameraDetails, // Override with real data
      };
    }

    return generateMockCameraData();
  }, [externalData, cameras, summary, lastRefresh]);

  const deferredData = useDeferredValue(data);

  // Calculate overall health score
  const healthScore = useMemo(() => {
    if (deferredData.totalCameras === 0) return 0;

    const weights = {
      uptime: 0.3,
      online: 0.25,
      frameRate: 0.2,
      responseTime: 0.15,
      errors: 0.1,
    };

    const uptimeScore = deferredData.averageUptime;
    const onlineScore = (deferredData.onlineCameras / deferredData.totalCameras) * 100;
    const frameRateScore = Math.min(deferredData.averageFrameRate / 30 * 100, 100);
    const responseTimeScore = Math.max(100 - (deferredData.averageResponseTime / 10), 0);
    const errorScore = Math.max(100 - (deferredData.totalErrorCount * 5), 0);

    return Math.round(
      uptimeScore * weights.uptime +
      onlineScore * weights.online +
      frameRateScore * weights.frameRate +
      responseTimeScore * weights.responseTime +
      errorScore * weights.errors
    );
  }, [deferredData]);

  // Filter and sort camera data
  const filteredAndSortedCameras = useMemo(() => {
    let filtered = deferredData.cameraDetails;

    if (showOfflineOnly) {
      filtered = filtered.filter(camera => !camera.isOnline);
    }

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.cameraId.localeCompare(b.cameraId);
        case 'status':
          if (a.isOnline !== b.isOnline) return a.isOnline ? -1 : 1;
          return a.cameraId.localeCompare(b.cameraId);
        case 'frameRate':
          return b.frameRate - a.frameRate;
        case 'responseTime':
          return a.responseTime - b.responseTime;
        default:
          return 0;
      }
    });
  }, [deferredData.cameraDetails, showOfflineOnly, sortBy]);

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
  const handleRefresh = useCallback(() => {
    startTransition(() => {
      setLastRefresh(Date.now());
    });
  }, []);

  // Camera action handler
  const handleCameraAction = useCallback((cameraId: string, action: 'restart' | 'calibrate' | 'disable') => {
    console.log(`Camera action: ${action} on ${cameraId}`);
    onCameraAction?.(cameraId, action);

    // Simulate action effect
    if (action === 'restart') {
      updateCameraHealth(cameraId, {
        isOnline: true,
        frameRate: 25 + Math.random() * 5,
        responseTime: 50 + Math.random() * 50,
        errorCount: 0,
      });
    }
  }, [onCameraAction, updateCameraHealth]);

  // Time range change handler
  const handleTimeRangeChange = useCallback((range: '1h' | '6h' | '24h' | '7d') => {
    setTimeRange(range);
    handleRefresh();
  }, [handleRefresh]);

  // Metric card configurations
  const healthScoreMetric: MetricCardData = {
    value: healthScore,
    unit: '%',
    format: 'number',
    status: healthScore > 85 ? 'good' : healthScore > 60 ? 'warning' : 'critical',
    timeSeries: deferredData.healthHistory.map(h => ({
      timestamp: h.timestamp,
      value: h.healthScore,
    })),
    metadata: {
      description: 'Overall system health score based on camera performance metrics',
      confidence: 0.95,
    },
  };

  const uptimeMetric: MetricCardData = {
    value: deferredData.averageUptime,
    unit: '%',
    format: 'number',
    status: deferredData.averageUptime > 95 ? 'good' :
           deferredData.averageUptime > 85 ? 'warning' : 'critical',
    metadata: {
      description: 'Average uptime across all cameras',
    },
  };

  const frameRateMetric: MetricCardData = {
    value: deferredData.averageFrameRate,
    unit: 'fps',
    format: 'number',
    status: deferredData.averageFrameRate > 25 ? 'good' :
           deferredData.averageFrameRate > 15 ? 'warning' : 'critical',
    timeSeries: deferredData.healthHistory.map(h => ({
      timestamp: h.timestamp,
      value: h.avgFrameRate,
    })),
    metadata: {
      description: 'Average frame rate across online cameras',
    },
  };

  const responseTimeMetric: MetricCardData = {
    value: deferredData.averageResponseTime,
    unit: 'ms',
    format: 'number',
    status: deferredData.averageResponseTime < 100 ? 'good' :
           deferredData.averageResponseTime < 500 ? 'warning' : 'critical',
    timeSeries: deferredData.healthHistory.map(h => ({
      timestamp: h.timestamp,
      value: h.avgResponseTime,
    })),
    metadata: {
      description: 'Average response time for camera commands',
    },
  };

  // Convert camera data for bar chart
  const cameraChartData: BarChartData[] = filteredAndSortedCameras.map(camera => ({
    category: camera.cameraId,
    value: camera.frameRate,
    timestamp: Date.now(),
    metadata: {
      location: camera.location,
      quality: camera.quality,
      responseTime: camera.responseTime,
      isOnline: camera.isOnline,
    },
  }));

  return (
    <div className={cn('space-y-6', className)}>
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Camera Performance</h2>
          <p className="text-muted-foreground">
            Real-time monitoring and health analysis of camera network
          </p>
        </div>

        {showControls && (
          <div className="flex items-center space-x-2">
            {/* Time range selector */}
            <Select value={timeRange} onValueChange={handleTimeRangeChange}>
              <SelectTrigger className="w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">1H</SelectItem>
                <SelectItem value="6h">6H</SelectItem>
                <SelectItem value="24h">24H</SelectItem>
                <SelectItem value="7d">7D</SelectItem>
              </SelectContent>
            </Select>

            {/* Connection status */}
            <Badge variant={isConnected ? 'default' : 'destructive'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>

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
                Export Data
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Health Score"
          subtitle="Overall system health"
          data={healthScoreMetric}
          size="medium"
          showSparkline={true}
        />

        <MetricCard
          title="Average Uptime"
          subtitle="Camera availability"
          data={uptimeMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="Frame Rate"
          subtitle="Average FPS"
          data={frameRateMetric}
          size="medium"
          showSparkline={true}
        />

        <MetricCard
          title="Response Time"
          subtitle="Command latency"
          data={responseTimeMetric}
          size="medium"
          showSparkline={true}
        />
      </div>

      {/* Tabbed Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="cameras">Camera Details</TabsTrigger>
          <TabsTrigger value="analytics">Performance Analytics</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Camera Status Summary */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Camera Status</h3>
                  <p className="text-sm text-muted-foreground">Network overview</p>
                </div>
                <IconCamera className="h-8 w-8 text-primary" />
              </div>

              <div className="mt-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <IconCheck className="h-4 w-4 text-success" />
                    <span className="text-sm">Online</span>
                  </div>
                  <div className="text-lg font-bold text-success">
                    {deferredData.onlineCameras}
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <IconX className="h-4 w-4 text-destructive" />
                    <span className="text-sm">Offline</span>
                  </div>
                  <div className="text-lg font-bold text-destructive">
                    {deferredData.offlineCameras}
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <IconClock className="h-4 w-4 text-warning" />
                    <span className="text-sm">Maintenance</span>
                  </div>
                  <div className="text-lg font-bold text-warning">
                    {deferredData.maintenanceCameras}
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Alert Summary</h3>
                  <p className="text-sm text-muted-foreground">Active alerts</p>
                </div>
                <IconAlertTriangle className="h-8 w-8 text-warning" />
              </div>

              <div className="mt-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Critical</span>
                  <Badge variant="destructive">{deferredData.alertCounts.critical}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Warning</span>
                  <Badge variant="secondary">{deferredData.alertCounts.warning}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Info</span>
                  <Badge variant="outline">{deferredData.alertCounts.info}</Badge>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Network Health</h3>
                  <p className="text-sm text-muted-foreground">Connectivity status</p>
                </div>
                <IconWifi className="h-8 w-8 text-success" />
              </div>

              <div className="mt-4 space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Avg Latency</span>
                    <span>{Math.round(deferredData.averageResponseTime)}ms</span>
                  </div>
                  <Progress
                    value={Math.max(100 - (deferredData.averageResponseTime / 10), 0)}
                    className="h-2"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Data Quality</span>
                    <span>{Math.round(deferredData.averageFrameRate / 30 * 100)}%</span>
                  </div>
                  <Progress
                    value={deferredData.averageFrameRate / 30 * 100}
                    className="h-2"
                  />
                </div>
              </div>
            </Card>
          </div>

          {/* Performance Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DashboardWidget
              config={{
                id: 'health-trend',
                title: 'Health Score Trend',
                subtitle: 'Overall system health over time',
                size: 'large',
                priority: 'high',
                category: 'camera',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.healthHistory,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.healthHistory.map(h => ({
                  timestamp: h.timestamp,
                  value: h.healthScore,
                }))}
                height={300}
                gradientColors={['hsl(var(--success))', 'transparent']}
                strokeColor="hsl(var(--success))"
                formatValue={(value) => `${Math.round(value)}%`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <DashboardWidget
              config={{
                id: 'camera-performance',
                title: 'Camera Frame Rates',
                subtitle: 'Real-time performance by camera',
                size: 'large',
                priority: 'medium',
                category: 'camera',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: cameraChartData,
              }}
            >
              <RealTimeBarChart
                data={cameraChartData}
                height={300}
                orientation="horizontal"
                showValues={true}
                sortBars="desc"
                formatValue={(value) => `${Math.round(value)} fps`}
                barColor="hsl(var(--primary))"
              />
            </DashboardWidget>
          </div>
        </TabsContent>

        {/* Camera Details Tab */}
        <TabsContent value="cameras" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Label htmlFor="sort-by" className="text-sm">Sort by:</Label>
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="name">Name</SelectItem>
                    <SelectItem value="status">Status</SelectItem>
                    <SelectItem value="frameRate">Frame Rate</SelectItem>
                    <SelectItem value="responseTime">Response Time</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  checked={showOfflineOnly}
                  onCheckedChange={setShowOfflineOnly}
                />
                <Label className="text-sm">Show offline only</Label>
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              Showing {filteredAndSortedCameras.length} of {deferredData.totalCameras} cameras
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredAndSortedCameras.map((camera) => (
              <Card key={camera.cameraId} className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {camera.isOnline ? (
                      <IconWifi className="h-4 w-4 text-success" />
                    ) : (
                      <IconWifiOff className="h-4 w-4 text-destructive" />
                    )}
                    <span className="font-medium">{camera.cameraId}</span>
                  </div>
                  <Badge
                    variant={camera.isOnline ? 'default' : 'destructive'}
                    className={cn(
                      qualityConfig[camera.quality].color,
                      qualityConfig[camera.quality].bgColor
                    )}
                  >
                    {camera.isOnline ? qualityConfig[camera.quality].label : 'Offline'}
                  </Badge>
                </div>

                <div className="text-sm text-muted-foreground mb-3">
                  {camera.location}
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Frame Rate:</span>
                    <span className={cn(
                      'font-medium',
                      camera.frameRate > 25 ? 'text-success' :
                      camera.frameRate > 15 ? 'text-warning' : 'text-destructive'
                    )}>
                      {camera.frameRate.toFixed(1)} fps
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span>Response Time:</span>
                    <span className={cn(
                      'font-medium',
                      camera.responseTime < 100 ? 'text-success' :
                      camera.responseTime < 500 ? 'text-warning' : 'text-destructive'
                    )}>
                      {Math.round(camera.responseTime)}ms
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span>Uptime:</span>
                    <span className="font-medium">{camera.uptime.toFixed(1)}%</span>
                  </div>

                  <div className="flex justify-between">
                    <span>Resolution:</span>
                    <span className="font-medium">{camera.resolution}</span>
                  </div>

                  {camera.errorCount > 0 && (
                    <div className="flex justify-between">
                      <span>Errors:</span>
                      <span className="font-medium text-destructive">{camera.errorCount}</span>
                    </div>
                  )}
                </div>

                {camera.isOnline && (
                  <div className="mt-4 flex space-x-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleCameraAction(camera.cameraId, 'restart')}
                      className="flex-1"
                    >
                      Restart
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleCameraAction(camera.cameraId, 'calibrate')}
                      className="flex-1"
                    >
                      Calibrate
                    </Button>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Performance Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DashboardWidget
              config={{
                id: 'response-time-trend',
                title: 'Response Time Trend',
                subtitle: 'Network latency over time',
                size: 'large',
                priority: 'medium',
                category: 'camera',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.healthHistory,
              }}
            >
              <RealTimeLineChart
                data={deferredData.healthHistory.map(h => ({
                  timestamp: h.timestamp,
                  value: h.avgResponseTime,
                }))}
                height={300}
                color="warning"
                formatValue={(value) => `${Math.round(value)}ms`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <DashboardWidget
              config={{
                id: 'frame-rate-distribution',
                title: 'Frame Rate Distribution',
                subtitle: 'Performance quality breakdown',
                size: 'large',
                priority: 'medium',
                category: 'camera',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: Object.entries(qualityConfig).map(([key, config]) => ({
                  category: config.label,
                  value: filteredAndSortedCameras.filter(c => c.quality === key).length,
                  timestamp: Date.now(),
                })),
              }}
            >
              <RealTimeBarChart
                data={Object.entries(qualityConfig).map(([key, config]) => ({
                  category: config.label,
                  value: filteredAndSortedCameras.filter(c => c.quality === key).length,
                  timestamp: Date.now(),
                }))}
                height={300}
                orientation="vertical"
                showValues={true}
                sortBars="desc"
                formatValue={(value) => `${value} cameras`}
              />
            </DashboardWidget>
          </div>

          <DashboardWidget
            config={{
              id: 'network-latency-heatmap',
              title: 'Network Performance Matrix',
              subtitle: 'Latency and throughput analysis by camera',
              size: 'xlarge',
              priority: 'medium',
              category: 'camera',
            }}
            data={{
              timestamp: Date.now(),
              isLoading: false,
              data: null,
            }}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Camera</th>
                    <th className="text-left p-2">Status</th>
                    <th className="text-left p-2">Frame Rate</th>
                    <th className="text-left p-2">Response Time</th>
                    <th className="text-left p-2">Data Usage</th>
                    <th className="text-left p-2">Quality</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAndSortedCameras.slice(0, 10).map((camera) => (
                    <tr key={camera.cameraId} className="border-b hover:bg-muted/50">
                      <td className="p-2 font-medium">{camera.cameraId}</td>
                      <td className="p-2">
                        <Badge
                          variant={camera.isOnline ? 'default' : 'destructive'}
                          className="text-xs"
                        >
                          {camera.isOnline ? 'Online' : 'Offline'}
                        </Badge>
                      </td>
                      <td className="p-2">
                        <div className="flex items-center space-x-2">
                          <span>{camera.frameRate.toFixed(1)} fps</span>
                          <div className="flex-1 max-w-16">
                            <Progress
                              value={camera.frameRate / 30 * 100}
                              className="h-1"
                            />
                          </div>
                        </div>
                      </td>
                      <td className="p-2">
                        <div className="flex items-center space-x-2">
                          <span>{Math.round(camera.responseTime)}ms</span>
                          <div className="flex-1 max-w-16">
                            <Progress
                              value={Math.max(100 - (camera.responseTime / 10), 0)}
                              className="h-1"
                            />
                          </div>
                        </div>
                      </td>
                      <td className="p-2">{Math.round(camera.dataUsage)} MB/h</td>
                      <td className="p-2">
                        <span className={cn(
                          'text-xs font-medium',
                          qualityConfig[camera.quality].color
                        )}>
                          {qualityConfig[camera.quality].label}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </DashboardWidget>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CameraPerformanceDashboard;
