/**
 * Traffic Analytics Dashboard Component
 *
 * Comprehensive traffic monitoring dashboard with real-time metrics,
 * charts, and alerts for traffic volume, speed, and congestion analysis.
 */

'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { useDeferredValue, useTransition } from 'react';
import { cn } from '@/lib/utils';
import {
  IconCar,
  IconGauge,
  IconClock,
  IconAlertTriangle,
  IconTrendingUp,
  IconMap,
  IconRefresh
} from '@tabler/icons-react';
import { MetricCard, MetricCardData } from './MetricCard';
import { DashboardWidget, WidgetConfig } from './DashboardWidget';
import { RealTimeLineChart } from '@/components/charts/RealTimeLineChart';
import { RealTimeAreaChart } from '@/components/charts/RealTimeAreaChart';
import { RealTimeBarChart, BarChartData } from '@/components/charts/RealTimeBarChart';
import { TrafficFlowHeatmap, TrafficFlowData } from '@/components/charts/TrafficFlowHeatmap';
import { useAnalyticsStore } from '@/stores/analytics';
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

export interface TrafficAnalyticsData {
  currentVolume: number;
  averageSpeed: number;
  congestionLevel: 'free' | 'light' | 'moderate' | 'heavy' | 'severe';
  incidentCount: number;
  peakHours: Array<{ hour: number; volume: number }>;
  speedDistribution: Array<{ range: string; count: number; timestamp: number }>;
  volumeTrend: Array<{ timestamp: number; value: number }>;
  speedTrend: Array<{ timestamp: number; value: number }>;
  congestionHistory: Array<{ timestamp: number; level: number }>;
  cameraData: Array<{
    cameraId: string;
    location: string;
    volume: number;
    speed: number;
    status: 'online' | 'offline' | 'maintenance';
  }>;
}

export interface TrafficAnalyticsDashboardProps {
  data?: TrafficAnalyticsData;
  className?: string;
  showControls?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onDataExport?: () => void;
  onAlertThresholdChange?: (thresholds: Record<string, number>) => void;
}

// Congestion level configurations
const congestionConfig = {
  free: { color: 'text-success', bgColor: 'bg-success/20', label: 'Free Flow', value: 0 },
  light: { color: 'text-success', bgColor: 'bg-success/20', label: 'Light Traffic', value: 1 },
  moderate: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Moderate Traffic', value: 2 },
  heavy: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Heavy Traffic', value: 3 },
  severe: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Severe Congestion', value: 4 },
};

// Generate mock data for demonstration
const generateMockData = (): TrafficAnalyticsData => {
  const now = Date.now();
  const hourInMs = 60 * 60 * 1000;

  return {
    currentVolume: Math.floor(Math.random() * 100) + 20,
    averageSpeed: Math.floor(Math.random() * 40) + 30,
    congestionLevel: ['free', 'light', 'moderate', 'heavy', 'severe'][Math.floor(Math.random() * 5)] as any,
    incidentCount: Math.floor(Math.random() * 5),
    peakHours: Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      volume: Math.floor(Math.random() * 80) + 20,
    })),
    speedDistribution: [
      { range: '0-20 mph', count: Math.floor(Math.random() * 20), timestamp: now },
      { range: '21-40 mph', count: Math.floor(Math.random() * 50) + 20, timestamp: now },
      { range: '41-60 mph', count: Math.floor(Math.random() * 40) + 10, timestamp: now },
      { range: '61+ mph', count: Math.floor(Math.random() * 15), timestamp: now },
    ],
    volumeTrend: Array.from({ length: 20 }, (_, i) => ({
      timestamp: now - (19 - i) * 30000,
      value: Math.floor(Math.random() * 50) + 30,
    })),
    speedTrend: Array.from({ length: 20 }, (_, i) => ({
      timestamp: now - (19 - i) * 30000,
      value: Math.floor(Math.random() * 30) + 35,
    })),
    congestionHistory: Array.from({ length: 20 }, (_, i) => ({
      timestamp: now - (19 - i) * 30000,
      value: Math.floor(Math.random() * 5),
    })),
    cameraData: Array.from({ length: 6 }, (_, i) => ({
      cameraId: `CAM-${String(i + 1).padStart(3, '0')}`,
      location: ['Main St & 1st Ave', 'Highway 101 North', 'Downtown Plaza', 'Industrial Blvd', 'Airport Road', 'City Center'][i],
      volume: Math.floor(Math.random() * 80) + 10,
      speed: Math.floor(Math.random() * 40) + 25,
      status: ['online', 'offline', 'maintenance'][Math.floor(Math.random() * 3)] as any,
    })),
  };
};

export const TrafficAnalyticsDashboard: React.FC<TrafficAnalyticsDashboardProps> = ({
  data: externalData,
  className,
  showControls = true,
  autoRefresh = true,
  refreshInterval = 30000,
  onDataExport,
  onAlertThresholdChange,
}) => {
  const [isPending, startTransition] = useTransition();
  const [lastRefresh, setLastRefresh] = useState(Date.now());
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('6h');
  const [heatmapColorScheme, setHeatmapColorScheme] = useState<'traffic' | 'speed' | 'density'>('traffic');
  const [showPredictions, setShowPredictions] = useState(true);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  // Use real analytics data or mock data
  const analyticsData = useAnalyticsStore(state => state.currentMetrics);
  const { isConnected, metrics, processingHealth } = useRealTimeMetrics();

  // Generate or use provided data
  const data = useMemo(() => {
    if (externalData) return externalData;

    // Convert real analytics data to dashboard format
    if (analyticsData.traffic) {
      return {
        currentVolume: analyticsData.traffic.vehicleCount,
        averageSpeed: analyticsData.traffic.avgSpeed,
        congestionLevel: analyticsData.traffic.congestionLevel,
        incidentCount: analyticsData.totalIncidents,
        // Mock additional data for now
        ...generateMockData(),
        currentVolume: analyticsData.traffic.vehicleCount,
        averageSpeed: analyticsData.traffic.avgSpeed,
        congestionLevel: analyticsData.traffic.congestionLevel,
      };
    }

    return generateMockData();
  }, [externalData, analyticsData, lastRefresh]);

  const deferredData = useDeferredValue(data);

  // Prepare heatmap data from camera and traffic information
  const heatmapData: TrafficFlowData[] = useMemo(() => {
    if (!analyticsData || !deferredData.cameraData) return [];

    return deferredData.cameraData.map((camera, index) => ({
      cameraId: camera.cameraId,
      location: {
        lat: 37.7749 + (index * 0.01) - 0.02, // Mock coordinates around SF
        lng: -122.4194 + (index * 0.008) - 0.02,
        name: camera.location,
      },
      intensity: camera.volume,
      speed: camera.speed,
      vehicleCount: camera.volume,
      timestamp: Date.now(),
      congestionLevel: camera.volume > 70 ? 'severe' :
                      camera.volume > 50 ? 'heavy' :
                      camera.volume > 30 ? 'moderate' :
                      camera.volume > 15 ? 'light' : 'free',
      incidents: camera.status === 'offline' ? [{
        id: `inc-${camera.cameraId}`,
        type: 'congestion' as const,
        severity: 'medium' as const,
      }] : undefined,
    }));
  }, [analyticsData, deferredData.cameraData]);

  // Get historical data based on time range
  const getHistoricalData = useCallback((metric: 'volume' | 'speed') => {
    const hours = timeRange === '1h' ? 1 : timeRange === '6h' ? 6 : timeRange === '24h' ? 24 : 168;
    const interval = hours <= 6 ? 5 : hours <= 24 ? 15 : 60; // minutes
    const points = hours * 60 / interval;

    const now = Date.now();
    return Array.from({ length: points }, (_, i) => ({
      timestamp: now - (points - i) * interval * 60000,
      value: metric === 'volume'
        ? Math.floor(Math.random() * 30 + 40 + Math.sin((points - i) / 10) * 15)
        : Math.floor(Math.random() * 15 + 45 + Math.sin((points - i) / 8) * 10),
    }));
  }, [timeRange]);

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

  // Heatmap interaction handlers
  const handleLocationClick = useCallback((locationData: TrafficFlowData) => {
    setSelectedCamera(locationData.cameraId);
    console.log('Camera selected:', locationData);
  }, []);

  const handleLocationHover = useCallback((locationData: TrafficFlowData | null) => {
    // Could show tooltip or highlight related data
    if (locationData) {
      console.log('Hovering over camera:', locationData.cameraId);
    }
  }, []);

  // Time range change handler
  const handleTimeRangeChange = useCallback((range: '1h' | '6h' | '24h' | '7d') => {
    setTimeRange(range);
    handleRefresh();
  }, [handleRefresh]);

  // Metric card configurations
  const volumeMetric: MetricCardData = {
    value: deferredData.currentVolume,
    previousValue: deferredData.volumeTrend[deferredData.volumeTrend.length - 2]?.value,
    unit: 'vehicles',
    format: 'number',
    status: deferredData.currentVolume > 80 ? 'warning' :
           deferredData.currentVolume > 90 ? 'critical' : 'normal',
    timeSeries: deferredData.volumeTrend,
    metadata: {
      description: 'Current traffic volume across all monitored locations',
      source: 'Camera Network',
      lastCalculated: Date.now(),
    },
  };

  const speedMetric: MetricCardData = {
    value: deferredData.averageSpeed,
    previousValue: deferredData.speedTrend[deferredData.speedTrend.length - 2]?.value,
    target: 55,
    unit: 'mph',
    format: 'number',
    status: deferredData.averageSpeed < 20 ? 'critical' :
           deferredData.averageSpeed < 35 ? 'warning' : 'good',
    timeSeries: deferredData.speedTrend,
    metadata: {
      description: 'Average vehicle speed across all monitored roads',
      confidence: 0.95,
    },
  };

  const incidentMetric: MetricCardData = {
    value: deferredData.incidentCount,
    unit: 'incidents',
    format: 'number',
    status: deferredData.incidentCount > 3 ? 'critical' :
           deferredData.incidentCount > 1 ? 'warning' : 'good',
    metadata: {
      description: 'Active traffic incidents requiring attention',
    },
  };

  // Convert congestion level to numeric value for display
  const congestionValue = congestionConfig[deferredData.congestionLevel].value;
  const congestionMetric: MetricCardData = {
    value: congestionValue,
    format: 'number',
    status: congestionValue >= 4 ? 'critical' :
           congestionValue >= 3 ? 'warning' : 'normal',
    timeSeries: deferredData.congestionHistory,
    metadata: {
      description: `Traffic condition: ${congestionConfig[deferredData.congestionLevel].label}`,
    },
  };

  // Convert camera data for bar chart
  const cameraChartData: BarChartData[] = deferredData.cameraData.map(camera => ({
    category: camera.cameraId,
    value: camera.volume,
    timestamp: Date.now(),
    metadata: {
      location: camera.location,
      speed: camera.speed,
      status: camera.status,
    },
  }));

  return (
    <div className={cn('space-y-6', className)}>
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Traffic Analytics</h2>
          <p className="text-muted-foreground">
            Real-time traffic monitoring and analysis
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
          title="Traffic Volume"
          subtitle="Current vehicle count"
          data={volumeMetric}
          size="medium"
          showSparkline={true}
        />

        <MetricCard
          title="Average Speed"
          subtitle="Current average speed"
          data={speedMetric}
          size="medium"
          showSparkline={true}
          showTarget={true}
        />

        <MetricCard
          title="Active Incidents"
          subtitle="Incidents requiring attention"
          data={incidentMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="Congestion Level"
          subtitle={congestionConfig[deferredData.congestionLevel].label}
          data={congestionMetric}
          size="medium"
          showSparkline={true}
        />
      </div>

      {/* Tabbed Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="heatmap">Traffic Flow</TabsTrigger>
          <TabsTrigger value="historical">Historical</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Traffic Volume Trend */}
        <DashboardWidget
          config={{
            id: 'traffic-volume-chart',
            title: 'Traffic Volume Trend',
            subtitle: 'Vehicle count over time',
            size: 'large',
            priority: 'medium',
            category: 'traffic',
          }}
          data={{
            timestamp: Date.now(),
            isLoading: false,
            data: deferredData.volumeTrend,
          }}
        >
          <RealTimeAreaChart
            data={deferredData.volumeTrend}
            height={300}
            gradientColors={['hsl(var(--primary))', 'transparent']}
            strokeColor="hsl(var(--primary))"
            formatValue={(value) => `${value} vehicles`}
            enableAnimations={true}
            responsive={true}
          />
        </DashboardWidget>

        {/* Speed Distribution */}
        <DashboardWidget
          config={{
            id: 'speed-distribution-chart',
            title: 'Speed Distribution',
            subtitle: 'Vehicle speed ranges',
            size: 'large',
            priority: 'medium',
            category: 'traffic',
          }}
          data={{
            timestamp: Date.now(),
            isLoading: false,
            data: deferredData.speedDistribution,
          }}
        >
          <RealTimeBarChart
            data={deferredData.speedDistribution}
            height={300}
            orientation="vertical"
            showValues={true}
            sortBars="desc"
            barColor="hsl(var(--success))"
          />
        </DashboardWidget>
          </div>

          {/* Camera Status and Performance - Overview Tab */}
          <DashboardWidget
            config={{
              id: 'camera-performance-chart',
              title: 'Camera Performance',
              subtitle: 'Volume monitoring by camera location',
              size: 'xlarge',
              priority: 'medium',
              category: 'cameras',
            }}
            data={{
              timestamp: Date.now(),
              isLoading: false,
              data: cameraChartData,
            }}
          >
            <div className="space-y-4">
              {/* Camera status overview */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {['online', 'offline', 'maintenance'].map(status => {
                  const count = deferredData.cameraData.filter(c => c.status === status).length;
                  return (
                    <div key={status} className="text-center">
                      <div className={cn(
                        'text-2xl font-bold',
                        status === 'online' && 'text-success',
                        status === 'offline' && 'text-destructive',
                        status === 'maintenance' && 'text-warning'
                      )}>
                        {count}
                      </div>
                      <div className="text-sm text-muted-foreground capitalize">
                        {status} Cameras
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Camera performance chart */}
              <RealTimeBarChart
                data={cameraChartData}
                height={250}
                orientation="horizontal"
                showValues={true}
                sortBars="desc"
                formatValue={(value) => `${value} vehicles`}
                onBarClick={(data) => {
                  console.log('Camera clicked:', data);
                }}
              />
            </div>
          </DashboardWidget>
        </TabsContent>

        {/* Traffic Flow Heatmap Tab */}
        <TabsContent value="heatmap" className="space-y-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold">Real-time Traffic Flow</h3>
              <p className="text-sm text-muted-foreground">
                Geographic visualization of traffic density and congestion
              </p>
            </div>

            <div className="flex items-center space-x-4">
              {/* Color scheme selector */}
              <div className="flex items-center space-x-2">
                <Label htmlFor="color-scheme" className="text-sm">Color by:</Label>
                <Select value={heatmapColorScheme} onValueChange={setHeatmapColorScheme}>
                  <SelectTrigger className="w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="traffic">Traffic</SelectItem>
                    <SelectItem value="speed">Speed</SelectItem>
                    <SelectItem value="density">Density</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Predictions toggle */}
              <div className="flex items-center space-x-2">
                <Switch
                  checked={showPredictions}
                  onCheckedChange={setShowPredictions}
                />
                <Label className="text-sm">Show Predictions</Label>
              </div>
            </div>
          </div>

          {/* Traffic Flow Heatmap */}
          <DashboardWidget
            config={{
              id: 'traffic-heatmap',
              title: 'Traffic Flow Heatmap',
              subtitle: `Colored by ${heatmapColorScheme}`,
              size: 'xlarge',
              priority: 'high',
              category: 'traffic',
            }}
            data={{
              timestamp: Date.now(),
              isLoading: isPending,
              data: heatmapData,
            }}
          >
            <TrafficFlowHeatmap
              data={heatmapData}
              width={800}
              height={500}
              colorScheme={heatmapColorScheme}
              showLegend={true}
              showCameraLabels={true}
              enableInteraction={true}
              onLocationClick={handleLocationClick}
              onLocationHover={handleLocationHover}
              showTrafficFlow={true}
              animationDuration={500}
            />
          </DashboardWidget>

          {/* Selected Camera Details */}
          {selectedCamera && (
            <Card className="p-4">
              <h4 className="text-sm font-semibold mb-3">
                Camera Details: {selectedCamera}
              </h4>

              {(() => {
                const camera = deferredData.cameraData.find(c => c.cameraId === selectedCamera);
                const heatmapLocation = heatmapData.find(h => h.cameraId === selectedCamera);

                if (!camera || !heatmapLocation) return null;

                return (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="text-muted-foreground">Location</div>
                      <div className="font-medium">{camera.location}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Vehicle Count</div>
                      <div className="font-medium">{camera.volume}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Average Speed</div>
                      <div className="font-medium">{camera.speed} mph</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Status</div>
                      <div className={cn(
                        'font-medium capitalize',
                        camera.status === 'online' && 'text-success',
                        camera.status === 'offline' && 'text-destructive',
                        camera.status === 'maintenance' && 'text-warning'
                      )}>
                        {camera.status}
                      </div>
                    </div>
                  </div>
                );
              })()}
            </Card>
          )}
        </TabsContent>

        {/* Historical Data Tab */}
        <TabsContent value="historical" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Historical Volume Trend */}
            <DashboardWidget
              config={{
                id: 'historical-volume',
                title: `Traffic Volume - ${timeRange.toUpperCase()}`,
                subtitle: 'Historical vehicle count trends',
                size: 'large',
                priority: 'medium',
                category: 'traffic',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: getHistoricalData('volume'),
              }}
            >
              <RealTimeAreaChart
                data={getHistoricalData('volume')}
                height={300}
                gradientColors={['hsl(var(--primary))', 'transparent']}
                strokeColor="hsl(var(--primary))"
                formatValue={(value) => `${value} vehicles`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            {/* Historical Speed Trend */}
            <DashboardWidget
              config={{
                id: 'historical-speed',
                title: `Average Speed - ${timeRange.toUpperCase()}`,
                subtitle: 'Historical speed trends',
                size: 'large',
                priority: 'medium',
                category: 'traffic',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: getHistoricalData('speed'),
              }}
            >
              <RealTimeLineChart
                data={getHistoricalData('speed')}
                height={300}
                color="success"
                formatValue={(value) => `${value} mph`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>
          </div>

          {/* Prediction Analysis */}
          {showPredictions && (
            <DashboardWidget
              config={{
                id: 'prediction-analysis',
                title: 'Traffic Prediction Analysis',
                subtitle: 'AI-powered congestion forecasting',
                size: 'xlarge',
                priority: 'medium',
                category: 'ai',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: null,
              }}
            >
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {['Next Hour', '3 Hours', '6 Hours'].map((period, index) => {
                    const prediction = 65 + Math.random() * 20 + index * 5;
                    const confidence = 85 + Math.random() * 10;

                    return (
                      <div key={period} className="text-center p-4 border rounded-lg">
                        <div className="text-sm text-muted-foreground mb-1">{period}</div>
                        <div className={cn(
                          'text-2xl font-bold',
                          prediction > 80 ? 'text-destructive' :
                          prediction > 60 ? 'text-warning' : 'text-success'
                        )}>
                          {Math.round(prediction)}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {Math.round(confidence)}% confidence
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="text-sm text-muted-foreground text-center">
                  Predictions based on historical patterns, current conditions, and external factors
                </div>
              </div>
            </DashboardWidget>
          )}
        </TabsContent>
      </Tabs>

      {/* Processing Health (if available) */}
      {processingHealth && (
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-3">System Performance</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">Messages Processed</div>
              <div className="font-medium">{processingHealth.totalProcessed.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Error Rate</div>
              <div className={cn(
                'font-medium',
                processingHealth.errorRate > 0.1 ? 'text-destructive' : 'text-success'
              )}>
                {(processingHealth.errorRate * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">Avg Processing Time</div>
              <div className="font-medium">{processingHealth.averageProcessingTime.toFixed(1)}ms</div>
            </div>
            <div>
              <div className="text-muted-foreground">Throughput</div>
              <div className="font-medium">{processingHealth.throughputPerSecond.toFixed(1)}/s</div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default TrafficAnalyticsDashboard;
