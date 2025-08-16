/**
 * System Metrics Dashboard Component
 *
 * Comprehensive system monitoring dashboard with resource utilization,
 * API response times, database performance, and service health analytics.
 */

'use client';

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { useDeferredValue, useTransition } from 'react';
import { cn } from '@/lib/utils';
import {
  IconCpu,
  IconDatabase,
  IconServer,
  IconNetwork,
  IconRefresh,
  IconTrendingUp,
  IconTrendingDown,
  IconAlertTriangle,
  IconCheck,
  IconX,
  IconActivity,
  IconMemory,
  IconClock,
  IconApi,
  IconChartBar
} from '@tabler/icons-react';
import { MetricCard, MetricCardData } from './MetricCard';
import { DashboardWidget, WidgetConfig } from './DashboardWidget';
import { RealTimeLineChart } from '@/components/charts/RealTimeLineChart';
import { RealTimeAreaChart } from '@/components/charts/RealTimeAreaChart';
import { RealTimeBarChart, BarChartData } from '@/components/charts/RealTimeBarChart';
import { useAnalyticsStore, useSystemHealth } from '@/stores/analytics';
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

export interface SystemMetricsData {
  // Resource utilization
  cpu: {
    usage: number; // percentage
    cores: number;
    frequency: number; // GHz
    temperature: number; // Celsius
    load: Array<{ timestamp: number; value: number }>;
  };
  memory: {
    used: number; // bytes
    total: number; // bytes
    usage: number; // percentage
    available: number; // bytes
    cache: number; // bytes
    load: Array<{ timestamp: number; value: number }>;
  };
  disk: {
    used: number; // bytes
    total: number; // bytes
    usage: number; // percentage
    readSpeed: number; // MB/s
    writeSpeed: number; // MB/s
    iops: number;
    load: Array<{ timestamp: number; value: number }>;
  };
  network: {
    inbound: number; // bytes/s
    outbound: number; // bytes/s
    latency: number; // ms
    packetLoss: number; // percentage
    connections: number;
    bandwidth: number; // Mbps
    load: Array<{ timestamp: number; inbound: number; outbound: number }>;
  };

  // Application metrics
  api: {
    totalRequests: number;
    requestsPerSecond: number;
    averageResponseTime: number; // ms
    errorRate: number; // percentage
    p95ResponseTime: number; // ms
    p99ResponseTime: number; // ms
    endpoints: Array<{
      path: string;
      method: string;
      requests: number;
      avgResponseTime: number;
      errorRate: number;
    }>;
    load: Array<{ timestamp: number; requests: number; responseTime: number; errors: number }>;
  };

  // Database metrics
  database: {
    connections: {
      active: number;
      idle: number;
      max: number;
    };
    queries: {
      total: number;
      perSecond: number;
      averageExecutionTime: number; // ms
      slowQueries: number;
    };
    cache: {
      hitRate: number; // percentage
      size: number; // bytes
      evictions: number;
    };
    replication: {
      lag: number; // ms
      status: 'healthy' | 'warning' | 'critical';
    };
    load: Array<{ timestamp: number; connections: number; queries: number; executionTime: number }>;
  };

  // Service health
  services: Array<{
    name: string;
    status: 'healthy' | 'degraded' | 'unhealthy';
    uptime: number; // percentage
    responseTime: number; // ms
    lastCheck: number;
    endpoint: string;
    dependencies: string[];
  }>;

  // System alerts
  alerts: Array<{
    id: string;
    type: 'cpu' | 'memory' | 'disk' | 'network' | 'api' | 'database' | 'service';
    severity: 'info' | 'warning' | 'critical';
    title: string;
    description: string;
    timestamp: number;
    resolved: boolean;
  }>;

  timestamp: number;
}

export interface SystemMetricsDashboardProps {
  data?: SystemMetricsData;
  className?: string;
  showControls?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onDataExport?: () => void;
  onAlertAction?: (alertId: string, action: 'acknowledge' | 'resolve') => void;
  selectedTimeRange?: '5m' | '15m' | '1h' | '6h' | '24h';
}

// Status configurations
const statusConfig = {
  healthy: { color: 'text-success', bgColor: 'bg-success/20', label: 'Healthy' },
  degraded: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Degraded' },
  unhealthy: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Unhealthy' },
  warning: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Warning' },
  critical: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Critical' },
};

// Generate mock system metrics data
const generateMockSystemData = (): SystemMetricsData => {
  const now = Date.now();
  const points = 20;

  // Generate time series data
  const generateTimeSeries = (baseValue: number, variance: number) =>
    Array.from({ length: points }, (_, i) => ({
      timestamp: now - (points - i) * 60000, // Last 20 minutes
      value: Math.max(0, baseValue + (Math.random() - 0.5) * variance),
    }));

  const generateNetworkSeries = () =>
    Array.from({ length: points }, (_, i) => ({
      timestamp: now - (points - i) * 60000,
      inbound: Math.random() * 100 + 50, // MB/s
      outbound: Math.random() * 50 + 25, // MB/s
    }));

  const generateApiSeries = () =>
    Array.from({ length: points }, (_, i) => ({
      timestamp: now - (points - i) * 60000,
      requests: Math.floor(Math.random() * 100 + 200),
      responseTime: Math.random() * 200 + 50,
      errors: Math.floor(Math.random() * 5),
    }));

  const generateDbSeries = () =>
    Array.from({ length: points }, (_, i) => ({
      timestamp: now - (points - i) * 60000,
      connections: Math.floor(Math.random() * 20 + 50),
      queries: Math.floor(Math.random() * 500 + 1000),
      executionTime: Math.random() * 100 + 20,
    }));

  const cpuUsage = 45 + Math.random() * 30;
  const memoryUsage = 60 + Math.random() * 25;
  const diskUsage = 35 + Math.random() * 40;

  return {
    cpu: {
      usage: cpuUsage,
      cores: 8,
      frequency: 3.2 + Math.random() * 0.8,
      temperature: 55 + Math.random() * 20,
      load: generateTimeSeries(cpuUsage, 20),
    },
    memory: {
      used: Math.floor(8 * 1024 * 1024 * 1024 * memoryUsage / 100), // 8GB system
      total: 8 * 1024 * 1024 * 1024,
      usage: memoryUsage,
      available: Math.floor(8 * 1024 * 1024 * 1024 * (100 - memoryUsage) / 100),
      cache: Math.floor(2 * 1024 * 1024 * 1024),
      load: generateTimeSeries(memoryUsage, 15),
    },
    disk: {
      used: Math.floor(500 * 1024 * 1024 * 1024 * diskUsage / 100), // 500GB disk
      total: 500 * 1024 * 1024 * 1024,
      usage: diskUsage,
      readSpeed: 150 + Math.random() * 100,
      writeSpeed: 100 + Math.random() * 50,
      iops: Math.floor(1000 + Math.random() * 500),
      load: generateTimeSeries(diskUsage, 10),
    },
    network: {
      inbound: 75 + Math.random() * 50,
      outbound: 45 + Math.random() * 30,
      latency: 10 + Math.random() * 20,
      packetLoss: Math.random() * 0.5,
      connections: Math.floor(150 + Math.random() * 100),
      bandwidth: 1000, // 1Gbps
      load: generateNetworkSeries(),
    },
    api: {
      totalRequests: Math.floor(50000 + Math.random() * 20000),
      requestsPerSecond: 250 + Math.random() * 100,
      averageResponseTime: 120 + Math.random() * 80,
      errorRate: Math.random() * 2,
      p95ResponseTime: 300 + Math.random() * 200,
      p99ResponseTime: 800 + Math.random() * 400,
      endpoints: [
        { path: '/api/cameras', method: 'GET', requests: 15000, avgResponseTime: 85, errorRate: 0.1 },
        { path: '/api/analytics', method: 'GET', requests: 12000, avgResponseTime: 150, errorRate: 0.2 },
        { path: '/api/incidents', method: 'POST', requests: 3000, avgResponseTime: 200, errorRate: 0.5 },
        { path: '/api/stream', method: 'WebSocket', requests: 8000, avgResponseTime: 45, errorRate: 0.05 },
        { path: '/api/auth', method: 'POST', requests: 2000, avgResponseTime: 300, errorRate: 1.0 },
      ],
      load: generateApiSeries(),
    },
    database: {
      connections: {
        active: Math.floor(45 + Math.random() * 20),
        idle: Math.floor(25 + Math.random() * 15),
        max: 100,
      },
      queries: {
        total: Math.floor(1000000 + Math.random() * 500000),
        perSecond: Math.floor(1200 + Math.random() * 800),
        averageExecutionTime: 25 + Math.random() * 75,
        slowQueries: Math.floor(Math.random() * 10),
      },
      cache: {
        hitRate: 85 + Math.random() * 12,
        size: Math.floor(2 * 1024 * 1024 * 1024), // 2GB
        evictions: Math.floor(Math.random() * 100),
      },
      replication: {
        lag: Math.random() * 50,
        status: Math.random() > 0.9 ? 'warning' : 'healthy',
      },
      load: generateDbSeries(),
    },
    services: [
      {
        name: 'Camera AI Service',
        status: Math.random() > 0.1 ? 'healthy' : 'degraded',
        uptime: 98.5 + Math.random() * 1.4,
        responseTime: 45 + Math.random() * 30,
        lastCheck: now - Math.random() * 60000,
        endpoint: '/health/camera-ai',
        dependencies: ['database', 'redis', 'ml-service'],
      },
      {
        name: 'Analytics API',
        status: Math.random() > 0.05 ? 'healthy' : 'degraded',
        uptime: 99.2 + Math.random() * 0.7,
        responseTime: 25 + Math.random() * 20,
        lastCheck: now - Math.random() * 60000,
        endpoint: '/health/analytics',
        dependencies: ['database', 'websocket'],
      },
      {
        name: 'WebSocket Server',
        status: Math.random() > 0.15 ? 'healthy' : 'degraded',
        uptime: 97.8 + Math.random() * 2.1,
        responseTime: 15 + Math.random() * 15,
        lastCheck: now - Math.random() * 60000,
        endpoint: '/health/websocket',
        dependencies: ['redis', 'kafka'],
      },
      {
        name: 'ML Inference',
        status: Math.random() > 0.2 ? 'healthy' : Math.random() > 0.5 ? 'degraded' : 'unhealthy',
        uptime: 96.5 + Math.random() * 3.4,
        responseTime: 120 + Math.random() * 80,
        lastCheck: now - Math.random() * 60000,
        endpoint: '/health/ml-service',
        dependencies: ['gpu-cluster', 'model-registry'],
      },
      {
        name: 'Database Cluster',
        status: Math.random() > 0.05 ? 'healthy' : 'warning',
        uptime: 99.8 + Math.random() * 0.2,
        responseTime: 8 + Math.random() * 12,
        lastCheck: now - Math.random() * 60000,
        endpoint: '/health/database',
        dependencies: [],
      },
    ],
    alerts: Array.from({ length: Math.floor(Math.random() * 8) }, (_, i) => ({
      id: `alert-${i + 1}`,
      type: ['cpu', 'memory', 'disk', 'network', 'api', 'database', 'service'][Math.floor(Math.random() * 7)] as any,
      severity: ['info', 'warning', 'critical'][Math.floor(Math.random() * 3)] as any,
      title: `System Alert ${i + 1}`,
      description: `Alert description for incident ${i + 1}`,
      timestamp: now - Math.random() * 3600000,
      resolved: Math.random() > 0.7,
    })),
    timestamp: now,
  };
};

export const SystemMetricsDashboard: React.FC<SystemMetricsDashboardProps> = ({
  data: externalData,
  className,
  showControls = true,
  autoRefresh = true,
  refreshInterval = 10000,
  onDataExport,
  onAlertAction,
  selectedTimeRange = '1h',
}) => {
  const [isPending, startTransition] = useTransition();
  const [lastRefresh, setLastRefresh] = useState(Date.now());
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState(selectedTimeRange);
  const [selectedService, setSelectedService] = useState<string | null>(null);
  const [showResolvedAlerts, setShowResolvedAlerts] = useState(false);

  // Use real system health data
  const systemHealth = useSystemHealth();
  const { isConnected, processingHealth } = useRealTimeMetrics();

  // Generate or use provided data
  const data = useMemo(() => {
    if (externalData) return externalData;

    // Convert real system health data to dashboard format if available
    if (systemHealth) {
      const mockData = generateMockSystemData();
      return {
        ...mockData,
        cpu: {
          ...mockData.cpu,
          usage: systemHealth.cpuUsage,
        },
        memory: {
          ...mockData.memory,
          usage: systemHealth.memoryUsage,
        },
        network: {
          ...mockData.network,
          latency: systemHealth.networkLatency,
        },
      };
    }

    return generateMockSystemData();
  }, [externalData, systemHealth, lastRefresh]);

  const deferredData = useDeferredValue(data);

  // Calculate overall system health score
  const systemHealthScore = useMemo(() => {
    const weights = {
      cpu: 0.2,
      memory: 0.2,
      disk: 0.15,
      network: 0.15,
      api: 0.15,
      database: 0.1,
      services: 0.05,
    };

    const cpuScore = Math.max(100 - deferredData.cpu.usage, 0);
    const memoryScore = Math.max(100 - deferredData.memory.usage, 0);
    const diskScore = Math.max(100 - deferredData.disk.usage, 0);
    const networkScore = Math.max(100 - (deferredData.network.latency / 2), 0);
    const apiScore = Math.max(100 - (deferredData.api.errorRate * 20), 0);
    const dbScore = Math.max(100 - (deferredData.database.connections.active / deferredData.database.connections.max * 100), 0);
    const servicesScore = (deferredData.services.filter(s => s.status === 'healthy').length / deferredData.services.length) * 100;

    return Math.round(
      cpuScore * weights.cpu +
      memoryScore * weights.memory +
      diskScore * weights.disk +
      networkScore * weights.network +
      apiScore * weights.api +
      dbScore * weights.database +
      servicesScore * weights.services
    );
  }, [deferredData]);

  // Filter alerts
  const filteredAlerts = useMemo(() => {
    return deferredData.alerts.filter(alert =>
      showResolvedAlerts || !alert.resolved
    );
  }, [deferredData.alerts, showResolvedAlerts]);

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

  // Alert action handler
  const handleAlertAction = useCallback((alertId: string, action: 'acknowledge' | 'resolve') => {
    console.log(`Alert action: ${action} on ${alertId}`);
    onAlertAction?.(alertId, action);
  }, [onAlertAction]);

  // Time range change handler
  const handleTimeRangeChange = useCallback((range: '5m' | '15m' | '1h' | '6h' | '24h') => {
    setTimeRange(range);
    handleRefresh();
  }, [handleRefresh]);

  // Format bytes
  const formatBytes = useCallback((bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }, []);

  // Metric card configurations
  const healthScoreMetric: MetricCardData = {
    value: systemHealthScore,
    unit: '%',
    format: 'number',
    status: systemHealthScore > 85 ? 'good' : systemHealthScore > 60 ? 'warning' : 'critical',
    metadata: {
      description: 'Overall system health score across all metrics',
      confidence: 0.95,
    },
  };

  const cpuMetric: MetricCardData = {
    value: deferredData.cpu.usage,
    unit: '%',
    format: 'number',
    status: deferredData.cpu.usage < 70 ? 'good' :
           deferredData.cpu.usage < 85 ? 'warning' : 'critical',
    timeSeries: deferredData.cpu.load,
    metadata: {
      description: `${deferredData.cpu.cores} cores @ ${deferredData.cpu.frequency.toFixed(1)}GHz`,
    },
  };

  const memoryMetric: MetricCardData = {
    value: deferredData.memory.usage,
    unit: '%',
    format: 'number',
    status: deferredData.memory.usage < 80 ? 'good' :
           deferredData.memory.usage < 90 ? 'warning' : 'critical',
    timeSeries: deferredData.memory.load,
    metadata: {
      description: `${formatBytes(deferredData.memory.used)} / ${formatBytes(deferredData.memory.total)}`,
    },
  };

  const apiMetric: MetricCardData = {
    value: deferredData.api.averageResponseTime,
    unit: 'ms',
    format: 'number',
    status: deferredData.api.averageResponseTime < 100 ? 'good' :
           deferredData.api.averageResponseTime < 500 ? 'warning' : 'critical',
    timeSeries: deferredData.api.load.map(l => ({
      timestamp: l.timestamp,
      value: l.responseTime,
    })),
    metadata: {
      description: `${Math.round(deferredData.api.requestsPerSecond)} req/s | ${deferredData.api.errorRate.toFixed(2)}% errors`,
    },
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">System Metrics</h2>
          <p className="text-muted-foreground">
            Resource utilization and performance monitoring
          </p>
        </div>

        {showControls && (
          <div className="flex items-center space-x-2">
            {/* Time range selector */}
            <Select value={timeRange} onValueChange={handleTimeRangeChange}>
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="5m">5M</SelectItem>
                <SelectItem value="15m">15M</SelectItem>
                <SelectItem value="1h">1H</SelectItem>
                <SelectItem value="6h">6H</SelectItem>
                <SelectItem value="24h">24H</SelectItem>
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
          title="System Health"
          subtitle="Overall health score"
          data={healthScoreMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="CPU Usage"
          subtitle="Processor utilization"
          data={cpuMetric}
          size="medium"
          showSparkline={true}
        />

        <MetricCard
          title="Memory Usage"
          subtitle="RAM utilization"
          data={memoryMetric}
          size="medium"
          showSparkline={true}
        />

        <MetricCard
          title="API Response Time"
          subtitle="Average latency"
          data={apiMetric}
          size="medium"
          showSparkline={true}
        />
      </div>

      {/* Tabbed Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="resources">Resources</TabsTrigger>
          <TabsTrigger value="services">Services</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* System Resource Summary */}
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold">Resource Summary</h3>
                  <p className="text-sm text-muted-foreground">Current utilization</p>
                </div>
                <IconServer className="h-8 w-8 text-primary" />
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>CPU</span>
                    <span>{deferredData.cpu.usage.toFixed(1)}%</span>
                  </div>
                  <Progress value={deferredData.cpu.usage} variant={
                    deferredData.cpu.usage > 85 ? 'destructive' :
                    deferredData.cpu.usage > 70 ? 'warning' : 'success'
                  } />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Memory</span>
                    <span>{deferredData.memory.usage.toFixed(1)}%</span>
                  </div>
                  <Progress value={deferredData.memory.usage} variant={
                    deferredData.memory.usage > 90 ? 'destructive' :
                    deferredData.memory.usage > 80 ? 'warning' : 'success'
                  } />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Disk</span>
                    <span>{deferredData.disk.usage.toFixed(1)}%</span>
                  </div>
                  <Progress value={deferredData.disk.usage} variant={
                    deferredData.disk.usage > 90 ? 'destructive' :
                    deferredData.disk.usage > 80 ? 'warning' : 'success'
                  } />
                </div>
              </div>
            </Card>

            {/* API Performance */}
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold">API Performance</h3>
                  <p className="text-sm text-muted-foreground">Request handling</p>
                </div>
                <IconApi className="h-8 w-8 text-primary" />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm">Requests/sec</span>
                  <span className="font-medium">{Math.round(deferredData.api.requestsPerSecond)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Avg Response</span>
                  <span className="font-medium">{Math.round(deferredData.api.averageResponseTime)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Error Rate</span>
                  <span className={cn(
                    'font-medium',
                    deferredData.api.errorRate > 5 ? 'text-destructive' :
                    deferredData.api.errorRate > 1 ? 'text-warning' : 'text-success'
                  )}>
                    {deferredData.api.errorRate.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">P95 Response</span>
                  <span className="font-medium">{Math.round(deferredData.api.p95ResponseTime)}ms</span>
                </div>
              </div>
            </Card>

            {/* Database Health */}
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold">Database Health</h3>
                  <p className="text-sm text-muted-foreground">Connection & performance</p>
                </div>
                <IconDatabase className="h-8 w-8 text-primary" />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm">Active Connections</span>
                  <span className="font-medium">
                    {deferredData.database.connections.active}/{deferredData.database.connections.max}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Queries/sec</span>
                  <span className="font-medium">{Math.round(deferredData.database.queries.perSecond)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Avg Exec Time</span>
                  <span className="font-medium">{Math.round(deferredData.database.queries.averageExecutionTime)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Cache Hit Rate</span>
                  <span className="font-medium">{deferredData.database.cache.hitRate.toFixed(1)}%</span>
                </div>
              </div>
            </Card>
          </div>

          {/* Performance Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DashboardWidget
              config={{
                id: 'cpu-memory-usage',
                title: 'CPU & Memory Usage',
                subtitle: 'Resource utilization over time',
                size: 'large',
                priority: 'high',
                category: 'system',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.cpu.load,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.cpu.load}
                height={300}
                gradientColors={['hsl(var(--destructive))', 'transparent']}
                strokeColor="hsl(var(--destructive))"
                formatValue={(value) => `${Math.round(value)}%`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <DashboardWidget
              config={{
                id: 'api-performance',
                title: 'API Response Times',
                subtitle: 'Request latency trends',
                size: 'large',
                priority: 'medium',
                category: 'api',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.api.load,
              }}
            >
              <RealTimeLineChart
                data={deferredData.api.load.map(l => ({
                  timestamp: l.timestamp,
                  value: l.responseTime,
                }))}
                height={300}
                color="primary"
                formatValue={(value) => `${Math.round(value)}ms`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>
          </div>
        </TabsContent>

        {/* Resources Tab */}
        <TabsContent value="resources" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DashboardWidget
              config={{
                id: 'network-throughput',
                title: 'Network Throughput',
                subtitle: 'Inbound and outbound traffic',
                size: 'large',
                priority: 'medium',
                category: 'network',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.network.load,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.network.load.map(l => ({
                  timestamp: l.timestamp,
                  value: l.inbound + l.outbound,
                }))}
                height={300}
                gradientColors={['hsl(var(--success))', 'transparent']}
                strokeColor="hsl(var(--success))"
                formatValue={(value) => `${Math.round(value)} MB/s`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <DashboardWidget
              config={{
                id: 'database-performance',
                title: 'Database Performance',
                subtitle: 'Query execution and connections',
                size: 'large',
                priority: 'medium',
                category: 'database',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.database.load,
              }}
            >
              <RealTimeLineChart
                data={deferredData.database.load.map(l => ({
                  timestamp: l.timestamp,
                  value: l.executionTime,
                }))}
                height={300}
                color="warning"
                formatValue={(value) => `${Math.round(value)}ms`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>
          </div>

          {/* Detailed Resource Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <IconCpu className="h-5 w-5 mr-2 text-destructive" />
                CPU Details
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Cores:</span>
                  <span className="font-medium">{deferredData.cpu.cores}</span>
                </div>
                <div className="flex justify-between">
                  <span>Frequency:</span>
                  <span className="font-medium">{deferredData.cpu.frequency.toFixed(1)} GHz</span>
                </div>
                <div className="flex justify-between">
                  <span>Temperature:</span>
                  <span className={cn(
                    'font-medium',
                    deferredData.cpu.temperature > 80 ? 'text-destructive' :
                    deferredData.cpu.temperature > 65 ? 'text-warning' : 'text-success'
                  )}>
                    {Math.round(deferredData.cpu.temperature)}°C
                  </span>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <IconMemory className="h-5 w-5 mr-2 text-primary" />
                Memory Details
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Total:</span>
                  <span className="font-medium">{formatBytes(deferredData.memory.total)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Used:</span>
                  <span className="font-medium">{formatBytes(deferredData.memory.used)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Available:</span>
                  <span className="font-medium">{formatBytes(deferredData.memory.available)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Cache:</span>
                  <span className="font-medium">{formatBytes(deferredData.memory.cache)}</span>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <IconActivity className="h-5 w-5 mr-2 text-success" />
                Disk I/O
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Read Speed:</span>
                  <span className="font-medium">{Math.round(deferredData.disk.readSpeed)} MB/s</span>
                </div>
                <div className="flex justify-between">
                  <span>Write Speed:</span>
                  <span className="font-medium">{Math.round(deferredData.disk.writeSpeed)} MB/s</span>
                </div>
                <div className="flex justify-between">
                  <span>IOPS:</span>
                  <span className="font-medium">{deferredData.disk.iops.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Usage:</span>
                  <span className="font-medium">
                    {formatBytes(deferredData.disk.used)} / {formatBytes(deferredData.disk.total)}
                  </span>
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Services Tab */}
        <TabsContent value="services" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {deferredData.services.map((service) => (
              <Card key={service.name} className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {service.status === 'healthy' ? (
                      <IconCheck className="h-4 w-4 text-success" />
                    ) : service.status === 'degraded' ? (
                      <IconAlertTriangle className="h-4 w-4 text-warning" />
                    ) : (
                      <IconX className="h-4 w-4 text-destructive" />
                    )}
                    <span className="font-medium">{service.name}</span>
                  </div>
                  <Badge
                    variant={service.status === 'healthy' ? 'default' :
                            service.status === 'degraded' ? 'secondary' : 'destructive'}
                    className={cn(statusConfig[service.status].color)}
                  >
                    {statusConfig[service.status].label}
                  </Badge>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Uptime:</span>
                    <span className="font-medium">{service.uptime.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Response Time:</span>
                    <span className={cn(
                      'font-medium',
                      service.responseTime > 200 ? 'text-destructive' :
                      service.responseTime > 100 ? 'text-warning' : 'text-success'
                    )}>
                      {Math.round(service.responseTime)}ms
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Last Check:</span>
                    <span className="font-medium">
                      {Math.round((Date.now() - service.lastCheck) / 1000)}s ago
                    </span>
                  </div>

                  {service.dependencies.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs text-muted-foreground mb-1">Dependencies:</div>
                      <div className="flex flex-wrap gap-1">
                        {service.dependencies.map((dep) => (
                          <Badge key={dep} variant="outline" className="text-xs">
                            {dep}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Switch
                checked={showResolvedAlerts}
                onCheckedChange={setShowResolvedAlerts}
              />
              <Label className="text-sm">Show resolved alerts</Label>
            </div>

            <div className="text-sm text-muted-foreground">
              {filteredAlerts.length} alerts
            </div>
          </div>

          <div className="space-y-4">
            {filteredAlerts.map((alert) => (
              <Card key={alert.id} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <Badge
                        variant={alert.severity === 'critical' ? 'destructive' :
                                alert.severity === 'warning' ? 'secondary' : 'outline'}
                      >
                        {alert.severity}
                      </Badge>
                      <Badge variant="outline">{alert.type}</Badge>
                      {alert.resolved && (
                        <Badge variant="outline" className="text-success">
                          Resolved
                        </Badge>
                      )}
                    </div>

                    <h4 className="font-medium mb-1">{alert.title}</h4>
                    <p className="text-sm text-muted-foreground mb-2">{alert.description}</p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(alert.timestamp).toLocaleString()}
                    </p>
                  </div>

                  {!alert.resolved && (
                    <div className="flex space-x-2 ml-4">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleAlertAction(alert.id, 'acknowledge')}
                      >
                        Acknowledge
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleAlertAction(alert.id, 'resolve')}
                      >
                        Resolve
                      </Button>
                    </div>
                  )}
                </div>
              </Card>
            ))}

            {filteredAlerts.length === 0 && (
              <Card className="p-8 text-center">
                <IconCheck className="h-12 w-12 text-success mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No Active Alerts</h3>
                <p className="text-muted-foreground">
                  All systems are operating normally
                </p>
              </Card>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SystemMetricsDashboard;
