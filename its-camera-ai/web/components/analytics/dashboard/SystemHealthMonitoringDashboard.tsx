/**
 * System Health Monitoring Dashboard Component
 *
 * Comprehensive system health monitoring with infrastructure metrics,
 * performance indicators, resource usage, and alert management.
 */

'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { useDeferredValue, useTransition } from 'react';
import { cn } from '@/lib/utils';
import {
  IconCpu,
  IconMemory,
  IconDatabase,
  IconServer,
  IconWifi,
  IconHardDrive,
  IconBolt,
  IconThermometer,
  IconActivity,
  IconAlertTriangle,
  IconCheckCircle,
  IconX,
  IconRefresh,
  IconDownload,
  IconShield,
  IconCloud,
  IconChevronUp,
  IconChevronDown
} from '@tabler/icons-react';
import { MetricCard, MetricCardData } from './MetricCard';
import { DashboardWidget, WidgetConfig } from './DashboardWidget';
import { RealTimeLineChart } from '@/components/charts/RealTimeLineChart';
import { RealTimeAreaChart } from '@/components/charts/RealTimeAreaChart';
import { RealTimeBarChart, BarChartData } from '@/components/charts/RealTimeBarChart';
import { useAnalyticsStore } from '@/stores/analytics';
import { useRealTimeMetrics } from '@/providers/RealTimeProvider';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';

export interface SystemComponent {
  id: string;
  name: string;
  type: 'server' | 'database' | 'network' | 'storage' | 'application';
  status: 'healthy' | 'warning' | 'critical' | 'maintenance' | 'offline';
  health: {
    score: number; // 0-100
    lastChecked: number;
    uptime: number; // seconds
    responseTime: number; // ms
  };
  resources: {
    cpu: {
      usage: number; // percentage
      cores: number;
      temperature?: number; // Celsius
    };
    memory: {
      used: number; // GB
      total: number; // GB
      usage: number; // percentage
    };
    disk: {
      used: number; // GB
      total: number; // GB
      usage: number; // percentage
      iops?: number;
    };
    network: {
      bytesIn: number; // per second
      bytesOut: number; // per second
      connections: number;
      latency: number; // ms
    };
  };
  metrics: {
    requests: number; // per second
    errors: number; // per second
    errorRate: number; // percentage
    availability: number; // percentage
  };
  alerts: Array<{
    id: string;
    severity: 'info' | 'warning' | 'critical';
    message: string;
    timestamp: number;
    acknowledged?: boolean;
  }>;
}

export interface SystemHealthData {
  components: SystemComponent[];
  overallHealth: {
    score: number;
    status: 'healthy' | 'degraded' | 'critical';
    timestamp: number;
  };
  resourceUsage: {
    cpu: { value: number; trend: Array<{ timestamp: number; value: number }> };
    memory: { value: number; trend: Array<{ timestamp: number; value: number }> };
    disk: { value: number; trend: Array<{ timestamp: number; value: number }> };
    network: { value: number; trend: Array<{ timestamp: number; value: number }> };
  };
  performance: {
    responseTime: { value: number; trend: Array<{ timestamp: number; value: number }> };
    throughput: { value: number; trend: Array<{ timestamp: number; value: number }> };
    errorRate: { value: number; trend: Array<{ timestamp: number; value: number }> };
    availability: { value: number; trend: Array<{ timestamp: number; value: number }> };
  };
  infrastructure: {
    totalServers: number;
    activeServers: number;
    totalStorage: number; // TB
    usedStorage: number; // TB
    bandwidth: number; // Gbps
    connections: number;
  };
  alerts: {
    total: number;
    critical: number;
    warning: number;
    info: number;
    unacknowledged: number;
  };
}

export interface SystemHealthMonitoringDashboardProps {
  data?: SystemHealthData;
  className?: string;
  showControls?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onComponentSelect?: (component: SystemComponent) => void;
  onAlertAcknowledge?: (alertId: string) => void;
  onDataExport?: () => void;
}

// Component status configurations
const statusConfig = {
  healthy: { color: 'text-success', bgColor: 'bg-success/20', label: 'Healthy', icon: IconCheckCircle },
  warning: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Warning', icon: IconAlertTriangle },
  critical: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Critical', icon: IconX },
  maintenance: { color: 'text-info', bgColor: 'bg-info/20', label: 'Maintenance', icon: IconActivity },
  offline: { color: 'text-muted-foreground', bgColor: 'bg-muted/20', label: 'Offline', icon: IconServer },
};

// Component type configurations
const typeConfig = {
  server: { icon: IconServer, label: 'Server' },
  database: { icon: IconDatabase, label: 'Database' },
  network: { icon: IconWifi, label: 'Network' },
  storage: { icon: IconHardDrive, label: 'Storage' },
  application: { icon: IconActivity, label: 'Application' },
};

// Generate mock system health data
const generateMockSystemHealthData = (): SystemHealthData => {
  const now = Date.now();

  const components: SystemComponent[] = [
    {
      id: 'web-server-01',
      name: 'Web Server 01',
      type: 'server',
      status: 'healthy',
      health: {
        score: 95,
        lastChecked: now - 30000,
        uptime: 2592000, // 30 days
        responseTime: 45,
      },
      resources: {
        cpu: { usage: 35, cores: 8, temperature: 45 },
        memory: { used: 12.5, total: 32, usage: 39 },
        disk: { used: 450, total: 1000, usage: 45, iops: 1200 },
        network: { bytesIn: 1500000, bytesOut: 2300000, connections: 450, latency: 15 },
      },
      metrics: {
        requests: 2500,
        errors: 12,
        errorRate: 0.48,
        availability: 99.95,
      },
      alerts: [],
    },
    {
      id: 'db-primary',
      name: 'Primary Database',
      type: 'database',
      status: 'warning',
      health: {
        score: 78,
        lastChecked: now - 15000,
        uptime: 1728000, // 20 days
        responseTime: 125,
      },
      resources: {
        cpu: { usage: 68, cores: 16, temperature: 58 },
        memory: { used: 78, total: 128, usage: 61 },
        disk: { used: 3200, total: 5000, usage: 64, iops: 4500 },
        network: { bytesIn: 5200000, bytesOut: 3800000, connections: 1200, latency: 8 },
      },
      metrics: {
        requests: 8500,
        errors: 45,
        errorRate: 0.53,
        availability: 99.87,
      },
      alerts: [
        {
          id: 'db-high-cpu',
          severity: 'warning',
          message: 'CPU usage above 65% for 15 minutes',
          timestamp: now - 900000,
          acknowledged: false,
        },
      ],
    },
    {
      id: 'load-balancer',
      name: 'Load Balancer',
      type: 'network',
      status: 'healthy',
      health: {
        score: 92,
        lastChecked: now - 45000,
        uptime: 5184000, // 60 days
        responseTime: 8,
      },
      resources: {
        cpu: { usage: 25, cores: 4 },
        memory: { used: 3.2, total: 8, usage: 40 },
        disk: { used: 25, total: 100, usage: 25 },
        network: { bytesIn: 25000000, bytesOut: 24500000, connections: 5000, latency: 2 },
      },
      metrics: {
        requests: 15000,
        errors: 8,
        errorRate: 0.05,
        availability: 99.99,
      },
      alerts: [],
    },
  ];

  return {
    components,
    overallHealth: {
      score: 88,
      status: 'healthy',
      timestamp: now,
    },
    resourceUsage: {
      cpu: {
        value: 42,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 20 + 30,
        })),
      },
      memory: {
        value: 48,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 15 + 40,
        })),
      },
      disk: {
        value: 52,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 10 + 45,
        })),
      },
      network: {
        value: 35,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 20 + 25,
        })),
      },
    },
    performance: {
      responseTime: {
        value: 85,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 40 + 60,
        })),
      },
      throughput: {
        value: 12500,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 5000 + 10000,
        })),
      },
      errorRate: {
        value: 0.42,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 0.5 + 0.2,
        })),
      },
      availability: {
        value: 99.92,
        trend: Array.from({ length: 20 }, (_, i) => ({
          timestamp: now - (19 - i) * 30000,
          value: Math.random() * 0.2 + 99.8,
        })),
      },
    },
    infrastructure: {
      totalServers: 12,
      activeServers: 11,
      totalStorage: 50,
      usedStorage: 32,
      bandwidth: 10,
      connections: 8500,
    },
    alerts: {
      total: 7,
      critical: 1,
      warning: 3,
      info: 3,
      unacknowledged: 4,
    },
  };
};

export const SystemHealthMonitoringDashboard: React.FC<SystemHealthMonitoringDashboardProps> = ({
  data: externalData,
  className,
  showControls = true,
  autoRefresh = true,
  refreshInterval = 30000,
  onComponentSelect,
  onAlertAcknowledge,
  onDataExport,
}) => {
  const [isPending, startTransition] = useTransition();
  const [lastRefresh, setLastRefresh] = useState(Date.now());

  // Use real analytics data or mock data
  const analyticsData = useAnalyticsStore(state => state.currentMetrics);
  const { isConnected, metrics, processingHealth } = useRealTimeMetrics();

  // Generate or use provided data
  const data = useMemo(() => {
    if (externalData) return externalData;
    return generateMockSystemHealthData();
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

  // Metric card configurations
  const overallHealthMetric: MetricCardData = {
    value: deferredData.overallHealth.score,
    unit: '/100',
    format: 'number',
    status: deferredData.overallHealth.score > 85 ? 'good' :
           deferredData.overallHealth.score > 70 ? 'warning' : 'critical',
    metadata: {
      description: 'Overall system health score',
    },
  };

  const activeServersMetric: MetricCardData = {
    value: deferredData.infrastructure.activeServers,
    previousValue: deferredData.infrastructure.totalServers,
    unit: 'servers',
    format: 'number',
    status: deferredData.infrastructure.activeServers / deferredData.infrastructure.totalServers < 0.9 ? 'critical' : 'good',
    metadata: {
      description: 'Active servers out of total deployed',
    },
  };

  const responseTimeMetric: MetricCardData = {
    value: deferredData.performance.responseTime.value,
    target: 100,
    unit: 'ms',
    format: 'number',
    status: deferredData.performance.responseTime.value > 200 ? 'critical' :
           deferredData.performance.responseTime.value > 100 ? 'warning' : 'good',
    timeSeries: deferredData.performance.responseTime.trend,
    metadata: {
      description: 'Average response time across all services',
    },
  };

  const availabilityMetric: MetricCardData = {
    value: deferredData.performance.availability.value,
    target: 99.9,
    unit: '%',
    format: 'percentage',
    status: deferredData.performance.availability.value < 99.5 ? 'critical' :
           deferredData.performance.availability.value < 99.8 ? 'warning' : 'good',
    timeSeries: deferredData.performance.availability.trend,
    metadata: {
      description: 'System availability percentage',
    },
  };

  // Resource usage chart data
  const resourceChartData: BarChartData[] = [
    { category: 'CPU', value: deferredData.resourceUsage.cpu.value, timestamp: Date.now() },
    { category: 'Memory', value: deferredData.resourceUsage.memory.value, timestamp: Date.now() },
    { category: 'Disk', value: deferredData.resourceUsage.disk.value, timestamp: Date.now() },
    { category: 'Network', value: deferredData.resourceUsage.network.value, timestamp: Date.now() },
  ];

  return (
    <div className={cn('space-y-6', className)}>
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">System Health</h2>
          <p className="text-muted-foreground">
            Real-time infrastructure monitoring and performance metrics
          </p>
        </div>

        {showControls && (
          <div className="flex items-center space-x-2">
            {/* Connection status */}
            <Badge variant={isConnected ? 'default' : 'destructive'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </Badge>

            {/* System status */}
            <Badge
              variant={deferredData.overallHealth.status === 'healthy' ? 'default' : 'destructive'}
            >
              {deferredData.overallHealth.status}
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
          title="Overall Health"
          subtitle="System health score"
          data={overallHealthMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="Active Servers"
          subtitle="Online infrastructure"
          data={activeServersMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="Response Time"
          subtitle="Average latency"
          data={responseTimeMetric}
          size="medium"
          showSparkline={true}
          showTarget={true}
        />

        <MetricCard
          title="Availability"
          subtitle="System uptime"
          data={availabilityMetric}
          size="medium"
          showSparkline={true}
          showTarget={true}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Resource Usage Charts */}
        <div className="lg:col-span-2 space-y-6">
          {/* CPU and Memory Trends */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <DashboardWidget
              config={{
                id: 'cpu-usage-chart',
                title: 'CPU Usage',
                subtitle: 'Real-time processor utilization',
                size: 'large',
                priority: 'medium',
                category: 'system',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.resourceUsage.cpu.trend,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.resourceUsage.cpu.trend}
                height={200}
                gradientColors={['hsl(var(--primary))', 'transparent']}
                strokeColor="hsl(var(--primary))"
                formatValue={(value) => `${value.toFixed(1)}%`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <DashboardWidget
              config={{
                id: 'memory-usage-chart',
                title: 'Memory Usage',
                subtitle: 'RAM utilization over time',
                size: 'large',
                priority: 'medium',
                category: 'system',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.resourceUsage.memory.trend,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.resourceUsage.memory.trend}
                height={200}
                gradientColors={['hsl(var(--success))', 'transparent']}
                strokeColor="hsl(var(--success))"
                formatValue={(value) => `${value.toFixed(1)}%`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>
          </div>

          {/* Resource Usage Overview */}
          <DashboardWidget
            config={{
              id: 'resource-overview-chart',
              title: 'Resource Overview',
              subtitle: 'Current utilization across all resources',
              size: 'xlarge',
              priority: 'medium',
              category: 'system',
            }}
            data={{
              timestamp: Date.now(),
              isLoading: false,
              data: resourceChartData,
            }}
          >
            <RealTimeBarChart
              data={resourceChartData}
              height={250}
              orientation="vertical"
              showValues={true}
              formatValue={(value) => `${value.toFixed(1)}%`}
              barColor="hsl(var(--primary))"
            />
          </DashboardWidget>
        </div>

        {/* System Components & Alerts */}
        <div className="space-y-6">
          {/* Component Status */}
          <DashboardWidget
            config={{
              id: 'component-status',
              title: 'Component Status',
              subtitle: `${deferredData.components.length} components monitored`,
              size: 'large',
              priority: 'high',
              category: 'system',
            }}
            data={{
              timestamp: Date.now(),
              isLoading: false,
              data: deferredData.components,
            }}
          >
            <ScrollArea className="h-[300px]">
              <div className="space-y-3">
                {deferredData.components.map((component) => {
                  const statusInfo = statusConfig[component.status];
                  const typeInfo = typeConfig[component.type];
                  const StatusIcon = statusInfo.icon;
                  const TypeIcon = typeInfo.icon;

                  return (
                    <Card
                      key={component.id}
                      className={cn(
                        'p-3 cursor-pointer transition-all hover:shadow-md',
                        'border-l-4',
                        statusInfo.bgColor.replace('/20', '/50')
                      )}
                      onClick={() => onComponentSelect?.(component)}
                    >
                      <div className="flex items-center space-x-3">
                        <div className={cn(
                          'p-2 rounded-lg',
                          statusInfo.bgColor
                        )}>
                          <TypeIcon className="h-4 w-4 text-foreground" />
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <h4 className="text-sm font-semibold truncate">
                              {component.name}
                            </h4>
                            <Badge
                              variant="outline"
                              className={cn('text-xs', statusInfo.color)}
                            >
                              {statusInfo.label}
                            </Badge>
                          </div>

                          <div className="flex items-center justify-between mt-1 text-xs text-muted-foreground">
                            <span>{typeInfo.label}</span>
                            <span>Health: {component.health.score}/100</span>
                          </div>

                          {/* Resource indicators */}
                          <div className="mt-2 space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span>CPU</span>
                              <span className={component.resources.cpu.usage > 80 ? 'text-warning' : ''}>
                                {component.resources.cpu.usage}%
                              </span>
                            </div>
                            <Progress
                              value={component.resources.cpu.usage}
                              className="h-1"
                            />
                          </div>

                          <div className="mt-1 space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span>Memory</span>
                              <span className={component.resources.memory.usage > 80 ? 'text-warning' : ''}>
                                {component.resources.memory.usage}%
                              </span>
                            </div>
                            <Progress
                              value={component.resources.memory.usage}
                              className="h-1"
                            />
                          </div>
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            </ScrollArea>
          </DashboardWidget>

          {/* Recent Alerts */}
          <DashboardWidget
            config={{
              id: 'system-alerts',
              title: 'System Alerts',
              subtitle: `${deferredData.alerts.unacknowledged} unacknowledged`,
              size: 'large',
              priority: 'critical',
              category: 'system',
            }}
            data={{
              timestamp: Date.now(),
              isLoading: false,
              data: deferredData.alerts,
            }}
          >
            {/* Alert summary */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="text-center">
                <div className="text-lg font-bold text-destructive">
                  {deferredData.alerts.critical}
                </div>
                <div className="text-xs text-muted-foreground">Critical</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-warning">
                  {deferredData.alerts.warning}
                </div>
                <div className="text-xs text-muted-foreground">Warning</div>
              </div>
            </div>

            {/* Alert list */}
            <ScrollArea className="h-[200px]">
              <div className="space-y-2">
                {deferredData.components
                  .flatMap(c => c.alerts)
                  .filter(alert => !alert.acknowledged)
                  .sort((a, b) => b.timestamp - a.timestamp)
                  .slice(0, 10)
                  .map((alert) => (
                    <div
                      key={alert.id}
                      className={cn(
                        'p-2 rounded border-l-4 text-xs',
                        alert.severity === 'critical' && 'border-destructive bg-destructive/10',
                        alert.severity === 'warning' && 'border-warning bg-warning/10',
                        alert.severity === 'info' && 'border-info bg-info/10'
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{alert.message}</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                          onClick={() => onAlertAcknowledge?.(alert.id)}
                        >
                          <IconCheckCircle className="h-3 w-3" />
                        </Button>
                      </div>
                      <div className="text-muted-foreground mt-1">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}

                {deferredData.components.flatMap(c => c.alerts).filter(a => !a.acknowledged).length === 0 && (
                  <div className="text-center py-4 text-muted-foreground">
                    <IconCheckCircle className="h-6 w-6 mx-auto mb-2" />
                    <p>No unacknowledged alerts</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </DashboardWidget>
        </div>
      </div>

      {/* Processing Health (if available) */}
      {processingHealth && (
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-3">Event Processing Performance</h3>
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

export default SystemHealthMonitoringDashboard;
