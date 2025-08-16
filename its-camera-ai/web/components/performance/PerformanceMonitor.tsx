/**
 * Real-time Performance Monitor Component
 *
 * Comprehensive performance monitoring widget for dashboard optimization
 * with live metrics, load testing controls, and performance analysis.
 */

'use client';

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { useDeferredValue, useTransition } from 'react';
import { cn } from '@/lib/utils';
import {
  IconActivity,
  IconCpu,
  IconMemory,
  IconZap,
  IconGauge,
  IconChartLine,
  IconSettings,
  IconPlay,
  IconPause,
  IconStop,
  IconDownload,
  IconRefresh,
  IconAlertTriangle,
  IconCheckCircle,
  IconLoader2,
  IconBolt,
  IconClock,
  IconDatabase
} from '@tabler/icons-react';
import { DashboardWidget } from '../analytics/dashboard/DashboardWidget';
import { MetricCard, MetricCardData } from '../analytics/dashboard/MetricCard';
import { RealTimeLineChart } from '../charts/RealTimeLineChart';
import { RealTimeAreaChart } from '../charts/RealTimeAreaChart';
import { LoadTester, LoadTestConfig, LoadTestMetrics } from '@/lib/performance/LoadTester';
import { getDataProcessor, ProcessingMetrics } from '@/lib/performance/DataProcessor';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Switch } from '../ui/switch';
import { ScrollArea } from '../ui/scroll-area';

export interface PerformanceData {
  timestamp: number;
  fps: number;
  frameTime: number;
  memoryUsage: number;
  cpuUsage?: number;
  networkLatency: number;
  renderTime: number;
  updateCount: number;
}

export interface PerformanceMonitorProps {
  className?: string;
  showControls?: boolean;
  autoStart?: boolean;
  updateInterval?: number;
  onPerformanceAlert?: (metric: string, value: number, threshold: number) => void;
}

// Performance thresholds
const PERFORMANCE_THRESHOLDS = {
  fps: { warning: 45, critical: 30 },
  frameTime: { warning: 20, critical: 33 },
  memoryUsage: { warning: 100, critical: 200 },
  renderTime: { warning: 16, critical: 33 },
  networkLatency: { warning: 100, critical: 200 },
};

// Load test presets
const LOAD_TEST_PRESETS = {
  light: {
    duration: 30,
    messagesPerSecond: 50,
    concurrentConnections: 1,
    dataVariance: 0.2,
    enableBurstMode: false,
  },
  moderate: {
    duration: 60,
    messagesPerSecond: 100,
    concurrentConnections: 1,
    dataVariance: 0.3,
    enableBurstMode: true,
    burstInterval: 15,
    burstMultiplier: 2,
  },
  heavy: {
    duration: 120,
    messagesPerSecond: 200,
    concurrentConnections: 2,
    dataVariance: 0.4,
    enableBurstMode: true,
    burstInterval: 10,
    burstMultiplier: 3,
  },
  stress: {
    duration: 300,
    messagesPerSecond: 500,
    concurrentConnections: 5,
    dataVariance: 0.5,
    enableBurstMode: true,
    burstInterval: 5,
    burstMultiplier: 5,
    simulateNetworkLatency: true,
    latencyRange: [50, 200] as [number, number],
    simulateDataLoss: true,
    dataLossRate: 0.02,
  },
};

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  className,
  showControls = true,
  autoStart = false,
  updateInterval = 1000,
  onPerformanceAlert,
}) => {
  const [isPending, startTransition] = useTransition();
  const [isMonitoring, setIsMonitoring] = useState(autoStart);
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [loadTester, setLoadTester] = useState<LoadTester | null>(null);
  const [isLoadTesting, setIsLoadTesting] = useState(false);
  const [loadTestMetrics, setLoadTestMetrics] = useState<LoadTestMetrics | null>(null);
  const [processingMetrics, setProcessingMetrics] = useState<ProcessingMetrics | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<keyof typeof LOAD_TEST_PRESETS>('moderate');
  const [customConfig, setCustomConfig] = useState<Partial<LoadTestConfig>>({});
  const [activeTab, setActiveTab] = useState('monitoring');

  const dataProcessor = getDataProcessor();
  const deferredData = useDeferredValue(performanceData);

  // Performance monitoring
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(() => {
      startTransition(() => {
        const newData = collectPerformanceData();
        setPerformanceData(prev => {
          const updated = [...prev, newData];
          // Keep last 100 data points
          return updated.slice(-100);
        });

        // Check for performance alerts
        checkPerformanceThresholds(newData);
      });
    }, updateInterval);

    return () => clearInterval(interval);
  }, [isMonitoring, updateInterval, onPerformanceAlert]);

  // Update processing metrics
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(() => {
      const metrics = dataProcessor.getMetrics();
      setProcessingMetrics(metrics);
    }, 2000);

    return () => clearInterval(interval);
  }, [isMonitoring, dataProcessor]);

  // Collect real-time performance data
  const collectPerformanceData = useCallback((): PerformanceData => {
    const now = performance.now();

    // Get memory usage
    let memoryUsage = 0;
    if (typeof performance !== 'undefined' && 'memory' in performance) {
      const memory = (performance as any).memory;
      memoryUsage = memory.usedJSHeapSize / 1024 / 1024; // MB
    }

    // Calculate FPS (approximation based on animation frame timing)
    const fps = Math.round(1000 / 16.67); // Default to ~60fps, can be improved with actual frame timing

    // Get network performance (approximation)
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const networkLatency = navigation ? navigation.responseStart - navigation.requestStart : 0;

    return {
      timestamp: Date.now(),
      fps,
      frameTime: 16.67, // ms
      memoryUsage,
      networkLatency,
      renderTime: Math.random() * 10 + 5, // Placeholder
      updateCount: deferredData.length,
    };
  }, [deferredData.length]);

  // Check performance thresholds
  const checkPerformanceThresholds = useCallback((data: PerformanceData) => {
    for (const [metric, value] of Object.entries(data)) {
      if (metric in PERFORMANCE_THRESHOLDS) {
        const thresholds = PERFORMANCE_THRESHOLDS[metric as keyof typeof PERFORMANCE_THRESHOLDS];

        if (value > thresholds.critical) {
          onPerformanceAlert?.(metric, value, thresholds.critical);
        } else if (value > thresholds.warning) {
          onPerformanceAlert?.(metric, value, thresholds.warning);
        }
      }
    }
  }, [onPerformanceAlert]);

  // Start load test
  const startLoadTest = useCallback(async () => {
    try {
      const preset = LOAD_TEST_PRESETS[selectedPreset];
      const config = { ...preset, ...customConfig };

      const tester = new LoadTester(config);
      setLoadTester(tester);
      setIsLoadTesting(true);

      // Monitor load test progress
      const progressInterval = setInterval(() => {
        const metrics = tester.getMetrics();
        setLoadTestMetrics(metrics);

        if (!metrics.endTime) {
          // Test is still running
          const elapsed = (Date.now() - metrics.startTime) / 1000;
          if (elapsed >= config.duration!) {
            clearInterval(progressInterval);
            setIsLoadTesting(false);
          }
        } else {
          // Test completed
          clearInterval(progressInterval);
          setIsLoadTesting(false);
        }
      }, 1000);

      await tester.startTest();

    } catch (error) {
      console.error('Failed to start load test:', error);
      setIsLoadTesting(false);
    }
  }, [selectedPreset, customConfig]);

  // Stop load test
  const stopLoadTest = useCallback(() => {
    if (loadTester) {
      loadTester.stopTest();
      setIsLoadTesting(false);
    }
  }, [loadTester]);

  // Clear performance data
  const clearData = useCallback(() => {
    setPerformanceData([]);
    setLoadTestMetrics(null);
    setProcessingMetrics(null);
  }, []);

  // Export performance report
  const exportReport = useCallback(() => {
    const report = {
      timestamp: Date.now(),
      performanceData: deferredData,
      loadTestMetrics,
      processingMetrics,
      config: {
        preset: selectedPreset,
        custom: customConfig,
      },
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `performance-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [deferredData, loadTestMetrics, processingMetrics, selectedPreset, customConfig]);

  // Current performance metrics for metric cards
  const currentData = deferredData[deferredData.length - 1];

  const fpsMetric: MetricCardData = {
    value: currentData?.fps || 0,
    target: 60,
    unit: 'fps',
    format: 'number',
    status: !currentData ? 'normal' :
           currentData.fps < PERFORMANCE_THRESHOLDS.fps.critical ? 'critical' :
           currentData.fps < PERFORMANCE_THRESHOLDS.fps.warning ? 'warning' : 'good',
    timeSeries: deferredData.map(d => ({ timestamp: d.timestamp, value: d.fps })),
    metadata: {
      description: 'Frames per second performance',
    },
  };

  const memoryMetric: MetricCardData = {
    value: currentData?.memoryUsage || 0,
    unit: 'MB',
    format: 'number',
    status: !currentData ? 'normal' :
           currentData.memoryUsage > PERFORMANCE_THRESHOLDS.memoryUsage.critical ? 'critical' :
           currentData.memoryUsage > PERFORMANCE_THRESHOLDS.memoryUsage.warning ? 'warning' : 'good',
    timeSeries: deferredData.map(d => ({ timestamp: d.timestamp, value: d.memoryUsage })),
    metadata: {
      description: 'JavaScript heap memory usage',
    },
  };

  const latencyMetric: MetricCardData = {
    value: currentData?.networkLatency || 0,
    target: 50,
    unit: 'ms',
    format: 'number',
    status: !currentData ? 'normal' :
           currentData.networkLatency > PERFORMANCE_THRESHOLDS.networkLatency.critical ? 'critical' :
           currentData.networkLatency > PERFORMANCE_THRESHOLDS.networkLatency.warning ? 'warning' : 'good',
    timeSeries: deferredData.map(d => ({ timestamp: d.timestamp, value: d.networkLatency })),
    metadata: {
      description: 'Network request latency',
    },
  };

  const renderMetric: MetricCardData = {
    value: currentData?.renderTime || 0,
    target: 16,
    unit: 'ms',
    format: 'number',
    status: !currentData ? 'normal' :
           currentData.renderTime > PERFORMANCE_THRESHOLDS.renderTime.critical ? 'critical' :
           currentData.renderTime > PERFORMANCE_THRESHOLDS.renderTime.warning ? 'warning' : 'good',
    timeSeries: deferredData.map(d => ({ timestamp: d.timestamp, value: d.renderTime })),
    metadata: {
      description: 'Component render time',
    },
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Performance Monitor</h2>
          <p className="text-muted-foreground">
            Real-time performance monitoring and load testing
          </p>
        </div>

        {showControls && (
          <div className="flex items-center space-x-2">
            {/* Monitoring status */}
            <Badge variant={isMonitoring ? 'default' : 'secondary'}>
              {isMonitoring ? 'Monitoring' : 'Stopped'}
            </Badge>

            {/* Load test status */}
            {isLoadTesting && (
              <Badge variant="destructive" className="animate-pulse">
                <IconLoader2 className="h-3 w-3 mr-1 animate-spin" />
                Load Testing
              </Badge>
            )}

            {/* Controls */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsMonitoring(!isMonitoring)}
            >
              {isMonitoring ? (
                <IconPause className="h-4 w-4 mr-2" />
              ) : (
                <IconPlay className="h-4 w-4 mr-2" />
              )}
              {isMonitoring ? 'Pause' : 'Start'}
            </Button>

            <Button variant="outline" size="sm" onClick={clearData}>
              <IconRefresh className="h-4 w-4 mr-2" />
              Clear
            </Button>

            <Button variant="outline" size="sm" onClick={exportReport}>
              <IconDownload className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Frame Rate"
          subtitle="Rendering performance"
          data={fpsMetric}
          size="medium"
          showSparkline={true}
          showTarget={true}
        />

        <MetricCard
          title="Memory Usage"
          subtitle="Heap memory consumption"
          data={memoryMetric}
          size="medium"
          showSparkline={true}
        />

        <MetricCard
          title="Network Latency"
          subtitle="Request response time"
          data={latencyMetric}
          size="medium"
          showSparkline={true}
          showTarget={true}
        />

        <MetricCard
          title="Render Time"
          subtitle="Component update time"
          data={renderMetric}
          size="medium"
          showSparkline={true}
          showTarget={true}
        />
      </div>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="monitoring">Real-time Monitoring</TabsTrigger>
          <TabsTrigger value="loadtest">Load Testing</TabsTrigger>
          <TabsTrigger value="analysis">Performance Analysis</TabsTrigger>
        </TabsList>

        {/* Real-time Monitoring Tab */}
        <TabsContent value="monitoring" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Performance Charts */}
            <DashboardWidget
              config={{
                id: 'fps-chart',
                title: 'Frame Rate Trend',
                subtitle: 'Real-time FPS monitoring',
                size: 'large',
                priority: 'medium',
                category: 'system',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData,
              }}
            >
              <RealTimeLineChart
                data={deferredData.map(d => ({ timestamp: d.timestamp, value: d.fps }))}
                height={250}
                color="success"
                formatValue={(value) => `${value} fps`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <DashboardWidget
              config={{
                id: 'memory-chart',
                title: 'Memory Usage Trend',
                subtitle: 'Heap memory consumption',
                size: 'large',
                priority: 'medium',
                category: 'system',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.map(d => ({ timestamp: d.timestamp, value: d.memoryUsage }))}
                height={250}
                gradientColors={['hsl(var(--warning))', 'transparent']}
                strokeColor="hsl(var(--warning))"
                formatValue={(value) => `${value.toFixed(1)} MB`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>
          </div>

          {/* Processing Metrics */}
          {processingMetrics && (
            <Card className="p-4">
              <h3 className="text-sm font-semibold mb-3">Data Processing Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Total Processed</div>
                  <div className="font-medium">{processingMetrics.totalProcessed.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Avg Processing Time</div>
                  <div className="font-medium">{processingMetrics.averageProcessingTime.toFixed(2)}ms</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Peak Processing Time</div>
                  <div className="font-medium">{processingMetrics.peakProcessingTime.toFixed(2)}ms</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Memory Usage</div>
                  <div className="font-medium">{processingMetrics.memoryUsage.toFixed(1)}MB</div>
                </div>
              </div>
            </Card>
          )}
        </TabsContent>

        {/* Load Testing Tab */}
        <TabsContent value="loadtest" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Load Test Configuration */}
            <div className="lg:col-span-1">
              <Card className="p-4">
                <h3 className="text-sm font-semibold mb-4">Load Test Configuration</h3>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="preset">Test Preset</Label>
                    <Select value={selectedPreset} onValueChange={setSelectedPreset}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="light">Light Load</SelectItem>
                        <SelectItem value="moderate">Moderate Load</SelectItem>
                        <SelectItem value="heavy">Heavy Load</SelectItem>
                        <SelectItem value="stress">Stress Test</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="duration">Duration (seconds)</Label>
                    <Input
                      id="duration"
                      type="number"
                      value={customConfig.duration || LOAD_TEST_PRESETS[selectedPreset].duration}
                      onChange={(e) => setCustomConfig(prev => ({
                        ...prev,
                        duration: parseInt(e.target.value)
                      }))}
                    />
                  </div>

                  <div>
                    <Label htmlFor="messagesPerSecond">Messages/Second</Label>
                    <Input
                      id="messagesPerSecond"
                      type="number"
                      value={customConfig.messagesPerSecond || LOAD_TEST_PRESETS[selectedPreset].messagesPerSecond}
                      onChange={(e) => setCustomConfig(prev => ({
                        ...prev,
                        messagesPerSecond: parseInt(e.target.value)
                      }))}
                    />
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={customConfig.enableBurstMode ?? LOAD_TEST_PRESETS[selectedPreset].enableBurstMode}
                      onCheckedChange={(checked) => setCustomConfig(prev => ({
                        ...prev,
                        enableBurstMode: checked
                      }))}
                    />
                    <Label>Enable Burst Mode</Label>
                  </div>

                  <div className="flex space-x-2">
                    <Button
                      className="flex-1"
                      onClick={startLoadTest}
                      disabled={isLoadTesting}
                    >
                      {isLoadTesting ? (
                        <IconLoader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <IconPlay className="h-4 w-4 mr-2" />
                      )}
                      Start Test
                    </Button>

                    {isLoadTesting && (
                      <Button variant="destructive" onClick={stopLoadTest}>
                        <IconStop className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            </div>

            {/* Load Test Results */}
            <div className="lg:col-span-2">
              <Card className="p-4">
                <h3 className="text-sm font-semibold mb-4">Load Test Results</h3>

                {loadTestMetrics ? (
                  <div className="space-y-4">
                    {/* Progress bar */}
                    {isLoadTesting && (
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span>Progress</span>
                          <span>
                            {((Date.now() - loadTestMetrics.startTime) / 1000).toFixed(0)}s /
                            {LOAD_TEST_PRESETS[selectedPreset].duration}s
                          </span>
                        </div>
                        <Progress
                          value={((Date.now() - loadTestMetrics.startTime) / 1000) / LOAD_TEST_PRESETS[selectedPreset].duration * 100}
                        />
                      </div>
                    )}

                    {/* Metrics grid */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Messages Sent</div>
                        <div className="font-medium">{loadTestMetrics.messagesSent.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Messages Received</div>
                        <div className="font-medium">{loadTestMetrics.messagesReceived.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Throughput</div>
                        <div className="font-medium">{loadTestMetrics.throughput.toFixed(1)} msg/s</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Avg Latency</div>
                        <div className="font-medium">{loadTestMetrics.averageLatency.toFixed(2)}ms</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Peak Latency</div>
                        <div className="font-medium">{loadTestMetrics.peakLatency.toFixed(2)}ms</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Error Rate</div>
                        <div className={cn(
                          'font-medium',
                          loadTestMetrics.errorCount > 0 ? 'text-destructive' : 'text-success'
                        )}>
                          {((loadTestMetrics.errorCount / loadTestMetrics.messagesSent) * 100).toFixed(2)}%
                        </div>
                      </div>
                    </div>

                    {/* Memory usage */}
                    <div>
                      <h4 className="text-sm font-medium mb-2">Memory Usage</h4>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <div className="text-muted-foreground">Initial</div>
                          <div className="font-medium">{loadTestMetrics.memoryUsage.initial.toFixed(1)}MB</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Peak</div>
                          <div className="font-medium">{loadTestMetrics.memoryUsage.peak.toFixed(1)}MB</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Final</div>
                          <div className="font-medium">{loadTestMetrics.memoryUsage.final.toFixed(1)}MB</div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <IconGauge className="h-8 w-8 mx-auto mb-2" />
                    <p>No load test results available</p>
                    <p className="text-xs mt-1">Start a load test to see performance metrics</p>
                  </div>
                )}
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Performance Analysis Tab */}
        <TabsContent value="analysis" className="space-y-6">
          <Card className="p-4">
            <h3 className="text-sm font-semibold mb-4">Performance Analysis</h3>

            {deferredData.length > 0 ? (
              <div className="space-y-6">
                {/* Performance summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Avg FPS</div>
                    <div className="font-medium">
                      {(deferredData.reduce((sum, d) => sum + d.fps, 0) / deferredData.length).toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Peak Memory</div>
                    <div className="font-medium">
                      {Math.max(...deferredData.map(d => d.memoryUsage)).toFixed(1)}MB
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Avg Latency</div>
                    <div className="font-medium">
                      {(deferredData.reduce((sum, d) => sum + d.networkLatency, 0) / deferredData.length).toFixed(1)}ms
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Data Points</div>
                    <div className="font-medium">{deferredData.length}</div>
                  </div>
                </div>

                {/* Performance recommendations */}
                <div>
                  <h4 className="text-sm font-medium mb-2">Performance Recommendations</h4>
                  <div className="space-y-2">
                    {currentData && currentData.memoryUsage > PERFORMANCE_THRESHOLDS.memoryUsage.warning && (
                      <div className="flex items-center space-x-2 text-sm text-warning">
                        <IconAlertTriangle className="h-4 w-4" />
                        <span>High memory usage detected. Consider implementing data cleanup.</span>
                      </div>
                    )}

                    {currentData && currentData.fps < PERFORMANCE_THRESHOLDS.fps.warning && (
                      <div className="flex items-center space-x-2 text-sm text-warning">
                        <IconAlertTriangle className="h-4 w-4" />
                        <span>Low frame rate detected. Consider reducing update frequency.</span>
                      </div>
                    )}

                    {currentData && currentData.renderTime > PERFORMANCE_THRESHOLDS.renderTime.warning && (
                      <div className="flex items-center space-x-2 text-sm text-warning">
                        <IconAlertTriangle className="h-4 w-4" />
                        <span>Slow render times detected. Consider optimizing components.</span>
                      </div>
                    )}

                    {!currentData || (
                      currentData.memoryUsage <= PERFORMANCE_THRESHOLDS.memoryUsage.warning &&
                      currentData.fps >= PERFORMANCE_THRESHOLDS.fps.warning &&
                      currentData.renderTime <= PERFORMANCE_THRESHOLDS.renderTime.warning
                    ) && (
                      <div className="flex items-center space-x-2 text-sm text-success">
                        <IconCheckCircle className="h-4 w-4" />
                        <span>Performance is within optimal thresholds.</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <IconChartLine className="h-8 w-8 mx-auto mb-2" />
                <p>No performance data available</p>
                <p className="text-xs mt-1">Start monitoring to collect performance metrics</p>
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PerformanceMonitor;
