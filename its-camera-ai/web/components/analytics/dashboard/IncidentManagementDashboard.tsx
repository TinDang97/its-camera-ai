/**
 * Incident Management Dashboard Component
 *
 * Comprehensive incident monitoring dashboard with real-time alerts,
 * timeline visualization, response analytics, and incident lifecycle management.
 */

'use client';

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { useDeferredValue, useTransition } from 'react';
import { cn } from '@/lib/utils';
import {
  IconAlertTriangle,
  IconAlertCircle,
  IconAlertOctagon,
  IconCheck,
  IconX,
  IconClock,
  IconUser,
  IconMapPin,
  IconRefresh,
  IconFilter,
  IconSort,
  IconActivity,
  IconTrendingUp,
  IconCamera
} from '@tabler/icons-react';
import { MetricCard, MetricCardData } from './MetricCard';
import { DashboardWidget, WidgetConfig } from './DashboardWidget';
import { RealTimeLineChart } from '@/components/charts/RealTimeLineChart';
import { RealTimeAreaChart } from '@/components/charts/RealTimeAreaChart';
import { RealTimeBarChart, BarChartData } from '@/components/charts/RealTimeBarChart';
import { useAnalyticsStore, useIncidentData } from '@/stores/analytics';
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
import { Textarea } from '@/components/ui/textarea';

export interface IncidentData {
  id: string;
  title: string;
  description: string;
  type: 'traffic_accident' | 'camera_malfunction' | 'system_outage' | 'network_issue' | 'security_alert' | 'weather_event' | 'maintenance';
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  priority: 'p1' | 'p2' | 'p3' | 'p4';

  // Location and context
  location: {
    cameraId?: string;
    intersection?: string;
    coordinates?: { lat: number; lng: number };
    address?: string;
  };

  // Timing
  createdAt: number;
  updatedAt: number;
  resolvedAt?: number;
  closedAt?: number;

  // Assignment and response
  assignedTo?: {
    userId: string;
    name: string;
    role: string;
    contactInfo: {
      email: string;
      phone?: string;
    };
  };
  responders: Array<{
    type: 'police' | 'fire' | 'medical' | 'traffic' | 'technical' | 'maintenance';
    name: string;
    status: 'dispatched' | 'en_route' | 'on_scene' | 'completed';
    eta?: number;
    arrivedAt?: number;
  }>;

  // Impact assessment
  impact: {
    trafficDisruption: 'none' | 'minimal' | 'moderate' | 'severe';
    affectedCameras: string[];
    estimatedVehicles: number;
    expectedDuration: number; // minutes
  };

  // Communication and updates
  updates: Array<{
    id: string;
    timestamp: number;
    authorId: string;
    authorName: string;
    message: string;
    type: 'status_update' | 'comment' | 'escalation' | 'resolution';
    attachments?: Array<{
      type: 'image' | 'video' | 'document';
      url: string;
      name: string;
    }>;
  }>;

  // Analytics
  metrics: {
    responseTime: number; // minutes
    resolutionTime?: number; // minutes
    escalationCount: number;
    affectedUsers: number;
    costEstimate?: number;
  };

  // Automation and ML
  detectionSource: 'manual' | 'camera_ai' | 'traffic_sensor' | 'citizen_report' | 'system_alert';
  confidenceScore?: number; // 0-1 for AI detections
  relatedIncidents: string[];
  tags: string[];
}

export interface IncidentManagementData {
  incidents: IncidentData[];
  summary: {
    total: number;
    open: number;
    inProgress: number;
    resolved: number;
    closed: number;
    avgResponseTime: number; // minutes
    avgResolutionTime: number; // minutes
  };
  metrics: {
    incidentsPerHour: Array<{ timestamp: number; count: number; severity: Record<string, number> }>;
    responseTimeHistory: Array<{ timestamp: number; responseTime: number; incidentType: string }>;
    resolutionTrends: Array<{ timestamp: number; resolved: number; escalated: number }>;
    typeDistribution: Array<{ type: string; count: number; avgSeverity: number }>;
  };
  alerts: {
    slaBreaches: number;
    escalationAlerts: number;
    highPriorityOpen: number;
    overdueIncidents: number;
  };
  performance: {
    slaCompliance: number; // percentage
    firstResponseTime: number; // minutes
    customerSatisfaction?: number; // 1-5 scale
    teamUtilization: number; // percentage
  };
  timestamp: number;
}

export interface IncidentManagementDashboardProps {
  data?: IncidentManagementData;
  className?: string;
  showControls?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onIncidentAction?: (incidentId: string, action: 'assign' | 'escalate' | 'resolve' | 'close', data?: any) => void;
  onDataExport?: () => void;
  selectedTimeRange?: '1h' | '6h' | '24h' | '7d';
  selectedFilters?: {
    severity?: string[];
    type?: string[];
    status?: string[];
    priority?: string[];
  };
}

// Severity and status configurations
const severityConfig = {
  low: { color: 'text-success', bgColor: 'bg-success/20', label: 'Low', icon: IconAlertCircle },
  medium: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'Medium', icon: IconAlertTriangle },
  high: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'High', icon: IconAlertTriangle },
  critical: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Critical', icon: IconAlertOctagon },
};

const statusConfig = {
  open: { color: 'text-destructive', bgColor: 'bg-destructive/20', label: 'Open' },
  in_progress: { color: 'text-warning', bgColor: 'bg-warning/20', label: 'In Progress' },
  resolved: { color: 'text-success', bgColor: 'bg-success/20', label: 'Resolved' },
  closed: { color: 'text-muted-foreground', bgColor: 'bg-muted/20', label: 'Closed' },
};

const priorityConfig = {
  p1: { color: 'text-destructive', label: 'P1 - Critical' },
  p2: { color: 'text-destructive', label: 'P2 - High' },
  p3: { color: 'text-warning', label: 'P3 - Medium' },
  p4: { color: 'text-muted-foreground', label: 'P4 - Low' },
};

// Generate mock incident data
const generateMockIncidentData = (): IncidentManagementData => {
  const now = Date.now();
  const incidents: IncidentData[] = Array.from({ length: 15 }, (_, i) => {
    const createdAt = now - Math.random() * 7 * 24 * 60 * 60 * 1000; // Last 7 days
    const severity = ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any;
    const status = ['open', 'in_progress', 'resolved', 'closed'][Math.floor(Math.random() * 4)] as any;
    const type = ['traffic_accident', 'camera_malfunction', 'system_outage', 'network_issue', 'security_alert'][Math.floor(Math.random() * 5)] as any;

    const responseTime = 5 + Math.random() * 30; // 5-35 minutes
    const resolutionTime = status === 'resolved' || status === 'closed' ? responseTime + Math.random() * 120 : undefined;

    return {
      id: `INC-${String(i + 1).padStart(4, '0')}`,
      title: `Incident ${i + 1}: ${type.replace('_', ' ')} detected`,
      description: `Detailed description for incident ${i + 1}`,
      type,
      severity,
      status,
      priority: ['p1', 'p2', 'p3', 'p4'][Math.floor(Math.random() * 4)] as any,
      location: {
        cameraId: `CAM-${String(Math.floor(Math.random() * 10) + 1).padStart(3, '0')}`,
        intersection: `Street ${i + 1} & Avenue ${String.fromCharCode(65 + (i % 6))}`,
        coordinates: { lat: 40.7128 + Math.random() * 0.01, lng: -74.0060 + Math.random() * 0.01 },
        address: `${100 + i * 10} Street ${i + 1}`,
      },
      createdAt,
      updatedAt: createdAt + Math.random() * 60000,
      resolvedAt: status === 'resolved' || status === 'closed' ? createdAt + resolutionTime! * 60000 : undefined,
      closedAt: status === 'closed' ? createdAt + (resolutionTime! + 10) * 60000 : undefined,
      assignedTo: Math.random() > 0.3 ? {
        userId: `user-${i}`,
        name: `Operator ${i + 1}`,
        role: 'Traffic Analyst',
        contactInfo: {
          email: `operator${i + 1}@example.com`,
          phone: `+1-555-${String(i).padStart(3, '0')}-${String(Math.floor(Math.random() * 10000)).padStart(4, '0')}`,
        },
      } : undefined,
      responders: Math.random() > 0.5 ? [
        {
          type: 'police',
          name: 'Unit 123',
          status: ['dispatched', 'en_route', 'on_scene', 'completed'][Math.floor(Math.random() * 4)] as any,
          eta: Math.random() > 0.5 ? Math.floor(5 + Math.random() * 15) : undefined,
          arrivedAt: Math.random() > 0.7 ? createdAt + Math.random() * 1800000 : undefined,
        },
      ] : [],
      impact: {
        trafficDisruption: ['none', 'minimal', 'moderate', 'severe'][Math.floor(Math.random() * 4)] as any,
        affectedCameras: [`CAM-${String(Math.floor(Math.random() * 10) + 1).padStart(3, '0')}`],
        estimatedVehicles: Math.floor(Math.random() * 100),
        expectedDuration: Math.floor(15 + Math.random() * 120),
      },
      updates: Array.from({ length: Math.floor(1 + Math.random() * 5) }, (_, j) => ({
        id: `update-${i}-${j}`,
        timestamp: createdAt + j * 300000, // Every 5 minutes
        authorId: `user-${i}`,
        authorName: `Operator ${i + 1}`,
        message: `Update ${j + 1} for incident ${i + 1}`,
        type: ['status_update', 'comment', 'escalation', 'resolution'][Math.floor(Math.random() * 4)] as any,
      })),
      metrics: {
        responseTime,
        resolutionTime,
        escalationCount: Math.floor(Math.random() * 3),
        affectedUsers: Math.floor(Math.random() * 1000),
        costEstimate: Math.random() * 10000,
      },
      detectionSource: ['manual', 'camera_ai', 'traffic_sensor', 'citizen_report', 'system_alert'][Math.floor(Math.random() * 5)] as any,
      confidenceScore: Math.random(),
      relatedIncidents: [],
      tags: ['traffic', 'urgent', 'downtown'].slice(0, Math.floor(1 + Math.random() * 3)),
    };
  });

  // Calculate summary
  const summary = {
    total: incidents.length,
    open: incidents.filter(i => i.status === 'open').length,
    inProgress: incidents.filter(i => i.status === 'in_progress').length,
    resolved: incidents.filter(i => i.status === 'resolved').length,
    closed: incidents.filter(i => i.status === 'closed').length,
    avgResponseTime: incidents.reduce((sum, i) => sum + i.metrics.responseTime, 0) / incidents.length,
    avgResolutionTime: incidents
      .filter(i => i.metrics.resolutionTime)
      .reduce((sum, i) => sum + i.metrics.resolutionTime!, 0) /
      incidents.filter(i => i.metrics.resolutionTime).length || 0,
  };

  // Generate time series data
  const incidentsPerHour = Array.from({ length: 24 }, (_, i) => ({
    timestamp: now - (23 - i) * 60 * 60 * 1000,
    count: Math.floor(Math.random() * 5),
    severity: {
      low: Math.floor(Math.random() * 2),
      medium: Math.floor(Math.random() * 2),
      high: Math.floor(Math.random() * 1),
      critical: Math.floor(Math.random() * 1),
    },
  }));

  const responseTimeHistory = Array.from({ length: 20 }, (_, i) => ({
    timestamp: now - (19 - i) * 60 * 60 * 1000,
    responseTime: 10 + Math.random() * 20,
    incidentType: ['traffic_accident', 'camera_malfunction', 'system_outage'][Math.floor(Math.random() * 3)],
  }));

  const resolutionTrends = Array.from({ length: 12 }, (_, i) => ({
    timestamp: now - (11 - i) * 60 * 60 * 1000,
    resolved: Math.floor(Math.random() * 8),
    escalated: Math.floor(Math.random() * 3),
  }));

  const typeDistribution = [
    { type: 'Traffic Accident', count: 45, avgSeverity: 2.8 },
    { type: 'Camera Malfunction', count: 23, avgSeverity: 2.1 },
    { type: 'System Outage', count: 12, avgSeverity: 3.5 },
    { type: 'Network Issue', count: 18, avgSeverity: 2.3 },
    { type: 'Security Alert', count: 8, avgSeverity: 3.2 },
  ];

  return {
    incidents,
    summary,
    metrics: {
      incidentsPerHour,
      responseTimeHistory,
      resolutionTrends,
      typeDistribution,
    },
    alerts: {
      slaBreaches: Math.floor(Math.random() * 5),
      escalationAlerts: Math.floor(Math.random() * 3),
      highPriorityOpen: incidents.filter(i => ['p1', 'p2'].includes(i.priority) && ['open', 'in_progress'].includes(i.status)).length,
      overdueIncidents: Math.floor(Math.random() * 4),
    },
    performance: {
      slaCompliance: 85 + Math.random() * 12,
      firstResponseTime: 8 + Math.random() * 7,
      customerSatisfaction: 3.5 + Math.random() * 1.5,
      teamUtilization: 60 + Math.random() * 30,
    },
    timestamp: now,
  };
};

export const IncidentManagementDashboard: React.FC<IncidentManagementDashboardProps> = ({
  data: externalData,
  className,
  showControls = true,
  autoRefresh = true,
  refreshInterval = 30000,
  onIncidentAction,
  onDataExport,
  selectedTimeRange = '24h',
  selectedFilters = {},
}) => {
  const [isPending, startTransition] = useTransition();
  const [lastRefresh, setLastRefresh] = useState(Date.now());
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState(selectedTimeRange);
  const [selectedIncident, setSelectedIncident] = useState<string | null>(null);
  const [filters, setFilters] = useState(selectedFilters);
  const [sortBy, setSortBy] = useState<'created' | 'severity' | 'priority' | 'status'>('created');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Use real incident data
  const incidentData = useIncidentData();
  const { isConnected } = useRealTimeMetrics();

  // Generate or use provided data
  const data = useMemo(() => {
    if (externalData) return externalData;

    // Convert real incident data to dashboard format if available
    if (incidentData && incidentData.length > 0) {
      const mockData = generateMockIncidentData();
      // In a real implementation, transform real data here
      return mockData;
    }

    return generateMockIncidentData();
  }, [externalData, incidentData, lastRefresh]);

  const deferredData = useDeferredValue(data);

  // Filter and sort incidents
  const filteredAndSortedIncidents = useMemo(() => {
    let filtered = deferredData.incidents;

    // Apply filters
    if (filters.severity?.length) {
      filtered = filtered.filter(incident => filters.severity!.includes(incident.severity));
    }
    if (filters.type?.length) {
      filtered = filtered.filter(incident => filters.type!.includes(incident.type));
    }
    if (filters.status?.length) {
      filtered = filtered.filter(incident => filters.status!.includes(incident.status));
    }
    if (filters.priority?.length) {
      filtered = filtered.filter(incident => filters.priority!.includes(incident.priority));
    }

    // Apply sorting
    return filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'created':
          comparison = a.createdAt - b.createdAt;
          break;
        case 'severity':
          const severityOrder = { low: 1, medium: 2, high: 3, critical: 4 };
          comparison = severityOrder[a.severity] - severityOrder[b.severity];
          break;
        case 'priority':
          const priorityOrder = { p4: 1, p3: 2, p2: 3, p1: 4 };
          comparison = priorityOrder[a.priority] - priorityOrder[b.priority];
          break;
        case 'status':
          const statusOrder = { closed: 1, resolved: 2, in_progress: 3, open: 4 };
          comparison = statusOrder[a.status] - statusOrder[b.status];
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });
  }, [deferredData.incidents, filters, sortBy, sortOrder]);

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

  // Incident action handler
  const handleIncidentAction = useCallback((incidentId: string, action: 'assign' | 'escalate' | 'resolve' | 'close', data?: any) => {
    console.log(`Incident action: ${action} on ${incidentId}`, data);
    onIncidentAction?.(incidentId, action, data);
    handleRefresh();
  }, [onIncidentAction, handleRefresh]);

  // Time range change handler
  const handleTimeRangeChange = useCallback((range: '1h' | '6h' | '24h' | '7d') => {
    setTimeRange(range);
    handleRefresh();
  }, [handleRefresh]);

  // Format duration
  const formatDuration = useCallback((minutes: number): string => {
    if (minutes < 60) return `${Math.round(minutes)}m`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  }, []);

  // Metric card configurations
  const totalIncidentsMetric: MetricCardData = {
    value: deferredData.summary.total,
    format: 'number',
    status: 'normal',
    metadata: {
      description: 'Total incidents in selected time range',
    },
  };

  const openIncidentsMetric: MetricCardData = {
    value: deferredData.summary.open + deferredData.summary.inProgress,
    format: 'number',
    status: deferredData.summary.open > 10 ? 'critical' :
           deferredData.summary.open > 5 ? 'warning' : 'good',
    metadata: {
      description: 'Active incidents requiring attention',
    },
  };

  const avgResponseTimeMetric: MetricCardData = {
    value: deferredData.summary.avgResponseTime,
    unit: 'min',
    format: 'number',
    status: deferredData.summary.avgResponseTime > 15 ? 'critical' :
           deferredData.summary.avgResponseTime > 10 ? 'warning' : 'good',
    timeSeries: deferredData.metrics.responseTimeHistory.map(r => ({
      timestamp: r.timestamp,
      value: r.responseTime,
    })),
    metadata: {
      description: 'Average time to first response',
    },
  };

  const slaComplianceMetric: MetricCardData = {
    value: deferredData.performance.slaCompliance,
    unit: '%',
    format: 'number',
    status: deferredData.performance.slaCompliance > 95 ? 'good' :
           deferredData.performance.slaCompliance > 85 ? 'warning' : 'critical',
    metadata: {
      description: 'SLA compliance rate',
    },
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Dashboard Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Incident Management</h2>
          <p className="text-muted-foreground">
            Real-time incident monitoring and response coordination
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

            {/* Filter controls */}
            <Button variant="outline" size="sm">
              <IconFilter className="h-4 w-4 mr-2" />
              Filter
            </Button>

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

      {/* Alert Summary Bar */}
      {(deferredData.alerts.slaBreaches > 0 || deferredData.alerts.escalationAlerts > 0 || deferredData.alerts.overdueIncidents > 0) && (
        <Card className="p-4 border-destructive bg-destructive/5">
          <div className="flex items-center space-x-4 text-sm">
            <IconAlertTriangle className="h-5 w-5 text-destructive" />
            <div className="flex space-x-6">
              {deferredData.alerts.slaBreaches > 0 && (
                <span>
                  <strong>{deferredData.alerts.slaBreaches}</strong> SLA breaches
                </span>
              )}
              {deferredData.alerts.escalationAlerts > 0 && (
                <span>
                  <strong>{deferredData.alerts.escalationAlerts}</strong> escalation alerts
                </span>
              )}
              {deferredData.alerts.overdueIncidents > 0 && (
                <span>
                  <strong>{deferredData.alerts.overdueIncidents}</strong> overdue incidents
                </span>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* Key Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Incidents"
          subtitle="In selected period"
          data={totalIncidentsMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="Active Incidents"
          subtitle="Open & in progress"
          data={openIncidentsMetric}
          size="medium"
          showSparkline={false}
        />

        <MetricCard
          title="Avg Response Time"
          subtitle="Time to first response"
          data={avgResponseTimeMetric}
          size="medium"
          showSparkline={true}
        />

        <MetricCard
          title="SLA Compliance"
          subtitle="Meeting response SLA"
          data={slaComplianceMetric}
          size="medium"
          showSparkline={false}
        />
      </div>

      {/* Tabbed Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="incidents">Incidents</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Status Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-destructive">Open</h3>
                  <p className="text-3xl font-bold">{deferredData.summary.open}</p>
                </div>
                <IconAlertTriangle className="h-8 w-8 text-destructive" />
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-warning">In Progress</h3>
                  <p className="text-3xl font-bold">{deferredData.summary.inProgress}</p>
                </div>
                <IconActivity className="h-8 w-8 text-warning" />
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-success">Resolved</h3>
                  <p className="text-3xl font-bold">{deferredData.summary.resolved}</p>
                </div>
                <IconCheck className="h-8 w-8 text-success" />
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-muted-foreground">Closed</h3>
                  <p className="text-3xl font-bold">{deferredData.summary.closed}</p>
                </div>
                <IconX className="h-8 w-8 text-muted-foreground" />
              </div>
            </Card>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DashboardWidget
              config={{
                id: 'incident-volume',
                title: 'Incident Volume',
                subtitle: 'Incidents per hour over time',
                size: 'large',
                priority: 'high',
                category: 'incident',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.metrics.incidentsPerHour,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.metrics.incidentsPerHour.map(i => ({
                  timestamp: i.timestamp,
                  value: i.count,
                }))}
                height={300}
                gradientColors={['hsl(var(--destructive))', 'transparent']}
                strokeColor="hsl(var(--destructive))"
                formatValue={(value) => `${Math.round(value)} incidents`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <DashboardWidget
              config={{
                id: 'response-time-trend',
                title: 'Response Time Trend',
                subtitle: 'Average response time over time',
                size: 'large',
                priority: 'medium',
                category: 'incident',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.metrics.responseTimeHistory,
              }}
            >
              <RealTimeLineChart
                data={deferredData.metrics.responseTimeHistory.map(r => ({
                  timestamp: r.timestamp,
                  value: r.responseTime,
                }))}
                height={300}
                color="warning"
                formatValue={(value) => `${Math.round(value)} min`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>
          </div>

          {/* Type Distribution */}
          <DashboardWidget
            config={{
              id: 'incident-types',
              title: 'Incident Type Distribution',
              subtitle: 'Most common incident types',
              size: 'large',
              priority: 'medium',
              category: 'incident',
            }}
            data={{
              timestamp: Date.now(),
              isLoading: false,
              data: deferredData.metrics.typeDistribution,
            }}
          >
            <RealTimeBarChart
              data={deferredData.metrics.typeDistribution.map(t => ({
                category: t.type,
                value: t.count,
                timestamp: Date.now(),
                metadata: { avgSeverity: t.avgSeverity },
              }))}
              height={300}
              orientation="horizontal"
              showValues={true}
              sortBars="desc"
              formatValue={(value) => `${value} incidents`}
            />
          </DashboardWidget>
        </TabsContent>

        {/* Incidents Tab */}
        <TabsContent value="incidents" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Label htmlFor="sort-by" className="text-sm">Sort by:</Label>
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="created">Created</SelectItem>
                    <SelectItem value="severity">Severity</SelectItem>
                    <SelectItem value="priority">Priority</SelectItem>
                    <SelectItem value="status">Status</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              >
                <IconSort className="h-4 w-4 mr-2" />
                {sortOrder === 'asc' ? 'Ascending' : 'Descending'}
              </Button>
            </div>

            <div className="text-sm text-muted-foreground">
              Showing {filteredAndSortedIncidents.length} of {deferredData.incidents.length} incidents
            </div>
          </div>

          <div className="space-y-4">
            {filteredAndSortedIncidents.map((incident) => (
              <Card key={incident.id} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="font-medium">{incident.id}</span>
                      <Badge
                        variant={incident.severity === 'critical' || incident.severity === 'high' ? 'destructive' :
                                incident.severity === 'medium' ? 'secondary' : 'outline'}
                        className={severityConfig[incident.severity].color}
                      >
                        {severityConfig[incident.severity].label}
                      </Badge>
                      <Badge
                        variant={incident.status === 'open' ? 'destructive' :
                                incident.status === 'in_progress' ? 'secondary' :
                                incident.status === 'resolved' ? 'default' : 'outline'}
                      >
                        {statusConfig[incident.status].label}
                      </Badge>
                      <Badge variant="outline" className={priorityConfig[incident.priority].color}>
                        {incident.priority.toUpperCase()}
                      </Badge>
                    </div>

                    <h4 className="font-medium mb-1">{incident.title}</h4>
                    <p className="text-sm text-muted-foreground mb-2">{incident.description}</p>

                    <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                      <div className="flex items-center space-x-1">
                        <IconMapPin className="h-3 w-3" />
                        <span>{incident.location.intersection}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <IconClock className="h-3 w-3" />
                        <span>{new Date(incident.createdAt).toLocaleString()}</span>
                      </div>
                      {incident.assignedTo && (
                        <div className="flex items-center space-x-1">
                          <IconUser className="h-3 w-3" />
                          <span>{incident.assignedTo.name}</span>
                        </div>
                      )}
                      <div className="flex items-center space-x-1">
                        <IconCamera className="h-3 w-3" />
                        <span>{incident.location.cameraId}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex space-x-2 ml-4">
                    {incident.status === 'open' && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleIncidentAction(incident.id, 'assign')}
                      >
                        Assign
                      </Button>
                    )}
                    {['open', 'in_progress'].includes(incident.status) && (
                      <>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleIncidentAction(incident.id, 'escalate')}
                        >
                          Escalate
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleIncidentAction(incident.id, 'resolve')}
                        >
                          Resolve
                        </Button>
                      </>
                    )}
                    {incident.status === 'resolved' && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleIncidentAction(incident.id, 'close')}
                      >
                        Close
                      </Button>
                    )}
                  </div>
                </div>

                {/* Incident metrics */}
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
                  <div>
                    <p className="text-xs text-muted-foreground">Response Time</p>
                    <p className="font-medium">{formatDuration(incident.metrics.responseTime)}</p>
                  </div>
                  {incident.metrics.resolutionTime && (
                    <div>
                      <p className="text-xs text-muted-foreground">Resolution Time</p>
                      <p className="font-medium">{formatDuration(incident.metrics.resolutionTime)}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-xs text-muted-foreground">Affected Vehicles</p>
                    <p className="font-medium">{incident.impact.estimatedVehicles}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Updates</p>
                    <p className="font-medium">{incident.updates.length}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DashboardWidget
              config={{
                id: 'resolution-trends',
                title: 'Resolution Trends',
                subtitle: 'Resolved vs escalated incidents',
                size: 'large',
                priority: 'medium',
                category: 'incident',
              }}
              data={{
                timestamp: Date.now(),
                isLoading: false,
                data: deferredData.metrics.resolutionTrends,
              }}
            >
              <RealTimeAreaChart
                data={deferredData.metrics.resolutionTrends.map(r => ({
                  timestamp: r.timestamp,
                  value: r.resolved,
                }))}
                height={300}
                gradientColors={['hsl(var(--success))', 'transparent']}
                strokeColor="hsl(var(--success))"
                formatValue={(value) => `${Math.round(value)} resolved`}
                enableAnimations={true}
                responsive={true}
              />
            </DashboardWidget>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>SLA Compliance</span>
                    <span>{deferredData.performance.slaCompliance.toFixed(1)}%</span>
                  </div>
                  <Progress value={deferredData.performance.slaCompliance} variant="success" />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Team Utilization</span>
                    <span>{deferredData.performance.teamUtilization.toFixed(1)}%</span>
                  </div>
                  <Progress value={deferredData.performance.teamUtilization} />
                </div>

                <div className="pt-2 space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">First Response Time</span>
                    <span className="font-medium">{formatDuration(deferredData.performance.firstResponseTime)}</span>
                  </div>
                  {deferredData.performance.customerSatisfaction && (
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Customer Satisfaction</span>
                      <span className="font-medium">{deferredData.performance.customerSatisfaction.toFixed(1)}/5.0</span>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Timeline Tab */}
        <TabsContent value="timeline" className="space-y-6">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Recent Incident Activity</h3>

            {filteredAndSortedIncidents.slice(0, 10).map((incident) => (
              <Card key={incident.id} className="p-4">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    {React.createElement(severityConfig[incident.severity].icon, {
                      className: cn('h-5 w-5', severityConfig[incident.severity].color)
                    })}
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">{incident.title}</h4>
                      <span className="text-xs text-muted-foreground">
                        {new Date(incident.createdAt).toLocaleString()}
                      </span>
                    </div>

                    <p className="text-sm text-muted-foreground mt-1">
                      {incident.description}
                    </p>

                    <div className="mt-2 flex items-center space-x-4 text-xs">
                      <Badge variant="outline">{incident.location.intersection}</Badge>
                      <Badge variant="outline">{incident.status}</Badge>
                      <Badge variant="outline">{incident.severity}</Badge>
                    </div>

                    {incident.updates.length > 0 && (
                      <div className="mt-3 pl-4 border-l border-muted">
                        <div className="text-xs text-muted-foreground">Latest update:</div>
                        <p className="text-sm mt-1">
                          {incident.updates[incident.updates.length - 1].message}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default IncidentManagementDashboard;
