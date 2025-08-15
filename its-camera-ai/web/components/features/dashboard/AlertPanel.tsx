'use client';

import { useState, useMemo, useCallback, memo, use, Suspense, startTransition } from 'react';
import {
  IconAlertCircle,
  IconClock,
  IconMapPin,
  IconCamera,
  IconAlertTriangle,
  IconBolt,
  IconShield
} from '@tabler/icons-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { AlertDetailsModal } from '@/components/features/alerts/AlertDetailsModal';

type AlertType = 'critical' | 'warning' | 'info';
type AlertCategory = 'traffic_violation' | 'accident' | 'congestion' | 'system' | 'security';

interface AlertData {
  id: string;
  type: AlertType;
  category: AlertCategory;
  title: string;
  description: string;
  location: string;
  cameraId: string;
  timestamp: string;
  confidence: number;
  status: 'active' | 'resolved' | 'investigating';
  metadata?: Record<string, any>;
}

interface AlertPanelProps {
  maxHeight?: string;
}

const mockAlerts: AlertData[] = [
  {
    id: 'alert-001',
    type: 'critical',
    category: 'traffic_violation',
    title: 'Speed Violation Detected',
    description: 'Vehicle exceeded speed limit by 25 km/h in school zone',
    location: 'Main St & Oak Ave (Camera-07)',
    cameraId: 'CAM-007',
    timestamp: '2024-01-15T14:30:22Z',
    confidence: 95.2,
    status: 'active',
    metadata: {
      vehicleType: 'sedan',
      licensePlate: 'ABC-123',
      speed: 65,
      speedLimit: 40,
      coordinates: { lat: 40.7128, lng: -74.0060 }
    }
  },
  {
    id: 'alert-002',
    type: 'warning',
    category: 'congestion',
    title: 'Traffic Congestion',
    description: 'Heavy traffic detected, estimated delay 15 minutes',
    location: 'Highway 101 North (Camera-12)',
    cameraId: 'CAM-012',
    timestamp: '2024-01-15T14:25:10Z',
    confidence: 87.8,
    status: 'active',
    metadata: {
      vehicleCount: 47,
      avgSpeed: 15,
      normalSpeed: 65,
      coordinates: { lat: 40.7589, lng: -73.9851 }
    }
  },
  {
    id: 'alert-003',
    type: 'critical',
    category: 'accident',
    title: 'Possible Accident',
    description: 'Vehicle stopped in traffic lane, emergency services notified',
    location: 'Broadway & 5th St (Camera-03)',
    cameraId: 'CAM-003',
    timestamp: '2024-01-15T14:22:45Z',
    confidence: 92.1,
    status: 'investigating',
    metadata: {
      vehicleCount: 2,
      emergencyServices: true,
      coordinates: { lat: 40.7505, lng: -73.9934 }
    }
  },
  {
    id: 'alert-004',
    type: 'info',
    category: 'system',
    title: 'Camera Maintenance',
    description: 'Scheduled maintenance completed on Camera-15',
    location: 'Park Ave & 3rd St (Camera-15)',
    cameraId: 'CAM-015',
    timestamp: '2024-01-15T13:45:30Z',
    confidence: 100,
    status: 'resolved',
    metadata: {
      maintenanceType: 'lens_cleaning',
      duration: '15 minutes',
      coordinates: { lat: 40.7614, lng: -73.9776 }
    }
  },
  {
    id: 'alert-005',
    type: 'warning',
    category: 'security',
    title: 'Suspicious Activity',
    description: 'Person detected in restricted area for extended period',
    location: 'Government Plaza (Camera-21)',
    cameraId: 'CAM-021',
    timestamp: '2024-01-15T13:30:15Z',
    confidence: 78.5,
    status: 'investigating',
    metadata: {
      duration: '12 minutes',
      personCount: 1,
      restrictedZone: 'Zone-A',
      coordinates: { lat: 40.7282, lng: -74.0776 }
    }
  }
];

// Memoized constants for better performance with design system colors
const ALERT_TYPE_STYLES = {
  critical: 'border-primary/20 bg-primary/5 hover:bg-primary/10',
  warning: 'border-accent/20 bg-accent/5 hover:bg-accent/10',
  info: 'border-secondary/20 bg-secondary/5 hover:bg-secondary/10'
} as const;

const ALERT_TYPE_ICONS = {
  critical: IconAlertTriangle,
  warning: IconAlertCircle,
  info: IconBolt
} as const;

const ALERT_TYPE_BADGES = {
  critical: 'destructive',
  warning: 'warning',
  info: 'secondary'
} as const;

const CATEGORY_ICONS = {
  traffic_violation: IconBolt,
  accident: IconAlertTriangle,
  congestion: IconClock,
  system: IconCamera,
  security: IconShield
} as const;

const ALERT_TYPE_COLORS = {
  critical: 'text-primary',
  warning: 'text-accent-foreground',
  info: 'text-secondary'
} as const;

// Optimized timestamp formatting with caching
const timestampCache = new Map<string, string>();

function formatTimestamp(timestamp: string): string {
  if (timestampCache.has(timestamp)) {
    return timestampCache.get(timestamp)!;
  }

  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));

  let result: string;
  if (diffMins < 1) result = 'Just now';
  else if (diffMins < 60) result = `${diffMins}m ago`;
  else {
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) result = `${diffHours}h ago`;
    else {
      const diffDays = Math.floor(diffHours / 24);
      result = `${diffDays}d ago`;
    }
  }

  timestampCache.set(timestamp, result);
  return result;
}

// Mock data provider with React 19 use() API
const getAlertsData = (): Promise<AlertData[]> => Promise.resolve(mockAlerts);

// Memoized Alert Item Component
const AlertItem = memo<{
  alert: AlertData;
  onSelect: (alert: AlertData) => void;
}>(({ alert, onSelect }) => {
  const handleClick = useCallback(() => {
    startTransition(() => {
      onSelect(alert);
    });
  }, [alert, onSelect]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  }, [handleClick]);

  const AlertIcon = ALERT_TYPE_ICONS[alert.type];
  const CategoryIcon = CATEGORY_ICONS[alert.category];
  const styleClass = ALERT_TYPE_STYLES[alert.type];
  const colorClass = ALERT_TYPE_COLORS[alert.type];

  return (
    <div
      className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 hover:shadow-soft hover:scale-[1.02] ${styleClass}`}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      aria-label={`Alert: ${alert.title}`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <div className="flex items-center gap-2 mt-1">
            <div className={`p-1.5 rounded-lg ${alert.type === 'critical' ? 'bg-primary/10' : alert.type === 'warning' ? 'bg-accent/10' : 'bg-secondary/10'}`}>
              <AlertIcon className={`h-4 w-4 ${colorClass}`} />
            </div>
            <CategoryIcon className="h-3 w-3 text-muted-foreground" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1.5">
              <h4 className="font-semibold text-sm text-foreground truncate">
                {alert.title}
              </h4>
              <Badge
                variant={ALERT_TYPE_BADGES[alert.type]}
                className="text-2xs px-2 py-0.5 h-5 font-medium"
              >
                {alert.confidence}%
              </Badge>
            </div>

            <p className="text-xs text-muted-foreground mb-3 line-clamp-2 leading-relaxed">
              {alert.description}
            </p>

            <div className="flex items-center gap-4 text-2xs text-muted-foreground">
              <div className="flex items-center gap-1.5">
                <IconMapPin className="h-3 w-3 text-secondary" />
                <span className="truncate font-medium">{alert.location}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <IconClock className="h-3 w-3 text-accent" />
                <span className="font-medium">{formatTimestamp(alert.timestamp)}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <Badge
            variant={alert.status === 'active' ? 'default' :
                    alert.status === 'investigating' ? 'secondary' : 'outline'}
            className="text-2xs shrink-0 font-medium"
          >
            {alert.status}
          </Badge>
          <div className={`w-2 h-2 rounded-full ${alert.status === 'active' ? 'bg-online animate-status-online' : alert.status === 'investigating' ? 'bg-maintenance' : 'bg-muted-foreground'}`} />
        </div>
      </div>
    </div>
  );
});

AlertItem.displayName = 'AlertItem';

// Loading skeleton component
const AlertSkeleton = memo(() => (
  <div className="space-y-2 p-4 pt-0">
    {Array.from({ length: 3 }, (_, i) => (
      <div key={i} className="p-4 rounded-lg border animate-pulse">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <div className="flex items-center gap-1 mt-0.5">
              <Skeleton className="h-4 w-4 rounded" />
              <Skeleton className="h-3 w-3 rounded" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <Skeleton className="h-4 w-32 rounded" />
                <Skeleton className="h-5 w-12 rounded" />
              </div>
              <Skeleton className="h-3 w-full mb-2 rounded" />
              <div className="flex items-center gap-3">
                <Skeleton className="h-3 w-20 rounded" />
                <Skeleton className="h-3 w-16 rounded" />
              </div>
            </div>
          </div>
          <Skeleton className="h-5 w-12 rounded" />
        </div>
      </div>
    ))}
  </div>
));

AlertSkeleton.displayName = 'AlertSkeleton';

// Alerts content component using React 19 use() API
function AlertsContent({ onAlertSelect }: { onAlertSelect: (alert: AlertData) => void }) {
  const alerts = use(getAlertsData());

  const sortedAlerts = useMemo(() => {
    const severityOrder = { critical: 0, warning: 1, info: 2 };
    return [...alerts].sort((a, b) =>
      severityOrder[a.type] - severityOrder[b.type]
    );
  }, [alerts]);

  const activeAlertsCount = useMemo(() =>
    alerts.filter(alert => alert.status === 'active').length,
    [alerts]
  );

  return (
    <>
      <CardHeader className="pb-4 border-b border-border/50">
        <CardTitle className="text-xl font-semibold flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <IconAlertCircle className="h-5 w-5 text-primary" />
          </div>
          <span className="text-foreground">Recent Alerts</span>
          <Badge variant="secondary" className="ml-auto font-medium">
            {activeAlertsCount} Active
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea style={{ height: '400px' }}>
          <div className="space-y-3 p-6 pt-4">
            {sortedAlerts.map((alert) => (
              <AlertItem
                key={alert.id}
                alert={alert}
                onSelect={onAlertSelect}
              />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </>
  );
}

export const AlertPanel = memo(({ maxHeight = "400px" }: AlertPanelProps) => {
  const [selectedAlert, setSelectedAlert] = useState<AlertData | null>(null);

  const handleAlertSelect = useCallback((alert: AlertData) => {
    setSelectedAlert(alert);
  }, []);

  const handleModalClose = useCallback(() => {
    setSelectedAlert(null);
  }, []);

  return (
    <Card className="h-full border-border/50 shadow-soft hover:shadow-medium transition-shadow duration-300">
      <Suspense fallback={<AlertSkeleton />}>
        <AlertsContent onAlertSelect={handleAlertSelect} />
      </Suspense>

      <AlertDetailsModal
        alert={selectedAlert}
        isOpen={selectedAlert !== null}
        onClose={handleModalClose}
      />
    </Card>
  );
});

AlertPanel.displayName = 'AlertPanel';
