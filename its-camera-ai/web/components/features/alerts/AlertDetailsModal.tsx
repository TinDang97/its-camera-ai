'use client';

import { useState, useCallback, memo, useMemo, startTransition } from 'react';
import { useTranslations } from 'next-intl';
import { useLocalizedFormat } from '@/components/ui/language-switcher';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  IconAlertTriangle,
  IconClock,
  IconMapPin,
  IconCamera,
  IconShield,
  IconBolt,
  IconEye,
  IconDownload,
  IconShare2,
  IconDots,
  IconPlayerPlay,
  IconPlayerPause,
  IconVolume,
  IconVolumeOff,
  IconMaximize,
  IconChevronDown,
  IconChevronUp,
  IconNavigation,
  IconCar,
  IconUsers,
  IconActivity,
  IconTrendingUp,
  IconAlertCircle,
  IconCircleCheck,
  IconX
} from '@tabler/icons-react';

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

interface AlertDetailsModalProps {
  alert: AlertData | null;
  isOpen: boolean;
  onClose: () => void;
}

// Mock detection data for object detection panel
const mockDetections = [
  { id: 1, type: 'vehicle', confidence: 95.2, bbox: [120, 80, 240, 160], details: { make: 'Toyota', color: 'Blue', licensePlate: 'ABC-123' } },
  { id: 2, type: 'person', confidence: 87.8, bbox: [300, 100, 340, 200], details: { age: 'Adult', clothing: 'Red jacket' } },
  { id: 3, type: 'traffic_sign', confidence: 92.1, bbox: [450, 60, 480, 100], details: { type: 'Speed Limit', value: '40 km/h' } }
];

// Mock timeline events
const mockTimelineEvents = [
  { time: '14:30:22', event: 'Speed violation detected', type: 'critical', details: 'Vehicle exceeded speed limit by 25 km/h' },
  { time: '14:30:20', event: 'Vehicle entered frame', type: 'info', details: 'Blue sedan entered monitoring zone' },
  { time: '14:30:15', event: 'Speed measurement started', type: 'info', details: 'Vehicle tracking initiated' },
  { time: '14:29:58', event: 'Camera active', type: 'info', details: 'Normal monitoring operation' }
];

const LocationMap = memo(({ coordinates, location }: { coordinates?: { lat: number; lng: number }, location: string }) => {
  const [mapExpanded, setMapExpanded] = useState(false);

  const toggleExpanded = useCallback(() => {
    startTransition(() => {
      setMapExpanded(!mapExpanded);
    });
  }, [mapExpanded]);

  const formattedCoordinates = useMemo(() => {
    return coordinates
      ? `${coordinates.lat.toFixed(4)}, ${coordinates.lng.toFixed(4)}`
      : null;
  }, [coordinates]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <IconMapPin className="h-4 w-4 text-secondary" />
            Location Details
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleExpanded}
          >
            {mapExpanded ? <IconChevronUp className="h-4 w-4" /> : <IconChevronDown className="h-4 w-4" />}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex items-start gap-2">
            <IconNavigation className="h-4 w-4 text-muted-foreground mt-0.5" />
            <div>
              <p className="text-sm font-medium">{location}</p>
              {formattedCoordinates && (
                <p className="text-xs text-gray-500">
                  {formattedCoordinates}
                </p>
              )}
            </div>
          </div>

          {mapExpanded && (
            <div className="mt-4">
              <div className="bg-gray-100 rounded-lg h-32 flex items-center justify-center border-2 border-dashed border-gray-300">
                <div className="text-center">
                  <IconMapPin className="h-6 w-6 text-muted-foreground mx-auto mb-2" />
                  <p className="text-xs text-gray-500">Interactive Map View</p>
                  <p className="text-xs text-gray-400">(Map integration placeholder)</p>
                </div>
              </div>
              <div className="flex gap-2 mt-2">
                <Button variant="outline" size="sm" className="text-xs">
                  <IconNavigation className="h-3 w-3 mr-1" />
                  Directions
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  <IconShare2 className="h-3 w-3 mr-1" />
                  Share Location
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
});

LocationMap.displayName = 'LocationMap';

const CameraStreamViewer = memo(({ cameraId }: { cameraId: string }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const togglePlaying = useCallback(() => {
    startTransition(() => {
      setIsPlaying(!isPlaying);
    });
  }, [isPlaying]);

  const toggleMuted = useCallback(() => {
    setIsMuted(!isMuted);
  }, [isMuted]);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(!isFullscreen);
  }, [isFullscreen]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <IconCamera className="h-4 w-4 text-online" />
          Camera Feed - {cameraId}
          <Badge variant="outline" className="ml-auto text-xs">
            Live
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative bg-black rounded-lg overflow-hidden">
          <div className="aspect-video bg-gray-900 flex items-center justify-center">
            <div className="text-center text-white">
              <IconCamera className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm opacity-75">Camera Stream</p>
              <p className="text-xs opacity-50">Stream placeholder</p>
            </div>
          </div>

          {/* Video Controls Overlay */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-3">
            <div className="flex items-center justify-between text-white">
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0 text-white hover:bg-white/20"
                  onClick={togglePlaying}
                >
                  {isPlaying ? <IconPlayerPause className="h-4 w-4" /> : <IconPlayerPlay className="h-4 w-4" />}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0 text-white hover:bg-white/20"
                  onClick={toggleMuted}
                >
                  {isMuted ? <IconVolumeOff className="h-4 w-4" /> : <IconVolume className="h-4 w-4" />}
                </Button>
                <span className="text-xs">14:30:22</span>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0 text-white hover:bg-white/20"
                  onClick={toggleFullscreen}
                >
                  <IconMaximize className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>

        <div className="flex gap-2 mt-3">
          <Button variant="outline" size="sm" className="text-xs">
            <IconDownload className="h-3 w-3 mr-1" />
            Download Clip
          </Button>
          <Button variant="outline" size="sm" className="text-xs">
            <IconEye className="h-3 w-3 mr-1" />
            View History
          </Button>
        </div>
      </CardContent>
    </Card>
  );
});

CameraStreamViewer.displayName = 'CameraStreamViewer';

function ObjectDetectionPanel() {
  const [expandedDetection, setExpandedDetection] = useState<number | null>(null);

  const getDetectionIcon = (type: string) => {
    switch (type) {
      case 'vehicle': return IconCar;
      case 'person': return IconUsers;
      case 'traffic_sign': return IconShield;
      default: return IconAlertCircle;
    }
  };

  const getDetectionColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600 bg-green-50 border-green-200';
    if (confidence >= 70) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <IconActivity className="h-4 w-4 text-accent" />
          Object Detection Results
          <Badge variant="secondary" className="ml-auto text-xs">
            {mockDetections.length} Objects
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {mockDetections.map((detection) => {
            const Icon = getDetectionIcon(detection.type);
            const isExpanded = expandedDetection === detection.id;

            return (
              <div
                key={detection.id}
                className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${getDetectionColor(detection.confidence)}`}
                onClick={() => setExpandedDetection(isExpanded ? null : detection.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="h-4 w-4" />
                    <span className="font-medium text-sm capitalize">
                      {detection.type.replace('_', ' ')}
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {detection.confidence}%
                    </Badge>
                  </div>
                  {isExpanded ? <IconChevronUp className="h-4 w-4" /> : <IconChevronDown className="h-4 w-4" />}
                </div>

                {isExpanded && (
                  <div className="mt-3 pt-2 border-t border-current/20">
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="font-medium">Bounding Box:</span>
                        <p className="text-xs opacity-75">
                          [{detection.bbox.join(', ')}]
                        </p>
                      </div>
                      <div>
                        <span className="font-medium">Details:</span>
                        <div className="space-y-1 opacity-75">
                          {Object.entries(detection.details).map(([key, value]) => (
                            <p key={key} className="text-xs">
                              <span className="capitalize">{key}:</span> {value}
                            </p>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function InteractiveTimeline() {
  const [selectedEvent, setSelectedEvent] = useState<number | null>(null);

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'critical': return IconAlertTriangle;
      case 'warning': return IconAlertCircle;
      case 'info': return IconCircleCheck;
      default: return IconActivity;
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'critical': return 'text-red-600 border-red-200';
      case 'warning': return 'text-yellow-600 border-yellow-200';
      case 'info': return 'text-blue-600 border-blue-200';
      default: return 'text-gray-600 border-gray-200';
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <IconClock className="h-4 w-4 text-secondary" />
          Event Timeline
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-4 top-0 bottom-0 w-px bg-gray-200"></div>

          <div className="space-y-3">
            {mockTimelineEvents.map((event, index) => {
              const Icon = getEventIcon(event.type);
              const isSelected = selectedEvent === index;

              return (
                <div
                  key={index}
                  className={`relative flex items-start gap-3 cursor-pointer transition-all duration-200 ${
                    isSelected ? 'bg-gray-50 -mx-2 px-2 py-1 rounded-lg' : ''
                  }`}
                  onClick={() => setSelectedEvent(isSelected ? null : index)}
                >
                  {/* Timeline dot */}
                  <div className={`relative z-10 flex items-center justify-center w-8 h-8 rounded-full border-2 bg-white ${getEventColor(event.type)}`}>
                    <Icon className="h-3 w-3" />
                  </div>

                  <div className="flex-1 min-w-0 pt-1">
                    <div className="flex items-center justify-between mb-1">
                      <p className="font-medium text-sm">{event.event}</p>
                      <span className="text-xs text-gray-500 font-mono">{event.time}</span>
                    </div>
                    <p className="text-xs text-gray-600">{event.details}</p>

                    {isSelected && (
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm" className="text-xs h-6">
                            View Details
                          </Button>
                          <Button variant="outline" size="sm" className="text-xs h-6">
                            Export
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export const AlertDetailsModal = memo<AlertDetailsModalProps>(({ alert, isOpen, onClose }) => {
  const t = useTranslations('Alerts');
  const { formatDateTime } = useLocalizedFormat();

  const handleOpenChange = useCallback((open: boolean) => {
    if (!open) {
      onClose();
    }
  }, [onClose]);

  const alertMetrics = useMemo(() => {
    if (!alert) return null;
    return {
      confidence: alert.confidence,
      camera: alert.cameraId,
      status: alert.status,
      timestamp: formatDateTime(new Date(alert.timestamp))
    };
  }, [alert, formatDateTime]);

  if (!alert) return null;

  const getAlertIcon = (type: AlertType) => {
    switch (type) {
      case 'critical': return IconAlertTriangle;
      case 'warning': return IconAlertCircle;
      case 'info': return IconBolt;
    }
  };

  const getAlertColor = (type: AlertType) => {
    switch (type) {
      case 'critical': return 'text-red-600';
      case 'warning': return 'text-yellow-600';
      case 'info': return 'text-blue-600';
    }
  };

  const AlertIcon = getAlertIcon(alert.type);

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden">
        <DialogHeader className="pb-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <AlertIcon className={`h-6 w-6 ${getAlertColor(alert.type)}`} />
              <div>
                <DialogTitle className="text-xl font-semibold">{alert.title}</DialogTitle>
                <DialogDescription className="mt-1 text-sm text-gray-600">
                  {alert.description}
                </DialogDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={alert.type === 'critical' ? 'destructive' : alert.type === 'warning' ? 'secondary' : 'outline'}>
                {alert.confidence}% Confidence
              </Badge>
              <Button variant="ghost" size="sm" onClick={onClose}>
                <IconX className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Quick Stats */}
          {alertMetrics && (
            <div className="grid grid-cols-4 gap-4 pt-4 border-t">
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{alertMetrics.confidence}%</p>
                <p className="text-xs text-gray-500">{t('details.confidence')}</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">{alertMetrics.camera}</p>
                <p className="text-xs text-gray-500">{t('details.camera')}</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-purple-600 capitalize">{t(`status.${alertMetrics.status}`)}</p>
                <p className="text-xs text-gray-500">Status</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-orange-600">
                  {alertMetrics.timestamp}
                </p>
                <p className="text-xs text-gray-500">{t('details.created')}</p>
              </div>
            </div>
          )}
        </DialogHeader>

        <Tabs defaultValue="overview" className="flex-1">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">{t('tabs.overview')}</TabsTrigger>
            <TabsTrigger value="detection">{t('tabs.detections')}</TabsTrigger>
            <TabsTrigger value="timeline">{t('tabs.timeline')}</TabsTrigger>
            <TabsTrigger value="actions">Actions</TabsTrigger>
          </TabsList>

          <ScrollArea className="h-[500px] mt-4">
            <TabsContent value="overview" className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <LocationMap
                  coordinates={alert.metadata?.coordinates}
                  location={alert.location}
                />
                <CameraStreamViewer cameraId={alert.cameraId} />
              </div>

              {/* Metadata */}
              {alert.metadata && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                      <IconTrendingUp className="h-4 w-4 text-muted-foreground" />
                      Additional Information
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      {Object.entries(alert.metadata).map(([key, value]) => {
                        if (key === 'coordinates') return null;
                        return (
                          <div key={key}>
                            <span className="font-medium capitalize text-gray-700">
                              {key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ')}:
                            </span>
                            <p className="text-gray-600 mt-1">
                              {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="detection">
              <ObjectDetectionPanel />
            </TabsContent>

            <TabsContent value="timeline">
              <InteractiveTimeline />
            </TabsContent>

            <TabsContent value="actions" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Available Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <Button variant="outline" className="justify-start">
                      <IconDownload className="h-4 w-4 mr-2" />
                      Export Report
                    </Button>
                    <Button variant="outline" className="justify-start">
                      <IconShare2 className="h-4 w-4 mr-2" />
                      Share Alert
                    </Button>
                    <Button variant="outline" className="justify-start">
                      <IconEye className="h-4 w-4 mr-2" />
                      View Similar
                    </Button>
                    <Button variant="outline" className="justify-start">
                      <IconAlertTriangle className="h-4 w-4 mr-2" />
                      Escalate
                    </Button>
                  </div>

                  <Separator />

                  <div className="space-y-2">
                    <Button variant="destructive" className="w-full justify-start">
                      <IconAlertTriangle className="h-4 w-4 mr-2" />
                      Mark as False Positive
                    </Button>
                    <Button variant="default" className="w-full justify-start">
                      <IconCircleCheck className="h-4 w-4 mr-2" />
                      Mark as Resolved
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
});

AlertDetailsModal.displayName = 'AlertDetailsModal';
