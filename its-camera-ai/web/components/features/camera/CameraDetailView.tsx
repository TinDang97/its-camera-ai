'use client';

import React, { useState, useCallback, useEffect, useTransition, useMemo } from 'react';
import {
  IconCamera,
  IconMapPin,
  IconSettings,
  IconArrowLeft,
  IconEdit,
  IconTrash,
  IconPlayerPlay,
  IconPlayerStop,
  IconPlayerRecord,
  IconDownload,
  IconWifi,
  IconWifiOff,
  IconActivity,
  IconAlertTriangle,
  IconRefresh,
  IconMaximize,
  IconMinimize,
  IconFullscreen,
  IconVolume,
  IconVolumeOff,
  IconChartArea,
  IconClock,
  IconEye,
  IconVideo,
  IconShield,
  IconTool,
} from '@tabler/icons-react';
import {
  Camera,
  CameraStats,
  StreamHealth,
  CameraStatus,
  camerasAPI,
  cameraUtils
} from '@/lib/api/cameras';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert } from '@/components/ui/alert';
import { Card } from '@/components/ui/card';
import { CameraForm } from './CameraForm';

interface CameraDetailViewProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  camera: Camera;
  onCameraUpdate?: (camera: Camera) => void;
  onCameraDelete?: (cameraId: string) => void;
}

interface StreamControlState {
  isPlaying: boolean;
  isRecording: boolean;
  isFullscreen: boolean;
  isMuted: boolean;
  quality: 'auto' | 'high' | 'medium' | 'low';
}

const INITIAL_STREAM_STATE: StreamControlState = {
  isPlaying: false,
  isRecording: false,
  isFullscreen: false,
  isMuted: false,
  quality: 'auto',
};

export const CameraDetailView: React.FC<CameraDetailViewProps> = ({
  isOpen,
  onOpenChange,
  camera,
  onCameraUpdate,
  onCameraDelete,
}) => {
  const [isPending, startTransition] = useTransition();
  const [activeTab, setActiveTab] = useState('live');
  const [streamState, setStreamState] = useState<StreamControlState>(INITIAL_STREAM_STATE);
  const [streamHealth, setStreamHealth] = useState<StreamHealth | null>(null);
  const [cameraStats, setCameraStats] = useState<CameraStats | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Fetch stream health and stats when camera changes
  useEffect(() => {
    if (isOpen && camera) {
      fetchStreamHealth();
      fetchCameraStats();
    }
  }, [isOpen, camera]);

  const fetchStreamHealth = useCallback(async () => {
    try {
      const health = await camerasAPI.getStreamHealth(camera.id);
      setStreamHealth(health);
    } catch (error) {
      console.error('Failed to fetch stream health:', error);
    }
  }, [camera.id]);

  const fetchCameraStats = useCallback(async () => {
    try {
      const stats = await camerasAPI.getStats(camera.id, 7);
      setCameraStats(stats);
    } catch (error) {
      console.error('Failed to fetch camera stats:', error);
    }
  }, [camera.id]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    startTransition(async () => {
      try {
        await Promise.all([
          fetchStreamHealth(),
          fetchCameraStats(),
        ]);
      } catch (error) {
        console.error('Failed to refresh camera data:', error);
      } finally {
        setRefreshing(false);
      }
    });
  }, [fetchStreamHealth, fetchCameraStats]);

  const handleStreamControl = useCallback(async (action: 'start' | 'stop' | 'restart') => {
    startTransition(async () => {
      try {
        await camerasAPI.controlStream(camera.id, action);

        // Update local stream state
        setStreamState(prev => ({
          ...prev,
          isPlaying: action === 'start' || action === 'restart',
        }));

        // Refresh health status
        await fetchStreamHealth();
      } catch (error) {
        console.error(`Failed to ${action} stream:`, error);
      }
    });
  }, [camera.id, fetchStreamHealth]);

  const handleRecordingToggle = useCallback(() => {
    setStreamState(prev => ({
      ...prev,
      isRecording: !prev.isRecording,
    }));
    // In real implementation, this would call the recording API
  }, []);

  const handleEdit = useCallback((updatedCamera: Camera) => {
    onCameraUpdate?.(updatedCamera);
    setShowEditModal(false);
  }, [onCameraUpdate]);

  const handleDelete = useCallback(async () => {
    startTransition(async () => {
      try {
        await camerasAPI.delete(camera.id);
        onCameraDelete?.(camera.id);
        onOpenChange(false);
        setShowDeleteDialog(false);
      } catch (error) {
        console.error('Failed to delete camera:', error);
      }
    });
  }, [camera.id, onCameraDelete, onOpenChange]);

  // Calculate health status
  const healthStatus = useMemo(() => {
    if (!streamHealth) return { color: 'gray', text: 'Unknown' };
    if (streamHealth.is_healthy) return { color: 'green', text: 'Healthy' };
    return { color: 'red', text: 'Unhealthy' };
  }, [streamHealth]);

  // Calculate uptime
  const uptimeText = useMemo(() => {
    if (!cameraStats) return 'N/A';
    return `${cameraStats.uptime_percentage.toFixed(1)}%`;
  }, [cameraStats]);

  const getStatusIcon = (status: CameraStatus) => {
    switch (status) {
      case 'online':
      case 'streaming':
        return <IconWifi className="w-4 h-4 text-green-600" />;
      case 'offline':
      case 'stopped':
        return <IconWifiOff className="w-4 h-4 text-red-600" />;
      case 'maintenance':
        return <IconTool className="w-4 h-4 text-yellow-600" />;
      default:
        return <IconWifi className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <>
      <Dialog open={isOpen} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onOpenChange(false)}
                  className="p-1"
                >
                  <IconArrowLeft className="w-4 h-4" />
                </Button>
                <div className="flex items-center gap-2">
                  <IconCamera className="w-5 h-5 text-primary" />
                  <DialogTitle className="text-xl">{camera.name}</DialogTitle>
                  <Badge variant={camera.status === 'online' ? 'default' : 'secondary'}>
                    {getStatusIcon(camera.status)}
                    <span className="ml-1">{camera.status}</span>
                  </Badge>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefresh}
                  disabled={refreshing || isPending}
                >
                  <IconRefresh className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowEditModal(true)}
                >
                  <IconEdit className="w-4 h-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowDeleteDialog(true)}
                  className="text-destructive hover:text-destructive"
                >
                  <IconTrash className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <DialogDescription className="flex items-center gap-4 text-sm">
              <span className="flex items-center gap-1">
                <IconMapPin className="w-4 h-4" />
                {camera.location}
              </span>
              <span className="flex items-center gap-1">
                <IconActivity className="w-4 h-4" />
                {cameraUtils.formatTimeSince(camera.last_seen_at)}
              </span>
              {streamHealth && (
                <span className="flex items-center gap-1">
                  <div className={`w-2 h-2 rounded-full bg-${healthStatus.color}-500`} />
                  {streamHealth.response_time_ms}ms
                </span>
              )}
            </DialogDescription>
          </DialogHeader>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="live" className="flex items-center gap-1">
                <IconVideo className="w-4 h-4" />
                <span className="hidden sm:inline">Live Stream</span>
              </TabsTrigger>
              <TabsTrigger value="analytics" className="flex items-center gap-1">
                <IconChartArea className="w-4 h-4" />
                <span className="hidden sm:inline">Analytics</span>
              </TabsTrigger>
              <TabsTrigger value="health" className="flex items-center gap-1">
                <IconActivity className="w-4 h-4" />
                <span className="hidden sm:inline">Health</span>
              </TabsTrigger>
              <TabsTrigger value="settings" className="flex items-center gap-1">
                <IconSettings className="w-4 h-4" />
                <span className="hidden sm:inline">Settings</span>
              </TabsTrigger>
            </TabsList>

            {/* Live Stream Tab */}
            <TabsContent value="live" className="space-y-4">
              <Card className="p-4">
                <div className="aspect-video bg-gray-900 rounded-lg relative overflow-hidden">
                  {/* Video placeholder - in real implementation, this would be a video component */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-white">
                      <IconVideo className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg">Live Stream</p>
                      <p className="text-sm opacity-75">{camera.stream_url}</p>
                    </div>
                  </div>

                  {/* Stream controls overlay */}
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleStreamControl(streamState.isPlaying ? 'stop' : 'start')}
                          className="text-white hover:bg-white/20"
                        >
                          {streamState.isPlaying ? (
                            <IconPlayerStop className="w-5 h-5" />
                          ) : (
                            <IconPlayerPlay className="w-5 h-5" />
                          )}
                        </Button>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleRecordingToggle}
                          className={`text-white hover:bg-white/20 ${
                            streamState.isRecording ? 'text-red-400' : ''
                          }`}
                        >
                          <IconPlayerRecord className="w-5 h-5" />
                        </Button>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setStreamState(prev => ({ ...prev, isMuted: !prev.isMuted }))}
                          className="text-white hover:bg-white/20"
                        >
                          {streamState.isMuted ? (
                            <IconVolumeOff className="w-5 h-5" />
                          ) : (
                            <IconVolume className="w-5 h-5" />
                          )}
                        </Button>
                      </div>

                      <div className="flex items-center gap-2">
                        <select
                          value={streamState.quality}
                          onChange={(e) => setStreamState(prev => ({
                            ...prev,
                            quality: e.target.value as StreamControlState['quality']
                          }))}
                          className="bg-black/50 text-white text-sm rounded px-2 py-1 border-none"
                        >
                          <option value="auto">Auto</option>
                          <option value="high">High</option>
                          <option value="medium">Medium</option>
                          <option value="low">Low</option>
                        </select>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setStreamState(prev => ({ ...prev, isFullscreen: !prev.isFullscreen }))}
                          className="text-white hover:bg-white/20"
                        >
                          {streamState.isFullscreen ? (
                            <IconMinimize className="w-5 h-5" />
                          ) : (
                            <IconFullscreen className="w-5 h-5" />
                          )}
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Stream info */}
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Protocol</p>
                    <p className="font-medium uppercase">{camera.stream_protocol}</p>
                  </div>
                  {streamHealth && (
                    <>
                      <div>
                        <p className="text-muted-foreground">Resolution</p>
                        <p className="font-medium">{streamHealth.resolution || 'N/A'}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Frame Rate</p>
                        <p className="font-medium">{streamHealth.frame_rate ? `${streamHealth.frame_rate} fps` : 'N/A'}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Bitrate</p>
                        <p className="font-medium">{streamHealth.bitrate_kbps ? `${streamHealth.bitrate_kbps} kbps` : 'N/A'}</p>
                      </div>
                    </>
                  )}
                </div>
              </Card>
            </TabsContent>

            {/* Analytics Tab */}
            <TabsContent value="analytics" className="space-y-4">
              {cameraStats ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <IconEye className="w-4 h-4 text-blue-500" />
                      <h3 className="font-medium">Frames Processed</h3>
                    </div>
                    <p className="text-2xl font-bold">{cameraStats.frames_processed.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">Last 7 days</p>
                  </Card>

                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <IconCamera className="w-4 h-4 text-green-500" />
                      <h3 className="font-medium">Vehicles Detected</h3>
                    </div>
                    <p className="text-2xl font-bold">{cameraStats.vehicles_detected.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">Last 7 days</p>
                  </Card>

                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <IconAlertTriangle className="w-4 h-4 text-orange-500" />
                      <h3 className="font-medium">Incidents</h3>
                    </div>
                    <p className="text-2xl font-bold">{cameraStats.incidents_detected}</p>
                    <p className="text-xs text-muted-foreground">Last 7 days</p>
                  </Card>

                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <IconClock className="w-4 h-4 text-purple-500" />
                      <h3 className="font-medium">Processing Time</h3>
                    </div>
                    <p className="text-2xl font-bold">{cameraStats.avg_processing_time.toFixed(1)}ms</p>
                    <p className="text-xs text-muted-foreground">Average</p>
                  </Card>
                </div>
              ) : (
                <Card className="p-8 text-center">
                  <IconChartArea className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">Loading analytics data...</p>
                </Card>
              )}
            </TabsContent>

            {/* Health Tab */}
            <TabsContent value="health" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4">
                  <h3 className="font-medium mb-3">Connection Status</h3>
                  {streamHealth ? (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Health Status</span>
                        <Badge variant={streamHealth.is_healthy ? 'default' : 'destructive'}>
                          {healthStatus.text}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Response Time</span>
                        <span className="font-medium">{streamHealth.response_time_ms}ms</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Last Checked</span>
                        <span className="font-medium">
                          {cameraUtils.formatTimeSince(streamHealth.last_checked)}
                        </span>
                      </div>
                      {streamHealth.error_message && (
                        <Alert variant="destructive">
                          <IconAlertTriangle className="h-4 w-4" />
                          <div>
                            <h4 className="font-medium">Connection Error</h4>
                            <p className="text-sm mt-1">{streamHealth.error_message}</p>
                          </div>
                        </Alert>
                      )}
                    </div>
                  ) : (
                    <p className="text-muted-foreground">Loading health data...</p>
                  )}
                </Card>

                <Card className="p-4">
                  <h3 className="font-medium mb-3">System Status</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Uptime</span>
                      <span className="font-medium">{uptimeText}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Camera Type</span>
                      <span className="font-medium capitalize">{camera.camera_type}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Active</span>
                      <Badge variant={camera.is_active ? 'default' : 'secondary'}>
                        {camera.is_active ? 'Yes' : 'No'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Zone</span>
                      <span className="font-medium">{camera.zone_id || 'Default'}</span>
                    </div>
                  </div>
                </Card>
              </div>
            </TabsContent>

            {/* Settings Tab */}
            <TabsContent value="settings" className="space-y-4">
              <Card className="p-4">
                <h3 className="font-medium mb-3">Camera Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Name</p>
                    <p className="font-medium">{camera.name}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Location</p>
                    <p className="font-medium">{camera.location}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Stream URL</p>
                    <p className="font-medium break-all">{camera.stream_url}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Protocol</p>
                    <p className="font-medium uppercase">{camera.stream_protocol}</p>
                  </div>
                  {camera.coordinates && (
                    <>
                      <div>
                        <p className="text-muted-foreground">Coordinates</p>
                        <p className="font-medium">
                          {camera.coordinates.lat.toFixed(6)}, {camera.coordinates.lng.toFixed(6)}
                        </p>
                      </div>
                      {camera.coordinates.altitude && (
                        <div>
                          <p className="text-muted-foreground">Altitude</p>
                          <p className="font-medium">{camera.coordinates.altitude}m</p>
                        </div>
                      )}
                    </>
                  )}
                  <div>
                    <p className="text-muted-foreground">Created</p>
                    <p className="font-medium">
                      {new Date(camera.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Last Updated</p>
                    <p className="font-medium">
                      {new Date(camera.updated_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>

                {camera.tags && camera.tags.length > 0 && (
                  <div className="mt-4">
                    <p className="text-muted-foreground text-sm mb-2">Tags</p>
                    <div className="flex flex-wrap gap-1">
                      {camera.tags.map((tag, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {camera.description && (
                  <div className="mt-4">
                    <p className="text-muted-foreground text-sm mb-2">Description</p>
                    <p className="text-sm">{camera.description}</p>
                  </div>
                )}
              </Card>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>

      {/* Edit Camera Modal */}
      <CameraForm
        isOpen={showEditModal}
        onOpenChange={setShowEditModal}
        mode="edit"
        camera={camera}
        onSuccess={handleEdit}
        onError={(error) => console.error('Edit error:', error)}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <IconTrash className="w-5 h-5 text-destructive" />
              Delete Camera
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{camera.name}"? This action cannot be undone and will permanently remove all camera data, recordings, and analytics.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
              disabled={isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={isPending}
            >
              {isPending ? 'Deleting...' : 'Delete Camera'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default CameraDetailView;
