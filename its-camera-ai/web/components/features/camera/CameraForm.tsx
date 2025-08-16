'use client';

import React, { useState, useCallback, useEffect, useTransition } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  IconCamera,
  IconMapPin,
  IconSettings,
  IconCheck,
  IconX,
  IconLoader2,
  IconAlertTriangle,
  IconTestPipe,
  IconWifi,
  IconShield,
  IconPlus,
  IconEye,
  IconEyeOff,
} from '@tabler/icons-react';
import {
  camerasAPI,
  CameraCreate,
  CameraUpdate,
  Camera,
  CameraStatus,
  CameraType,
  StreamProtocol,
  APIError
} from '@/lib/api/cameras';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';

// Form validation schema for camera creation/editing
const cameraFormSchema = z.object({
  name: z.string()
    .min(1, 'Camera name is required')
    .max(100, 'Name must be less than 100 characters'),
  description: z.string()
    .max(500, 'Description must be less than 500 characters')
    .nullable()
    .optional(),
  location: z.string()
    .min(1, 'Location is required')
    .max(200, 'Location must be less than 200 characters'),
  coordinates: z.object({
    lat: z.number()
      .min(-90, 'Latitude must be between -90 and 90')
      .max(90, 'Latitude must be between -90 and 90'),
    lng: z.number()
      .min(-180, 'Longitude must be between -180 and 180')
      .max(180, 'Longitude must be between -180 and 180'),
    altitude: z.number().optional(),
  }).nullable().optional(),
  camera_type: z.enum(['fixed', 'ptz', 'speed', 'traffic'] as const),
  stream_url: z.string()
    .url('Please enter a valid stream URL')
    .refine(url => {
      const validProtocols = ['rtsp://', 'rtmp://', 'http://', 'https://'];
      return validProtocols.some(protocol => url.startsWith(protocol));
    }, {
      message: 'Stream URL must use RTSP, RTMP, HTTP, or HTTPS protocol'
    }),
  stream_protocol: z.enum(['rtsp', 'hls', 'webrtc', 'http'] as const),
  backup_stream_url: z.string().url().nullable().optional(),
  username: z.string().max(100).nullable().optional(),
  password: z.string().max(200).nullable().optional(),
  config: z.object({
    detection_zones: z.array(z.record(z.any())).optional(),
    analytics_enabled: z.boolean().optional(),
    recording_enabled: z.boolean().optional(),
    quality_settings: z.object({
      resolution: z.string(),
      fps: z.number().min(1).max(60),
      bitrate: z.number().min(100).max(50000),
    }).optional(),
    notifications: z.object({
      alerts_enabled: z.boolean(),
      email_notifications: z.boolean(),
    }).optional(),
  }),
  zone_id: z.string().max(50).nullable().optional(),
  tags: z.array(z.string()).default([]),
});

type CameraFormData = z.infer<typeof cameraFormSchema>;

interface CameraFormProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  mode: 'create' | 'edit';
  camera?: Camera; // For edit mode
  onSuccess?: (camera: Camera) => void;
  onError?: (error: Error) => void;
}

interface TestState {
  isConnecting: boolean;
  connectionStatus: 'idle' | 'testing' | 'success' | 'failed';
  errorMessage: string | null;
}

const INITIAL_TEST_STATE: TestState = {
  isConnecting: false,
  connectionStatus: 'idle',
  errorMessage: null,
};

const RESOLUTION_PRESETS = [
  { label: '4K (3840x2160)', value: '3840x2160' },
  { label: '1080p (1920x1080)', value: '1920x1080' },
  { label: '720p (1280x720)', value: '1280x720' },
  { label: '480p (640x480)', value: '640x480' },
  { label: 'Custom', value: 'custom' },
];

const FPS_OPTIONS = [
  { label: '60 FPS', value: 60 },
  { label: '30 FPS', value: 30 },
  { label: '25 FPS', value: 25 },
  { label: '15 FPS', value: 15 },
  { label: '10 FPS', value: 10 },
];

const BITRATE_PRESETS = [
  { label: 'Ultra (8000 kbps)', value: 8000 },
  { label: 'High (3000 kbps)', value: 3000 },
  { label: 'Medium (1500 kbps)', value: 1500 },
  { label: 'Low (500 kbps)', value: 500 },
];

export const CameraForm: React.FC<CameraFormProps> = ({
  isOpen,
  onOpenChange,
  mode,
  camera,
  onSuccess,
  onError,
}) => {
  const [isPending, startTransition] = useTransition();
  const [testState, setTestState] = useState<TestState>(INITIAL_TEST_STATE);
  const [activeTab, setActiveTab] = useState('basic');
  const [showPassword, setShowPassword] = useState(false);
  const [tagsInput, setTagsInput] = useState('');

  // Form setup
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting, isDirty },
    reset,
    watch,
    setValue,
    getValues,
    trigger,
  } = useForm<CameraFormData>({
    resolver: zodResolver(cameraFormSchema),
    mode: 'onBlur',
    defaultValues: {
      config: {
        analytics_enabled: true,
        recording_enabled: false,
        quality_settings: {
          resolution: '1920x1080',
          fps: 30,
          bitrate: 1500,
        },
        notifications: {
          alerts_enabled: true,
          email_notifications: false,
        },
      },
      tags: [],
      camera_type: 'fixed',
      stream_protocol: 'rtsp',
    },
  });

  // Watch form values for dynamic updates
  const streamProtocol = watch('stream_protocol');
  const cameraType = watch('camera_type');

  // Initialize form with camera data for edit mode
  useEffect(() => {
    if (mode === 'edit' && camera && isOpen) {
      const cameraData: Partial<CameraFormData> = {
        name: camera.name,
        description: camera.description,
        location: camera.location,
        coordinates: camera.coordinates,
        camera_type: camera.camera_type,
        stream_url: camera.stream_url,
        stream_protocol: camera.stream_protocol,
        backup_stream_url: camera.backup_stream_url,
        config: {
          analytics_enabled: camera.config?.analytics_enabled ?? true,
          recording_enabled: camera.config?.recording_enabled ?? false,
          quality_settings: camera.config?.quality_settings || {
            resolution: '1920x1080',
            fps: 30,
            bitrate: 1500,
          },
          notifications: camera.config?.notifications || {
            alerts_enabled: true,
            email_notifications: false,
          },
        },
        zone_id: camera.zone_id,
        tags: camera.tags || [],
      };

      reset(cameraData);
      setTagsInput(camera.tags?.join(', ') || '');
    } else if (mode === 'create' && isOpen) {
      // Reset to default values for create mode
      reset({
        config: {
          analytics_enabled: true,
          recording_enabled: false,
          quality_settings: {
            resolution: '1920x1080',
            fps: 30,
            bitrate: 1500,
          },
          notifications: {
            alerts_enabled: true,
            email_notifications: false,
          },
        },
        tags: [],
        camera_type: 'fixed',
        stream_protocol: 'rtsp',
      });
      setTagsInput('');
    }
  }, [mode, camera, isOpen, reset]);

  // Handle tags input change
  const handleTagsChange = useCallback((value: string) => {
    setTagsInput(value);
    const tags = value.split(',').map(tag => tag.trim()).filter(tag => tag.length > 0);
    setValue('tags', tags, { shouldValidate: true });
  }, [setValue]);

  // Test connection to camera stream
  const handleTestConnection = useCallback(async () => {
    const streamUrl = getValues('stream_url');
    if (!streamUrl) {
      onError?.(new Error('Please enter a stream URL to test the connection'));
      return;
    }

    // Validate stream URL first
    const isValid = await trigger('stream_url');
    if (!isValid) {
      onError?.(new Error('Please enter a valid stream URL'));
      return;
    }

    setTestState({
      isConnecting: true,
      connectionStatus: 'testing',
      errorMessage: null,
    });

    try {
      // Simulate connection test - in real implementation, this would call the backend
      await new Promise(resolve => setTimeout(resolve, 2000));

      // For demo purposes, we'll randomly succeed or fail
      if (Math.random() > 0.3) {
        setTestState({
          isConnecting: false,
          connectionStatus: 'success',
          errorMessage: null,
        });
      } else {
        throw new Error('Unable to connect to stream. Please check the URL and network connectivity.');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Connection test failed';
      setTestState({
        isConnecting: false,
        connectionStatus: 'failed',
        errorMessage,
      });
    }
  }, [getValues, trigger, onError]);

  // Handle form submission
  const onSubmit = useCallback(async (data: CameraFormData) => {
    startTransition(async () => {
      try {
        let result: Camera;

        if (mode === 'create') {
          result = await camerasAPI.create(data as CameraCreate);
        } else if (mode === 'edit' && camera) {
          result = await camerasAPI.update(camera.id, data as CameraUpdate);
        } else {
          throw new Error('Invalid form mode or missing camera data');
        }

        onSuccess?.(result);
        onOpenChange(false);
      } catch (error) {
        const errorMessage = error instanceof APIError
          ? error.message
          : `Failed to ${mode} camera`;
        onError?.(new Error(errorMessage));
      }
    });
  }, [mode, camera, onSuccess, onError, onOpenChange]);

  // Handle modal close with unsaved changes warning
  const handleClose = useCallback(() => {
    if (isDirty && !isPending && !isSubmitting) {
      const confirmed = window.confirm(
        'You have unsaved changes. Are you sure you want to close without saving?'
      );
      if (!confirmed) return;
    }
    onOpenChange(false);
    setTestState(INITIAL_TEST_STATE);
  }, [isDirty, isPending, isSubmitting, onOpenChange]);

  const isLoading = isPending || isSubmitting;
  const title = mode === 'create' ? 'Add New Camera' : 'Edit Camera';
  const submitText = mode === 'create' ? 'Add Camera' : 'Save Changes';

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {mode === 'create' ? (
              <IconPlus className="w-5 h-5 text-primary" />
            ) : (
              <IconCamera className="w-5 h-5 text-primary" />
            )}
            <span>{title}</span>
          </DialogTitle>
          <DialogDescription>
            {mode === 'create'
              ? 'Configure a new camera for monitoring and analytics'
              : 'Update camera settings and configuration'
            }
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic" className="flex items-center gap-1">
                <IconCamera className="w-4 h-4" />
                <span className="hidden sm:inline">Basic</span>
              </TabsTrigger>
              <TabsTrigger value="stream" className="flex items-center gap-1">
                <IconWifi className="w-4 h-4" />
                <span className="hidden sm:inline">Stream</span>
              </TabsTrigger>
              <TabsTrigger value="config" className="flex items-center gap-1">
                <IconSettings className="w-4 h-4" />
                <span className="hidden sm:inline">Config</span>
              </TabsTrigger>
              <TabsTrigger value="auth" className="flex items-center gap-1">
                <IconShield className="w-4 h-4" />
                <span className="hidden sm:inline">Auth</span>
              </TabsTrigger>
            </TabsList>

            {/* Basic Information Tab */}
            <TabsContent value="basic" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Camera Name *</Label>
                  <Input
                    id="name"
                    placeholder="e.g., Main Entrance Camera"
                    disabled={isLoading}
                    {...register('name')}
                    className={errors.name ? 'border-destructive' : ''}
                  />
                  {errors.name && (
                    <p className="text-sm text-destructive">{errors.name.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="camera_type">Camera Type *</Label>
                  <Select
                    value={cameraType}
                    onValueChange={(value) => setValue('camera_type', value as CameraType, { shouldValidate: true })}
                  >
                    <SelectTrigger className={errors.camera_type ? 'border-destructive' : ''}>
                      <SelectValue placeholder="Select camera type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fixed">Fixed Camera</SelectItem>
                      <SelectItem value="ptz">PTZ Camera</SelectItem>
                      <SelectItem value="speed">Speed Camera</SelectItem>
                      <SelectItem value="traffic">Traffic Camera</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.camera_type && (
                    <p className="text-sm text-destructive">{errors.camera_type.message}</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="location">Location *</Label>
                <Input
                  id="location"
                  placeholder="e.g., Building A - Main Entrance"
                  disabled={isLoading}
                  {...register('location')}
                  className={errors.location ? 'border-destructive' : ''}
                />
                {errors.location && (
                  <p className="text-sm text-destructive">{errors.location.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Optional description for the camera"
                  disabled={isLoading}
                  {...register('description')}
                  className={errors.description ? 'border-destructive' : ''}
                />
                {errors.description && (
                  <p className="text-sm text-destructive">{errors.description.message}</p>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="lat">Latitude</Label>
                  <Input
                    id="lat"
                    type="number"
                    step="any"
                    placeholder="e.g., 40.7128"
                    disabled={isLoading}
                    {...register('coordinates.lat', { valueAsNumber: true })}
                    className={errors.coordinates?.lat ? 'border-destructive' : ''}
                  />
                  {errors.coordinates?.lat && (
                    <p className="text-sm text-destructive">{errors.coordinates.lat.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="lng">Longitude</Label>
                  <Input
                    id="lng"
                    type="number"
                    step="any"
                    placeholder="e.g., -74.0060"
                    disabled={isLoading}
                    {...register('coordinates.lng', { valueAsNumber: true })}
                    className={errors.coordinates?.lng ? 'border-destructive' : ''}
                  />
                  {errors.coordinates?.lng && (
                    <p className="text-sm text-destructive">{errors.coordinates.lng.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="altitude">Altitude (optional)</Label>
                  <Input
                    id="altitude"
                    type="number"
                    step="any"
                    placeholder="e.g., 10"
                    disabled={isLoading}
                    {...register('coordinates.altitude', { valueAsNumber: true })}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="tags">Tags</Label>
                <Input
                  id="tags"
                  placeholder="entrance, security, main-building (comma-separated)"
                  value={tagsInput}
                  onChange={(e) => handleTagsChange(e.target.value)}
                  disabled={isLoading}
                />
                <p className="text-xs text-muted-foreground">
                  Comma-separated tags for organization and filtering
                </p>
              </div>
            </TabsContent>

            {/* Stream Configuration Tab */}
            <TabsContent value="stream" className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label htmlFor="stream_url">Stream URL *</Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleTestConnection}
                    disabled={isLoading || testState.isConnecting}
                  >
                    {testState.isConnecting ? (
                      <IconLoader2 className="w-3 h-3 mr-1 animate-spin" />
                    ) : (
                      <IconTestPipe className="w-3 h-3 mr-1" />
                    )}
                    Test
                  </Button>
                  {testState.connectionStatus === 'success' && (
                    <Badge variant="default" className="text-xs">
                      <IconCheck className="w-3 h-3 mr-1" />
                      Connected
                    </Badge>
                  )}
                  {testState.connectionStatus === 'failed' && (
                    <Badge variant="destructive" className="text-xs">
                      <IconX className="w-3 h-3 mr-1" />
                      Failed
                    </Badge>
                  )}
                </div>
                <Input
                  id="stream_url"
                  type="url"
                  placeholder="rtsp://192.168.1.100:554/stream"
                  disabled={isLoading}
                  {...register('stream_url')}
                  className={errors.stream_url ? 'border-destructive' : ''}
                />
                {errors.stream_url && (
                  <p className="text-sm text-destructive">{errors.stream_url.message}</p>
                )}
                {testState.errorMessage && (
                  <p className="text-sm text-destructive">{testState.errorMessage}</p>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="stream_protocol">Stream Protocol *</Label>
                  <Select
                    value={streamProtocol}
                    onValueChange={(value) => setValue('stream_protocol', value as StreamProtocol, { shouldValidate: true })}
                  >
                    <SelectTrigger className={errors.stream_protocol ? 'border-destructive' : ''}>
                      <SelectValue placeholder="Select protocol" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rtsp">RTSP (Real-Time Streaming)</SelectItem>
                      <SelectItem value="hls">HLS (HTTP Live Streaming)</SelectItem>
                      <SelectItem value="webrtc">WebRTC (Real-Time Communication)</SelectItem>
                      <SelectItem value="http">HTTP (Web Streaming)</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.stream_protocol && (
                    <p className="text-sm text-destructive">{errors.stream_protocol.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="backup_stream_url">Backup Stream URL</Label>
                  <Input
                    id="backup_stream_url"
                    type="url"
                    placeholder="rtsp://backup.example.com/stream"
                    disabled={isLoading}
                    {...register('backup_stream_url')}
                    className={errors.backup_stream_url ? 'border-destructive' : ''}
                  />
                  {errors.backup_stream_url && (
                    <p className="text-sm text-destructive">{errors.backup_stream_url.message}</p>
                  )}
                </div>
              </div>
            </TabsContent>

            {/* Configuration Tab */}
            <TabsContent value="config" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="resolution">Resolution</Label>
                  <Select
                    value={watch('config.quality_settings.resolution')}
                    onValueChange={(value) => setValue('config.quality_settings.resolution', value, { shouldValidate: true })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select resolution" />
                    </SelectTrigger>
                    <SelectContent>
                      {RESOLUTION_PRESETS.map((preset) => (
                        <SelectItem key={preset.value} value={preset.value}>
                          {preset.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="fps">Frame Rate</Label>
                  <Select
                    value={String(watch('config.quality_settings.fps'))}
                    onValueChange={(value) => setValue('config.quality_settings.fps', Number(value), { shouldValidate: true })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select FPS" />
                    </SelectTrigger>
                    <SelectContent>
                      {FPS_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={String(option.value)}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="bitrate">Bitrate</Label>
                  <Select
                    value={String(watch('config.quality_settings.bitrate'))}
                    onValueChange={(value) => setValue('config.quality_settings.bitrate', Number(value), { shouldValidate: true })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select bitrate" />
                    </SelectTrigger>
                    <SelectContent>
                      {BITRATE_PRESETS.map((preset) => (
                        <SelectItem key={preset.value} value={String(preset.value)}>
                          {preset.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Separator />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="analytics_enabled">Analytics</Label>
                      <p className="text-xs text-muted-foreground">
                        Enable AI-powered traffic analytics
                      </p>
                    </div>
                    <Switch
                      id="analytics_enabled"
                      checked={watch('config.analytics_enabled')}
                      onCheckedChange={(checked) => setValue('config.analytics_enabled', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="recording_enabled">Recording</Label>
                      <p className="text-xs text-muted-foreground">
                        Enable continuous video recording
                      </p>
                    </div>
                    <Switch
                      id="recording_enabled"
                      checked={watch('config.recording_enabled')}
                      onCheckedChange={(checked) => setValue('config.recording_enabled', checked)}
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="alerts_enabled">Alerts</Label>
                      <p className="text-xs text-muted-foreground">
                        Enable incident and anomaly alerts
                      </p>
                    </div>
                    <Switch
                      id="alerts_enabled"
                      checked={watch('config.notifications.alerts_enabled')}
                      onCheckedChange={(checked) => setValue('config.notifications.alerts_enabled', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="email_notifications">Email Notifications</Label>
                      <p className="text-xs text-muted-foreground">
                        Send alerts via email
                      </p>
                    </div>
                    <Switch
                      id="email_notifications"
                      checked={watch('config.notifications.email_notifications')}
                      onCheckedChange={(checked) => setValue('config.notifications.email_notifications', checked)}
                    />
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Authentication Tab */}
            <TabsContent value="auth" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input
                    id="username"
                    placeholder="Camera username"
                    disabled={isLoading}
                    {...register('username')}
                  />
                  <p className="text-xs text-muted-foreground">
                    Required for cameras with authentication
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Camera password"
                      disabled={isLoading}
                      {...register('password')}
                      className="pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? (
                        <IconEyeOff className="h-4 w-4" />
                      ) : (
                        <IconEye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {mode === 'edit' ? 'Leave empty to keep existing password' : 'Optional if camera has no authentication'}
                  </p>
                </div>
              </div>

              <Alert>
                <IconShield className="h-4 w-4" />
                <div>
                  <h4 className="font-medium">Security Notice</h4>
                  <p className="text-sm mt-1">
                    Credentials are encrypted and stored securely. They are only used for camera authentication.
                  </p>
                </div>
              </Alert>
            </TabsContent>
          </Tabs>

          <DialogFooter className="gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={isLoading}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isLoading || (!isDirty && mode === 'edit')}
            >
              {isLoading ? (
                <>
                  <IconLoader2 className="w-4 h-4 mr-2 animate-spin" />
                  {mode === 'create' ? 'Adding...' : 'Saving...'}
                </>
              ) : (
                <>
                  <IconCheck className="w-4 h-4 mr-2" />
                  {submitText}
                </>
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CameraForm;
