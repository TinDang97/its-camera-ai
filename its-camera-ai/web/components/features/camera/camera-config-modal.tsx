'use client'

import React, { useState, useCallback, useEffect, useOptimistic, useTransition } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  IconCamera,
  IconSettings,
  IconCheck,
  IconX,
  IconLoader2,
  IconAlertTriangle,
  IconTestPipe,
  IconWifi,
  IconMapPin,
  IconAdjustments,
  IconShield
} from '@tabler/icons-react'
import { cameraUtils, Camera, APIError } from '@/lib/api'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { useToast } from '@/hooks/use-toast'

// Form validation schema
const cameraConfigSchema = z.object({
  name: z.string()
    .min(1, 'Camera name is required')
    .max(100, 'Name must be less than 100 characters'),
  location: z.string()
    .min(1, 'Location is required')
    .max(200, 'Location must be less than 200 characters'),
  stream_url: z.string()
    .url('Please enter a valid stream URL')
    .refine(url => url.startsWith('rtsp://') || url.startsWith('rtmp://') || url.startsWith('http'), {
      message: 'Stream URL must be RTSP, RTMP, or HTTP protocol'
    }),
  resolution: z.object({
    width: z.number().min(320, 'Minimum width is 320px').max(7680, 'Maximum width is 7680px'),
    height: z.number().min(240, 'Minimum height is 240px').max(4320, 'Maximum height is 4320px')
  }),
  fps: z.number()
    .min(1, 'Minimum FPS is 1')
    .max(60, 'Maximum FPS is 60'),
  quality: z.enum(['low', 'medium', 'high', 'ultra']),
  encoding: z.enum(['h264', 'h265', 'mjpeg']),
  bitrate: z.number()
    .min(100, 'Minimum bitrate is 100 kbps')
    .max(50000, 'Maximum bitrate is 50 Mbps'),
  enabled: z.boolean(),
  record_enabled: z.boolean(),
  motion_detection: z.boolean(),
  audio_enabled: z.boolean(),
  night_vision: z.boolean(),
  ptz_enabled: z.boolean(),
  username: z.string().optional(),
  password: z.string().optional(),
  tags: z.string().optional()
})

type CameraConfigFormData = z.infer<typeof cameraConfigSchema>

interface CameraConfigModalProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  cameraId: string
  onSuccess?: (camera: Camera) => void
}

interface TestState {
  isConnecting: boolean
  connectionStatus: 'idle' | 'testing' | 'success' | 'failed'
  errorMessage: string | null
}

const INITIAL_TEST_STATE: TestState = {
  isConnecting: false,
  connectionStatus: 'idle',
  errorMessage: null
}

const RESOLUTION_PRESETS = [
  { label: '4K (3840x2160)', width: 3840, height: 2160 },
  { label: '1080p (1920x1080)', width: 1920, height: 1080 },
  { label: '720p (1280x720)', width: 1280, height: 720 },
  { label: '480p (640x480)', width: 640, height: 480 },
  { label: 'Custom', width: 0, height: 0 }
]

const QUALITY_OPTIONS = [
  { value: 'low', label: 'Low (Fast)', bitrate: 500 },
  { value: 'medium', label: 'Medium (Balanced)', bitrate: 1500 },
  { value: 'high', label: 'High (Quality)', bitrate: 3000 },
  { value: 'ultra', label: 'Ultra (Best)', bitrate: 8000 }
]

export default function CameraConfigModal({
  isOpen,
  onOpenChange,
  cameraId,
  onSuccess
}: CameraConfigModalProps) {
  const [isPending, startTransition] = useTransition()
  const [testState, setTestState] = useState<TestState>(INITIAL_TEST_STATE)
  const [selectedResolution, setSelectedResolution] = useState('1080p')
  const [activeTab, setActiveTab] = useState('basic')

  const queryClient = useQueryClient()
  const { toast } = useToast()

  // Fetch camera data
  const {
    data: camera,
    isLoading,
    error: fetchError
  } = useQuery({
    queryKey: ['camera', cameraId],
    queryFn: () => cameraUtils.getById(cameraId),
    enabled: isOpen && !!cameraId,
    staleTime: 0 // Always fetch fresh data for config modal
  })

  // Form setup
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting, isDirty },
    reset,
    watch,
    setValue,
    getValues,
    trigger
  } = useForm<CameraConfigFormData>({
    resolver: zodResolver(cameraConfigSchema),
    mode: 'onBlur'
  })

  // Watch form values for dynamic updates
  const qualityValue = watch('quality')
  const resolutionValue = watch('resolution')

  // Update bitrate when quality changes
  useEffect(() => {
    if (qualityValue) {
      const qualityOption = QUALITY_OPTIONS.find(opt => opt.value === qualityValue)
      if (qualityOption) {
        setValue('bitrate', qualityOption.bitrate, { shouldValidate: true })
      }
    }
  }, [qualityValue, setValue])

  // Reset form when camera data loads
  useEffect(() => {
    if (camera && isOpen) {
      reset({
        name: camera.name,
        location: camera.location,
        stream_url: camera.stream_url,
        resolution: camera.resolution,
        fps: camera.fps,
        quality: (camera as any).quality || 'medium',
        encoding: (camera as any).encoding || 'h264',
        bitrate: (camera as any).bitrate || 1500,
        enabled: (camera as any).enabled ?? true,
        record_enabled: (camera as any).record_enabled ?? false,
        motion_detection: (camera as any).motion_detection ?? true,
        audio_enabled: (camera as any).audio_enabled ?? false,
        night_vision: (camera as any).night_vision ?? false,
        ptz_enabled: (camera as any).ptz_enabled ?? false,
        username: (camera as any).username || '',
        password: (camera as any).password || '',
        tags: (camera as any).tags || ''
      })

      // Set resolution preset
      const preset = RESOLUTION_PRESETS.find(p =>
        p.width === camera.resolution.width && p.height === camera.resolution.height
      )
      setSelectedResolution(preset?.label.split(' ')[0] || 'Custom')
    }
  }, [camera, isOpen, reset])

  // Update configuration mutation
  const updateConfigMutation = useMutation({
    mutationFn: (data: CameraConfigFormData) =>
      cameraUtils.update(cameraId, data as Partial<Camera>),
    onSuccess: (updatedCamera) => {
      queryClient.invalidateQueries({ queryKey: ['camera', cameraId] })
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
      onSuccess?.(updatedCamera)
      onOpenChange(false)
      toast({
        title: "Configuration Saved",
        description: `Camera ${updatedCamera.name} has been updated successfully.`
      })
    },
    onError: (error) => {
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Failed to update camera configuration'
      toast({
        title: "Configuration Failed",
        description: errorMessage,
        variant: "destructive"
      })
    }
  })

  // Test connection mutation
  const testConnectionMutation = useMutation({
    mutationFn: async (streamUrl: string) => {
      // This would call a backend endpoint to test the camera connection
      const response = await fetch(`/api/cameras/test-connection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream_url: streamUrl })
      })

      if (!response.ok) {
        throw new Error('Connection test failed')
      }

      return response.json()
    },
    onMutate: () => {
      setTestState({
        isConnecting: true,
        connectionStatus: 'testing',
        errorMessage: null
      })
    },
    onSuccess: () => {
      setTestState({
        isConnecting: false,
        connectionStatus: 'success',
        errorMessage: null
      })
      toast({
        title: "Connection Successful",
        description: "Camera stream is accessible and working properly."
      })
    },
    onError: (error) => {
      const errorMessage = error instanceof Error ? error.message : 'Connection test failed'
      setTestState({
        isConnecting: false,
        connectionStatus: 'failed',
        errorMessage
      })
    }
  })

  // Handle resolution preset change
  const handleResolutionPresetChange = useCallback((preset: string) => {
    setSelectedResolution(preset)
    const resolution = RESOLUTION_PRESETS.find(p => p.label.startsWith(preset))
    if (resolution && resolution.width > 0) {
      setValue('resolution.width', resolution.width, { shouldValidate: true })
      setValue('resolution.height', resolution.height, { shouldValidate: true })
    }
  }, [setValue])

  // Handle test connection
  const handleTestConnection = useCallback(async () => {
    const streamUrl = getValues('stream_url')
    if (!streamUrl) {
      toast({
        title: "Missing Stream URL",
        description: "Please enter a stream URL to test the connection.",
        variant: "destructive"
      })
      return
    }

    // Validate stream URL first
    const isValid = await trigger('stream_url')
    if (!isValid) {
      toast({
        title: "Invalid Stream URL",
        description: "Please enter a valid stream URL.",
        variant: "destructive"
      })
      return
    }

    testConnectionMutation.mutate(streamUrl)
  }, [getValues, trigger, testConnectionMutation, toast])

  // Handle form submission
  const onSubmit = useCallback((data: CameraConfigFormData) => {
    startTransition(() => {
      updateConfigMutation.mutate(data)
    })
  }, [updateConfigMutation])

  // Handle modal close
  const handleClose = useCallback(() => {
    if (isDirty && !updateConfigMutation.isPending) {
      const confirmed = window.confirm(
        'You have unsaved changes. Are you sure you want to close without saving?'
      )
      if (!confirmed) return
    }
    onOpenChange(false)
  }, [isDirty, updateConfigMutation.isPending, onOpenChange])

  const isLoading = isLoading || updateConfigMutation.isPending || isPending

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <IconCamera className="w-5 h-5" />
            <span>Camera Configuration</span>
          </DialogTitle>
          <DialogDescription>
            Configure camera settings, stream parameters, and recording options.
          </DialogDescription>
        </DialogHeader>

        {fetchError && (
          <Alert variant="destructive">
            <IconAlertTriangle className="h-4 w-4" />
            <div>
              <h4 className="font-medium">Failed to load camera data</h4>
              <p className="text-sm mt-1">
                {fetchError instanceof APIError ? fetchError.message : 'Unknown error occurred'}
              </p>
            </div>
          </Alert>
        )}

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic" className="flex items-center space-x-1">
                <IconSettings className="w-4 h-4" />
                <span>Basic</span>
              </TabsTrigger>
              <TabsTrigger value="stream" className="flex items-center space-x-1">
                <IconAdjustments className="w-4 h-4" />
                <span>Stream</span>
              </TabsTrigger>
              <TabsTrigger value="features" className="flex items-center space-x-1">
                <IconShield className="w-4 h-4" />
                <span>Features</span>
              </TabsTrigger>
              <TabsTrigger value="auth" className="flex items-center space-x-1">
                <IconWifi className="w-4 h-4" />
                <span>Auth</span>
              </TabsTrigger>
            </TabsList>

            {/* Basic Configuration */}
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
              </div>

              <div className="space-y-2">
                <div className="flex items-center space-x-2">
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

              <div className="space-y-2">
                <Label htmlFor="tags">Tags (optional)</Label>
                <Input
                  id="tags"
                  placeholder="entrance, security, main-building"
                  disabled={isLoading}
                  {...register('tags')}
                />
                <p className="text-xs text-muted-foreground">
                  Comma-separated tags for organization and filtering
                </p>
              </div>
            </TabsContent>

            {/* Stream Configuration */}
            <TabsContent value="stream" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Resolution Preset</Label>
                  <Select value={selectedResolution} onValueChange={handleResolutionPresetChange}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {RESOLUTION_PRESETS.map((preset) => (
                        <SelectItem key={preset.label} value={preset.label.split(' ')[0]}>
                          {preset.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="width">Width</Label>
                  <Input
                    id="width"
                    type="number"
                    min={320}
                    max={7680}
                    disabled={isLoading || selectedResolution !== 'Custom'}
                    {...register('resolution.width', { valueAsNumber: true })}
                    className={errors.resolution?.width ? 'border-destructive' : ''}
                  />
                  {errors.resolution?.width && (
                    <p className="text-sm text-destructive">{errors.resolution.width.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="height">Height</Label>
                  <Input
                    id="height"
                    type="number"
                    min={240}
                    max={4320}
                    disabled={isLoading || selectedResolution !== 'Custom'}
                    {...register('resolution.height', { valueAsNumber: true })}
                    className={errors.resolution?.height ? 'border-destructive' : ''}
                  />
                  {errors.resolution?.height && (
                    <p className="text-sm text-destructive">{errors.resolution.height.message}</p>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="fps">Frame Rate (FPS)</Label>
                  <Input
                    id="fps"
                    type="number"
                    min={1}
                    max={60}
                    disabled={isLoading}
                    {...register('fps', { valueAsNumber: true })}
                    className={errors.fps ? 'border-destructive' : ''}
                  />
                  {errors.fps && (
                    <p className="text-sm text-destructive">{errors.fps.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="quality">Quality</Label>
                  <Select {...register('quality')}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {QUALITY_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="encoding">Encoding</Label>
                  <Select {...register('encoding')}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="h264">H.264 (Compatible)</SelectItem>
                      <SelectItem value="h265">H.265 (Efficient)</SelectItem>
                      <SelectItem value="mjpeg">MJPEG (Simple)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="bitrate">Bitrate (kbps)</Label>
                  <Input
                    id="bitrate"
                    type="number"
                    min={100}
                    max={50000}
                    disabled={isLoading}
                    {...register('bitrate', { valueAsNumber: true })}
                    className={errors.bitrate ? 'border-destructive' : ''}
                  />
                  {errors.bitrate && (
                    <p className="text-sm text-destructive">{errors.bitrate.message}</p>
                  )}
                </div>
              </div>
            </TabsContent>

            {/* Features Configuration */}
            <TabsContent value="features" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="enabled">Camera Enabled</Label>
                      <p className="text-xs text-muted-foreground">
                        Enable or disable this camera
                      </p>
                    </div>
                    <Switch id="enabled" {...register('enabled')} />
                  </div>

                  <Separator />

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="record_enabled">Recording</Label>
                      <p className="text-xs text-muted-foreground">
                        Enable continuous recording
                      </p>
                    </div>
                    <Switch id="record_enabled" {...register('record_enabled')} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="motion_detection">Motion Detection</Label>
                      <p className="text-xs text-muted-foreground">
                        Detect motion and trigger alerts
                      </p>
                    </div>
                    <Switch id="motion_detection" {...register('motion_detection')} />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="audio_enabled">Audio Recording</Label>
                      <p className="text-xs text-muted-foreground">
                        Record audio with video
                      </p>
                    </div>
                    <Switch id="audio_enabled" {...register('audio_enabled')} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="night_vision">Night Vision</Label>
                      <p className="text-xs text-muted-foreground">
                        Enhanced low-light recording
                      </p>
                    </div>
                    <Switch id="night_vision" {...register('night_vision')} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="ptz_enabled">PTZ Control</Label>
                      <p className="text-xs text-muted-foreground">
                        Pan, tilt, and zoom capabilities
                      </p>
                    </div>
                    <Switch id="ptz_enabled" {...register('ptz_enabled')} />
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Authentication Configuration */}
            <TabsContent value="auth" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="username">Username (optional)</Label>
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
                  <Label htmlFor="password">Password (optional)</Label>
                  <Input
                    id="password"
                    type="password"
                    placeholder="Camera password"
                    disabled={isLoading}
                    {...register('password')}
                  />
                  <p className="text-xs text-muted-foreground">
                    Leave empty to keep existing password
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

          <DialogFooter>
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
              disabled={isLoading || !isDirty}
            >
              {isLoading ? (
                <>
                  <IconLoader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <IconCheck className="w-4 h-4 mr-2" />
                  Save Configuration
                </>
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
