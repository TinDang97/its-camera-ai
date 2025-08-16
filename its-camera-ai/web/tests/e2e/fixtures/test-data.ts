export interface TestUser {
  username: string;
  email: string;
  password: string;
  role: 'admin' | 'operator' | 'viewer';
  mfaEnabled: boolean;
}

export interface TestCamera {
  id?: string;
  name: string;
  location: string;
  rtspUrl?: string;
  status?: 'online' | 'offline' | 'maintenance';
  latitude?: number;
  longitude?: number;
  model?: string;
  ipAddress: string;
  frameRate?: number;
  description?: string;
}

export interface TestIncident {
  id: string;
  title: string;
  type: 'accident' | 'congestion' | 'road_closure' | 'weather' | 'construction';
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  cameraId: string;
  location: string;
  description: string;
}

export const TEST_USERS = {
  admin: {
    username: 'admin_user',
    email: 'admin@test.com',
    password: 'AdminTest123!',
    role: 'admin' as const,
    mfaEnabled: false,
  },
  user: {
    username: 'operator_user',
    email: 'operator@test.com',
    password: 'OperatorTest123!',
    role: 'operator' as const,
    mfaEnabled: false,
  },
  viewer: {
    username: 'viewer_user',
    email: 'viewer@test.com',
    password: 'ViewerTest123!',
    role: 'viewer' as const,
    mfaEnabled: false,
  },
  mfaUser: {
    username: 'mfa_user',
    email: 'mfa@test.com',
    password: 'MfaTest123!',
    role: 'operator' as const,
    mfaEnabled: true,
  },
  disabled: {
    username: 'disabled_user',
    email: 'disabled@test.com',
    password: 'DisabledTest123!',
    role: 'viewer' as const,
    mfaEnabled: false,
  },
} as const;

export const TEST_CAMERAS = {
  onlineCamera: {
    id: 'CAM001',
    name: 'Main St & 1st Ave',
    location: 'Downtown Intersection',
    rtspUrl: 'rtsp://test.camera.com:554/stream1',
    status: 'online' as const,
    latitude: 40.7128,
    longitude: -74.0060,
    model: 'HIKVISION DS-2CD2083G2',
    ipAddress: '192.168.1.101',
    frameRate: 30,
  },
  offlineCamera: {
    id: 'CAM002',
    name: '2nd Ave & Oak Dr',
    location: 'Residential Area',
    rtspUrl: 'rtsp://test.camera.com:554/stream2',
    status: 'offline' as const,
    latitude: 40.7200,
    longitude: -74.0100,
    model: 'AXIS P3375-LV',
    ipAddress: '192.168.1.102',
    frameRate: 0,
  },
  maintenanceCamera: {
    id: 'CAM003',
    name: 'Broadway & 5th St',
    location: 'Shopping District',
    rtspUrl: 'rtsp://test.camera.com:554/stream3',
    status: 'maintenance' as const,
    latitude: 40.7150,
    longitude: -74.0050,
    model: 'DAHUA IPC-HFW4631H',
    ipAddress: '192.168.1.103',
    frameRate: 0,
  },
  newCamera: {
    name: 'Test Camera New',
    location: 'Test Location',
    ipAddress: '192.168.1.200',
    latitude: 40.7589,
    longitude: -73.9851,
    model: 'HIKVISION DS-2CD2185FWD-I',
    frameRate: 25,
    description: 'Test camera for E2E testing',
  },
  editableCamera: {
    name: 'Editable Test Camera',
    location: 'Editable Location',
    ipAddress: '192.168.1.201',
    latitude: 40.7505,
    longitude: -73.9934,
    model: 'AXIS M3075-V',
    frameRate: 30,
    description: 'Camera for testing edit functionality',
  },
  deletableCamera: {
    name: 'Deletable Test Camera',
    location: 'Deletable Location',
    ipAddress: '192.168.1.202',
    latitude: 40.7614,
    longitude: -73.9776,
    model: 'BOSCH FLEXIDOME IP starlight 6000',
    frameRate: 20,
    description: 'Camera for testing delete functionality',
  },
} as const;

export const TEST_INCIDENTS: TestIncident[] = [
  {
    id: 'INC001',
    title: 'Traffic Accident on Main St',
    type: 'accident',
    severity: 'high',
    status: 'open',
    cameraId: 'CAM001',
    location: 'Main St & 1st Ave',
    description: 'Multi-vehicle collision blocking two lanes',
  },
  {
    id: 'INC002',
    title: 'Road Construction Work',
    type: 'construction',
    severity: 'medium',
    status: 'in_progress',
    cameraId: 'CAM003',
    location: 'Broadway & 5th St',
    description: 'Scheduled maintenance work on traffic signals',
  },
];

export const WEBSOCKET_MESSAGES = {
  CAMERA_STATUS_UPDATE: {
    type: 'camera_status',
    data: {
      cameraId: 'CAM001',
      status: 'online',
      timestamp: new Date().toISOString(),
      frameRate: 30,
      uptime: 86400,
    },
  },
  TRAFFIC_UPDATE: {
    type: 'traffic_data',
    data: {
      cameraId: 'CAM001',
      vehicleCount: 45,
      averageSpeed: 35.5,
      congestionLevel: 'moderate',
      timestamp: new Date().toISOString(),
    },
  },
  NEW_INCIDENT: {
    type: 'incident_created',
    data: {
      id: 'INC003',
      title: 'Test Incident',
      type: 'accident',
      severity: 'critical',
      cameraId: 'CAM001',
      timestamp: new Date().toISOString(),
    },
  },
  SYSTEM_ALERT: {
    type: 'system_alert',
    data: {
      id: 'ALERT001',
      message: 'High CPU usage detected',
      severity: 'warning',
      source: 'monitoring',
      timestamp: new Date().toISOString(),
    },
  },
};

export const MOCK_API_RESPONSES = {
  CAMERAS_LIST: {
    cameras: [TEST_CAMERAS.onlineCamera, TEST_CAMERAS.offlineCamera, TEST_CAMERAS.maintenanceCamera],
    total: 3,
    page: 1,
    limit: 10,
  },
  INCIDENTS_LIST: {
    incidents: TEST_INCIDENTS,
    total: TEST_INCIDENTS.length,
    page: 1,
    limit: 10,
  },
  ANALYTICS_DATA: {
    metrics: {
      totalCameras: 3,
      onlineCameras: 1,
      offlineCameras: 1,
      maintenanceCameras: 1,
      totalIncidents: 2,
      activeIncidents: 1,
    },
    trafficData: Array.from({ length: 24 }, (_, i) => ({
      timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
      vehicleCount: Math.floor(Math.random() * 50) + 10,
      averageSpeed: Math.floor(Math.random() * 20) + 40,
      congestionLevel: ['low', 'moderate', 'high'][Math.floor(Math.random() * 3)],
    })),
  },
};