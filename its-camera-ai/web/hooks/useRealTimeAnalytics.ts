'use client';

import { useState, useEffect, useMemo, useDeferredValue } from 'react';

interface AnalyticsMetrics {
  totalVehicles: number;
  currentFlow: number;
  alertsToday: number;
  peakHour: {
    hour: number;
    count: number;
  };
  flowTrends: {
    increasing: boolean;
    percentage: number;
  };
  alertTrends: {
    increasing: boolean;
    percentage: number;
  };
}

interface ChartDataPoint {
  x: string;
  y: number;
}

interface ChartData {
  vehicleCount: ChartDataPoint[];
  trafficFlow: ChartDataPoint[];
  alerts: ChartDataPoint[];
}

interface UseRealTimeAnalyticsOptions {
  maxDataPoints?: number;
  updateInterval?: number;
}

// Mock data generator for real-time simulation
function generateMockDataPoint(timestamp: string, baseValue: number, variance: number = 0.3) {
  const variation = (Math.random() - 0.5) * 2 * variance;
  return Math.max(0, Math.floor(baseValue * (1 + variation)));
}

export function useRealTimeAnalytics(options: UseRealTimeAnalyticsOptions = {}) {
  const { maxDataPoints = 50, updateInterval = 5000 } = options;

  const [rawData, setRawData] = useState<ChartData>({
    vehicleCount: [],
    trafficFlow: [],
    alerts: [],
  });

  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Use deferred value for performance optimization
  const deferredData = useDeferredValue(rawData);

  // Generate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const timestamp = now.toISOString();

      // Generate realistic data based on time of day
      const hour = now.getHours();
      const isRushHour = (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19);
      const isNightTime = hour >= 22 || hour <= 6;

      let baseVehicleCount = 25;
      let baseTrafficFlow = 15;
      let baseAlerts = 2;

      if (isRushHour) {
        baseVehicleCount = 85;
        baseTrafficFlow = 60;
        baseAlerts = 8;
      } else if (isNightTime) {
        baseVehicleCount = 5;
        baseTrafficFlow = 3;
        baseAlerts = 1;
      }

      setRawData(prev => {
        const newVehiclePoint = {
          x: timestamp,
          y: generateMockDataPoint(timestamp, baseVehicleCount),
        };

        const newFlowPoint = {
          x: timestamp,
          y: generateMockDataPoint(timestamp, baseTrafficFlow),
        };

        const newAlertPoint = {
          x: timestamp,
          y: generateMockDataPoint(timestamp, baseAlerts, 0.8),
        };

        return {
          vehicleCount: [...prev.vehicleCount.slice(-maxDataPoints + 1), newVehiclePoint],
          trafficFlow: [...prev.trafficFlow.slice(-maxDataPoints + 1), newFlowPoint],
          alerts: [...prev.alerts.slice(-maxDataPoints + 1), newAlertPoint],
        };
      });

      setLastUpdate(now);
    }, updateInterval);

    // Initialize with some data
    const initData = () => {
      const now = Date.now();
      const initialPoints = 10;
      const newData: ChartData = {
        vehicleCount: [],
        trafficFlow: [],
        alerts: [],
      };

      for (let i = initialPoints - 1; i >= 0; i--) {
        const timestamp = new Date(now - i * updateInterval).toISOString();
        newData.vehicleCount.push({
          x: timestamp,
          y: generateMockDataPoint(timestamp, 25),
        });
        newData.trafficFlow.push({
          x: timestamp,
          y: generateMockDataPoint(timestamp, 15),
        });
        newData.alerts.push({
          x: timestamp,
          y: generateMockDataPoint(timestamp, 2, 0.8),
        });
      }

      setRawData(newData);
    };

    initData();

    return () => clearInterval(interval);
  }, [maxDataPoints, updateInterval]);

  // Calculate analytics metrics
  const analytics = useMemo((): AnalyticsMetrics => {
    if (deferredData.vehicleCount.length < 2) {
      return {
        totalVehicles: 0,
        currentFlow: 0,
        alertsToday: 0,
        peakHour: { hour: 8, count: 0 },
        flowTrends: { increasing: false, percentage: 0 },
        alertTrends: { increasing: false, percentage: 0 },
      };
    }

    const totalVehicles = deferredData.vehicleCount.reduce((sum, point) => sum + point.y, 0);
    const currentFlow = deferredData.trafficFlow[deferredData.trafficFlow.length - 1]?.y || 0;
    const alertsToday = deferredData.alerts.reduce((sum, point) => sum + point.y, 0);

    // Calculate peak hour
    const hourCounts: Record<number, number> = {};
    deferredData.vehicleCount.forEach(point => {
      const hour = new Date(point.x).getHours();
      hourCounts[hour] = (hourCounts[hour] || 0) + point.y;
    });

    const peakHour = Object.entries(hourCounts).reduce(
      (peak, [hour, count]) => (count > peak.count ? { hour: parseInt(hour), count } : peak),
      { hour: 8, count: 0 }
    );

    // Calculate trends (comparing last 10 points with previous 10)
    const recentFlowPoints = deferredData.trafficFlow.slice(-10);
    const previousFlowPoints = deferredData.trafficFlow.slice(-20, -10);

    const recentFlowAvg = recentFlowPoints.reduce((sum, p) => sum + p.y, 0) / recentFlowPoints.length;
    const previousFlowAvg = previousFlowPoints.reduce((sum, p) => sum + p.y, 0) / previousFlowPoints.length || 1;

    const flowTrends = {
      increasing: recentFlowAvg > previousFlowAvg,
      percentage: Math.abs(((recentFlowAvg - previousFlowAvg) / previousFlowAvg) * 100),
    };

    // Similar calculation for alerts
    const recentAlertPoints = deferredData.alerts.slice(-10);
    const previousAlertPoints = deferredData.alerts.slice(-20, -10);

    const recentAlertAvg = recentAlertPoints.reduce((sum, p) => sum + p.y, 0) / recentAlertPoints.length;
    const previousAlertAvg = previousAlertPoints.reduce((sum, p) => sum + p.y, 0) / previousAlertPoints.length || 1;

    const alertTrends = {
      increasing: recentAlertAvg > previousAlertAvg,
      percentage: Math.abs(((recentAlertAvg - previousAlertAvg) / previousAlertAvg) * 100),
    };

    return {
      totalVehicles,
      currentFlow,
      alertsToday,
      peakHour,
      flowTrends,
      alertTrends,
    };
  }, [deferredData]);

  return {
    analytics,
    chartData: deferredData,
    lastUpdate,
  };
}
