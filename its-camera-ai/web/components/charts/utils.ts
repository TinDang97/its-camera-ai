/**
 * Chart Utilities
 *
 * Collection of utility functions for chart data processing, formatting,
 * and performance optimization in real-time analytics applications.
 */

import { ChartData } from './useD3Chart';
import { BarChartData } from './RealTimeBarChart';

// Data processing and optimization utilities
export class ChartDataProcessor {
  /**
   * Decimates data to reduce the number of points for performance
   * Uses advanced algorithms to preserve important features
   */
  static decimate(data: ChartData[], maxPoints: number, algorithm: 'uniform' | 'peak' | 'adaptive' = 'adaptive'): ChartData[] {
    if (data.length <= maxPoints) return data;

    switch (algorithm) {
      case 'uniform':
        return this.uniformDecimation(data, maxPoints);
      case 'peak':
        return this.peakPreservingDecimation(data, maxPoints);
      case 'adaptive':
        return this.adaptiveDecimation(data, maxPoints);
      default:
        return this.uniformDecimation(data, maxPoints);
    }
  }

  /**
   * Uniform decimation - evenly spaced sampling
   */
  private static uniformDecimation(data: ChartData[], maxPoints: number): ChartData[] {
    const step = Math.ceil(data.length / maxPoints);
    const decimated: ChartData[] = [];

    for (let i = 0; i < data.length; i += step) {
      decimated.push(data[i]);
    }

    // Always include the last point
    if (decimated[decimated.length - 1] !== data[data.length - 1]) {
      decimated.push(data[data.length - 1]);
    }

    return decimated;
  }

  /**
   * Peak-preserving decimation - preserves local maxima and minima
   */
  private static peakPreservingDecimation(data: ChartData[], maxPoints: number): ChartData[] {
    if (data.length <= 3) return data;

    const decimated: ChartData[] = [data[0]]; // Always include first point
    const step = Math.ceil(data.length / maxPoints);

    for (let i = step; i < data.length - step; i += step) {
      const window = data.slice(Math.max(0, i - step), Math.min(data.length, i + step));

      // Find local maximum and minimum in the window
      const max = window.reduce((prev, curr) => prev.value > curr.value ? prev : curr);
      const min = window.reduce((prev, curr) => prev.value < curr.value ? prev : curr);

      // Add both if they're different points
      if (max !== min) {
        if (max.timestamp < min.timestamp) {
          decimated.push(max, min);
        } else {
          decimated.push(min, max);
        }
      } else {
        decimated.push(max);
      }
    }

    // Always include the last point
    decimated.push(data[data.length - 1]);

    // Sort by timestamp and remove duplicates
    return [...new Map(decimated.map(item => [item.timestamp, item])).values()]
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(0, maxPoints);
  }

  /**
   * Adaptive decimation - intelligently preserves important data points
   */
  private static adaptiveDecimation(data: ChartData[], maxPoints: number): ChartData[] {
    if (data.length <= 3) return data;

    const decimated: ChartData[] = [data[0]];
    let lastIncluded = 0;

    // Calculate variance threshold
    const values = data.map(d => d.value);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const threshold = Math.sqrt(variance) * 0.1; // 10% of standard deviation

    for (let i = 1; i < data.length - 1; i++) {
      const current = data[i];
      const last = data[lastIncluded];

      // Include point if it represents a significant change
      const valueDiff = Math.abs(current.value - last.value);
      const timeDiff = current.timestamp - last.timestamp;
      const rate = valueDiff / (timeDiff || 1);

      if (valueDiff > threshold || rate > threshold || decimated.length >= maxPoints - 1) {
        decimated.push(current);
        lastIncluded = i;

        if (decimated.length >= maxPoints - 1) break;
      }
    }

    // Always include the last point
    decimated.push(data[data.length - 1]);

    return decimated;
  }

  /**
   * Smooths data using moving average
   */
  static smooth(data: ChartData[], windowSize: number = 5): ChartData[] {
    if (data.length < windowSize) return data;

    const smoothed: ChartData[] = [];

    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(data.length, i + Math.ceil(windowSize / 2));
      const window = data.slice(start, end);

      const avgValue = window.reduce((sum, item) => sum + item.value, 0) / window.length;

      smoothed.push({
        ...data[i],
        value: avgValue,
        metadata: {
          ...data[i].metadata,
          originalValue: data[i].value,
          smoothingWindow: windowSize,
        },
      });
    }

    return smoothed;
  }

  /**
   * Detects anomalies in the data
   */
  static detectAnomalies(
    data: ChartData[],
    threshold: number = 3,
    method: 'zscore' | 'iqr' = 'zscore'
  ): ChartData[] {
    if (data.length < 3) return [];

    const values = data.map(d => d.value);

    if (method === 'zscore') {
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      const stdDev = Math.sqrt(
        values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
      );

      return data.filter(item => {
        const zScore = Math.abs((item.value - mean) / stdDev);
        return zScore > threshold;
      });
    } else {
      // IQR method
      const sorted = [...values].sort((a, b) => a - b);
      const q1Index = Math.floor(sorted.length * 0.25);
      const q3Index = Math.floor(sorted.length * 0.75);
      const q1 = sorted[q1Index];
      const q3 = sorted[q3Index];
      const iqr = q3 - q1;
      const lowerBound = q1 - threshold * iqr;
      const upperBound = q3 + threshold * iqr;

      return data.filter(item =>
        item.value < lowerBound || item.value > upperBound
      );
    }
  }

  /**
   * Aggregates data into time buckets
   */
  static aggregate(
    data: ChartData[],
    bucketSize: number, // milliseconds
    method: 'avg' | 'sum' | 'max' | 'min' | 'count' = 'avg'
  ): ChartData[] {
    if (data.length === 0) return [];

    const buckets = new Map<number, ChartData[]>();

    // Group data into buckets
    data.forEach(item => {
      const bucketKey = Math.floor(item.timestamp / bucketSize) * bucketSize;
      if (!buckets.has(bucketKey)) {
        buckets.set(bucketKey, []);
      }
      buckets.get(bucketKey)!.push(item);
    });

    // Aggregate each bucket
    const aggregated: ChartData[] = [];

    buckets.forEach((bucketData, timestamp) => {
      let value: number;

      switch (method) {
        case 'avg':
          value = bucketData.reduce((sum, item) => sum + item.value, 0) / bucketData.length;
          break;
        case 'sum':
          value = bucketData.reduce((sum, item) => sum + item.value, 0);
          break;
        case 'max':
          value = Math.max(...bucketData.map(item => item.value));
          break;
        case 'min':
          value = Math.min(...bucketData.map(item => item.value));
          break;
        case 'count':
          value = bucketData.length;
          break;
        default:
          value = bucketData.reduce((sum, item) => sum + item.value, 0) / bucketData.length;
      }

      aggregated.push({
        timestamp,
        value,
        metadata: {
          aggregationMethod: method,
          bucketSize,
          originalCount: bucketData.length,
          originalData: bucketData,
        },
      });
    });

    return aggregated.sort((a, b) => a.timestamp - b.timestamp);
  }
}

// Formatting utilities
export class ChartFormatters {
  /**
   * Format large numbers with appropriate units
   */
  static formatLargeNumber(value: number, precision: number = 1): string {
    const units = ['', 'K', 'M', 'B', 'T'];
    let unitIndex = 0;
    let scaledValue = value;

    while (scaledValue >= 1000 && unitIndex < units.length - 1) {
      scaledValue /= 1000;
      unitIndex++;
    }

    return `${scaledValue.toFixed(precision)}${units[unitIndex]}`;
  }

  /**
   * Format time duration
   */
  static formatDuration(milliseconds: number): string {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  }

  /**
   * Format percentage with appropriate precision
   */
  static formatPercentage(value: number, precision: number = 1): string {
    return `${(value * 100).toFixed(precision)}%`;
  }

  /**
   * Format timestamp with intelligent granularity
   */
  static formatTimestamp(timestamp: number, granularity?: 'second' | 'minute' | 'hour' | 'day'): string {
    const date = new Date(timestamp);

    switch (granularity) {
      case 'second':
        return date.toLocaleTimeString();
      case 'minute':
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      case 'hour':
        return date.toLocaleTimeString([], { hour: '2-digit' });
      case 'day':
        return date.toLocaleDateString();
      default:
        // Auto-detect appropriate granularity
        const now = Date.now();
        const diff = now - timestamp;

        if (diff < 60000) return date.toLocaleTimeString(); // < 1 minute
        if (diff < 3600000) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); // < 1 hour
        if (diff < 86400000) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); // < 1 day
        return date.toLocaleDateString();
    }
  }
}

// Performance utilities
export class ChartPerformance {
  private static updateQueue = new Map<string, { data: any; timestamp: number }>();
  private static updateInterval: NodeJS.Timeout | null = null;

  /**
   * Batches chart updates to improve performance
   */
  static batchUpdate(chartId: string, data: any, callback: (data: any) => void, batchDelay: number = 16): void {
    // Add to update queue
    this.updateQueue.set(chartId, { data, timestamp: Date.now() });

    // Start batch processing if not already running
    if (!this.updateInterval) {
      this.updateInterval = setInterval(() => {
        const updates = Array.from(this.updateQueue.entries());
        this.updateQueue.clear();

        // Process all updates
        updates.forEach(([id, { data }]) => {
          callback(data);
        });

        // Stop interval if queue is empty
        if (this.updateQueue.size === 0) {
          clearInterval(this.updateInterval!);
          this.updateInterval = null;
        }
      }, batchDelay);
    }
  }

  /**
   * Throttles function calls to improve performance
   */
  static throttle<T extends (...args: any[]) => any>(
    func: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout | null = null;
    let lastExecTime = 0;

    return (...args: Parameters<T>) => {
      const currentTime = Date.now();

      if (currentTime - lastExecTime > delay) {
        func(...args);
        lastExecTime = currentTime;
      } else {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          func(...args);
          lastExecTime = Date.now();
        }, delay - (currentTime - lastExecTime));
      }
    };
  }

  /**
   * Debounces function calls
   */
  static debounce<T extends (...args: any[]) => any>(
    func: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout | null = null;

    return (...args: Parameters<T>) => {
      if (timeoutId) clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func(...args), delay);
    };
  }
}

// Color utilities
export class ChartColors {
  private static readonly COLOR_PALETTES = {
    primary: ['hsl(var(--primary))', 'hsl(var(--primary-light))', 'hsl(var(--primary-dark))'],
    traffic: ['hsl(var(--success))', 'hsl(var(--warning))', 'hsl(var(--destructive))'],
    status: ['hsl(var(--online))', 'hsl(var(--offline))', 'hsl(var(--maintenance))'],
    gradient: ['hsl(210, 100%, 56%)', 'hsl(210, 100%, 76%)', 'hsl(210, 100%, 96%)'],
  };

  /**
   * Gets a color palette for charts
   */
  static getPalette(name: keyof typeof ChartColors.COLOR_PALETTES): string[] {
    return this.COLOR_PALETTES[name];
  }

  /**
   * Generates a color for a specific category
   */
  static getCategoryColor(category: string, palette: string[] = this.COLOR_PALETTES.primary): string {
    const hash = category.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return palette[hash % palette.length];
  }

  /**
   * Interpolates between two colors
   */
  static interpolateColors(color1: string, color2: string, factor: number): string {
    // Simple linear interpolation for HSL colors
    // This is a simplified version - in a real implementation, you'd parse HSL values
    return factor < 0.5 ? color1 : color2;
  }
}

export default {
  ChartDataProcessor,
  ChartFormatters,
  ChartPerformance,
  ChartColors,
};
