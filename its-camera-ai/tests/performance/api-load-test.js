// k6 Performance Test for ITS Camera AI API
// Tests API endpoints under load to ensure performance requirements
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');
export const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 5 },   // Ramp up to 5 users
    { duration: '1m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down
  ],
  thresholds: {
    // API should respond within 500ms for 95% of requests
    http_req_duration: ['p(95)<500'],
    // Error rate should be less than 1%
    errors: ['rate<0.01'],
    // Response time trend
    response_time: ['p(95)<500'],
  },
};

// Base URL configuration
const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';

// Test scenarios
export default function () {
  // Health check endpoint
  testHealthEndpoint();

  // API endpoints (if available)
  testAPIEndpoints();

  // Sleep between requests
  sleep(1);
}

function testHealthEndpoint() {
  const response = http.get(`${BASE_URL}/health`);

  const success = check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  });

  errorRate.add(!success);
  responseTime.add(response.timings.duration);
}

function testAPIEndpoints() {
  // Test API v1 health endpoint
  const apiResponse = http.get(`${BASE_URL}/api/v1/health`);

  const apiSuccess = check(apiResponse, {
    'API v1 health status is 200': (r) => r.status === 200,
    'API v1 response time < 200ms': (r) => r.timings.duration < 200,
  });

  errorRate.add(!apiSuccess);
  responseTime.add(apiResponse.timings.duration);

  // Test metrics endpoint (if available)
  const metricsResponse = http.get(`${BASE_URL}/metrics`, {
    headers: {
      'Accept': 'text/plain',
    },
  });

  const metricsSuccess = check(metricsResponse, {
    'metrics endpoint accessible': (r) => r.status === 200 || r.status === 404, // 404 is acceptable if not exposed
    'metrics response time < 300ms': (r) => r.timings.duration < 300,
  });

  if (metricsResponse.status !== 404) {
    errorRate.add(!metricsSuccess);
    responseTime.add(metricsResponse.timings.duration);
  }
}

// Setup function (runs once at the beginning)
export function setup() {
  console.log(`🚀 Starting performance tests against ${BASE_URL}`);

  // Verify the API is accessible before starting load test
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error(`API health check failed: ${healthCheck.status}`);
  }

  return { baseUrl: BASE_URL };
}

// Teardown function (runs once at the end)
export function teardown(data) {
  console.log('📊 Performance test completed');
  console.log(`Base URL: ${data.baseUrl}`);
}
