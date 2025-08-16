#!/usr/bin/env python3
"""
Performance Report Generator

Generates comprehensive performance reports from test results and system metrics.
Used by the CI/CD pipeline to create performance summaries.
"""

import argparse
import json
import os
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2


class PerformanceReportGenerator:
    """Generates HTML performance reports from test data."""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.system_metrics: Dict[str, Any] = {}
    
    def load_junit_results(self, junit_files: List[str]):
        """Load test results from JUnit XML files."""
        for file_path in junit_files:
            if not os.path.exists(file_path):
                continue
            
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                test_suite = {
                    "name": root.get("name", os.path.basename(file_path)),
                    "tests": int(root.get("tests", 0)),
                    "failures": int(root.get("failures", 0)),
                    "errors": int(root.get("errors", 0)),
                    "skipped": int(root.get("skipped", 0)),
                    "time": float(root.get("time", 0)),
                    "test_cases": []
                }
                
                for testcase in root.findall(".//testcase"):
                    test_case = {
                        "name": testcase.get("name"),
                        "classname": testcase.get("classname"),
                        "time": float(testcase.get("time", 0)),
                        "status": "passed"
                    }
                    
                    if testcase.find("failure") is not None:
                        test_case["status"] = "failed"
                        test_case["failure"] = testcase.find("failure").text
                    elif testcase.find("error") is not None:
                        test_case["status"] = "error"
                        test_case["error"] = testcase.find("error").text
                    elif testcase.find("skipped") is not None:
                        test_case["status"] = "skipped"
                    
                    test_suite["test_cases"].append(test_case)
                
                self.test_results.append(test_suite)
                
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
    
    def extract_performance_metrics(self):
        """Extract performance metrics from test results."""
        response_times = []
        throughput_values = []
        error_rates = []
        
        for suite in self.test_results:
            if "performance" in suite["name"].lower() or "load" in suite["name"].lower():
                for test_case in suite["test_cases"]:
                    # Extract metrics from test names and times
                    if "response_time" in test_case["name"]:
                        response_times.append(test_case["time"])
                    elif "throughput" in test_case["name"]:
                        throughput_values.append(1.0 / test_case["time"] if test_case["time"] > 0 else 0)
                    
                    if test_case["status"] != "passed":
                        error_rates.append(1)
                    else:
                        error_rates.append(0)
        
        self.performance_metrics = {
            "response_times": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "p95": self._percentile(response_times, 95) if response_times else 0,
                "p99": self._percentile(response_times, 99) if response_times else 0,
            },
            "throughput": {
                "min": min(throughput_values) if throughput_values else 0,
                "max": max(throughput_values) if throughput_values else 0,
                "mean": statistics.mean(throughput_values) if throughput_values else 0,
            },
            "error_rate": statistics.mean(error_rates) if error_rates else 0,
            "total_tests": sum(suite["tests"] for suite in self.test_results),
            "total_failures": sum(suite["failures"] + suite["errors"] for suite in self.test_results),
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            import psutil
            
            self.system_metrics = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
                "disk_usage": psutil.disk_usage('/').percent,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Try to get GPU info if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.system_metrics["gpu_info"] = [
                        {
                            "name": gpu.name,
                            "memory_total": gpu.memoryTotal,
                            "memory_used": gpu.memoryUsed,
                            "temperature": gpu.temperature,
                        }
                        for gpu in gpus
                    ]
            except ImportError:
                pass
                
        except ImportError:
            self.system_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "note": "psutil not available - limited system metrics"
            }
    
    def generate_html_report(self, output_path: str):
        """Generate HTML performance report."""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ITS Camera AI Performance Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .test-results {
            margin: 30px 0;
        }
        .test-suite {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            margin: 10px 0;
            overflow: hidden;
        }
        .test-suite-header {
            background-color: #e9ecef;
            padding: 15px;
            font-weight: bold;
            border-bottom: 1px solid #dee2e6;
        }
        .test-case {
            padding: 10px 15px;
            border-bottom: 1px solid #f1f3f4;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-passed { color: #28a745; }
        .status-failed { color: #dc3545; }
        .status-skipped { color: #ffc107; }
        .status-error { color: #fd7e14; }
        .performance-chart {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .system-info {
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .summary-stats {
            display: flex;
            justify-content: space-around;
            background-color: #f1f8e9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2e7d2e;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ITS Camera AI Performance Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-value">{{ total_tests }}</div>
                <div>Total Tests</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ success_rate }}%</div>
                <div>Success Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ total_time }}s</div>
                <div>Total Time</div>
            </div>
        </div>

        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Mean Response Time</div>
                <div class="metric-value">{{ "%.0f"|format(performance_metrics.response_times.mean * 1000) }}ms</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">95th Percentile</div>
                <div class="metric-value">{{ "%.0f"|format(performance_metrics.response_times.p95 * 1000) }}ms</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Throughput</div>
                <div class="metric-value">{{ "%.1f"|format(performance_metrics.throughput.mean) }} RPS</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Error Rate</div>
                <div class="metric-value">{{ "%.2f"|format(performance_metrics.error_rate * 100) }}%</div>
            </div>
        </div>

        <h2>Response Time Statistics</h2>
        <div class="performance-chart">
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value (ms)</th>
                </tr>
                <tr>
                    <td>Minimum</td>
                    <td>{{ "%.1f"|format(performance_metrics.response_times.min * 1000) }}</td>
                </tr>
                <tr>
                    <td>Maximum</td>
                    <td>{{ "%.1f"|format(performance_metrics.response_times.max * 1000) }}</td>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{{ "%.1f"|format(performance_metrics.response_times.mean * 1000) }}</td>
                </tr>
                <tr>
                    <td>Median</td>
                    <td>{{ "%.1f"|format(performance_metrics.response_times.median * 1000) }}</td>
                </tr>
                <tr>
                    <td>95th Percentile</td>
                    <td>{{ "%.1f"|format(performance_metrics.response_times.p95 * 1000) }}</td>
                </tr>
                <tr>
                    <td>99th Percentile</td>
                    <td>{{ "%.1f"|format(performance_metrics.response_times.p99 * 1000) }}</td>
                </tr>
            </table>
        </div>

        <h2>Test Results Summary</h2>
        <div class="test-results">
            {% for suite in test_results %}
            <div class="test-suite">
                <div class="test-suite-header">
                    {{ suite.name }} - {{ suite.tests }} tests ({{ "%.2f"|format(suite.time) }}s)
                </div>
                {% for test_case in suite.test_cases %}
                <div class="test-case">
                    <span>{{ test_case.name }}</span>
                    <span class="status-{{ test_case.status }}">
                        {{ test_case.status.upper() }} ({{ "%.3f"|format(test_case.time) }}s)
                    </span>
                </div>
                {% endfor %}
            </div>
            {% endfor %}
        </div>

        <h2>System Information</h2>
        <div class="system-info">
            {% if system_metrics.cpu_count %}
            <p><strong>CPU Cores:</strong> {{ system_metrics.cpu_count }}</p>
            <p><strong>Memory:</strong> {{ "%.1f"|format(system_metrics.memory_total) }} GB</p>
            <p><strong>Disk Usage:</strong> {{ "%.1f"|format(system_metrics.disk_usage) }}%</p>
            {% endif %}
            
            {% if system_metrics.gpu_info %}
            <h3>GPU Information</h3>
            {% for gpu in system_metrics.gpu_info %}
            <p><strong>{{ gpu.name }}:</strong> {{ gpu.memory_used }}/{{ gpu.memory_total }} MB used ({{ gpu.temperature }}°C)</p>
            {% endfor %}
            {% endif %}
            
            <p><strong>Report Generated:</strong> {{ system_metrics.timestamp }}</p>
        </div>

        <h2>Performance Requirements Validation</h2>
        <div class="performance-chart">
            <table>
                <tr>
                    <th>Requirement</th>
                    <th>Target</th>
                    <th>Actual</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>API Response Time (P95)</td>
                    <td>&lt; 100ms</td>
                    <td>{{ "%.0f"|format(performance_metrics.response_times.p95 * 1000) }}ms</td>
                    <td class="{% if performance_metrics.response_times.p95 < 0.1 %}status-passed{% else %}status-failed{% endif %}">
                        {% if performance_metrics.response_times.p95 < 0.1 %}✓ PASS{% else %}✗ FAIL{% endif %}
                    </td>
                </tr>
                <tr>
                    <td>Error Rate</td>
                    <td>&lt; 1%</td>
                    <td>{{ "%.2f"|format(performance_metrics.error_rate * 100) }}%</td>
                    <td class="{% if performance_metrics.error_rate < 0.01 %}status-passed{% else %}status-failed{% endif %}">
                        {% if performance_metrics.error_rate < 0.01 %}✓ PASS{% else %}✗ FAIL{% endif %}
                    </td>
                </tr>
                <tr>
                    <td>Minimum Throughput</td>
                    <td>&gt; 100 RPS</td>
                    <td>{{ "%.1f"|format(performance_metrics.throughput.mean) }} RPS</td>
                    <td class="{% if performance_metrics.throughput.mean > 100 %}status-passed{% else %}status-failed{% endif %}">
                        {% if performance_metrics.throughput.mean > 100 %}✓ PASS{% else %}✗ FAIL{% endif %}
                    </td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
        """
        
        # Calculate summary statistics
        total_tests = sum(suite["tests"] for suite in self.test_results)
        total_failures = sum(suite["failures"] + suite["errors"] for suite in self.test_results)
        success_rate = ((total_tests - total_failures) / total_tests * 100) if total_tests > 0 else 0
        total_time = sum(suite["time"] for suite in self.test_results)
        
        # Render template
        template = jinja2.Template(template_content)
        html_content = template.render(
            test_results=self.test_results,
            performance_metrics=self.performance_metrics,
            system_metrics=self.system_metrics,
            total_tests=total_tests,
            success_rate=f"{success_rate:.1f}",
            total_time=f"{total_time:.1f}",
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Performance report generated: {output_path}")
    
    def generate_json_summary(self, output_path: str):
        """Generate JSON summary for programmatic access."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_summary": {
                "total_tests": sum(suite["tests"] for suite in self.test_results),
                "total_failures": sum(suite["failures"] + suite["errors"] for suite in self.test_results),
                "total_time": sum(suite["time"] for suite in self.test_results),
                "test_suites": len(self.test_results),
            },
            "performance_metrics": self.performance_metrics,
            "system_metrics": self.system_metrics,
            "requirements_validation": {
                "api_response_time_p95_pass": self.performance_metrics.get("response_times", {}).get("p95", 1) < 0.1,
                "error_rate_pass": self.performance_metrics.get("error_rate", 1) < 0.01,
                "throughput_pass": self.performance_metrics.get("throughput", {}).get("mean", 0) > 100,
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Performance summary generated: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate performance reports from test results")
    parser.add_argument("--output", required=True, help="Output HTML file path")
    parser.add_argument("--json-output", help="Output JSON summary file path")
    parser.add_argument("--junit-files", nargs="*", help="JUnit XML files to process")
    parser.add_argument("--junit-dir", help="Directory containing JUnit XML files")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PerformanceReportGenerator()
    
    # Collect JUnit files
    junit_files = args.junit_files or []
    
    if args.junit_dir and os.path.exists(args.junit_dir):
        junit_dir = Path(args.junit_dir)
        junit_files.extend([str(f) for f in junit_dir.glob("junit-*.xml")])
    
    # Look for JUnit files in current directory if none specified
    if not junit_files:
        junit_files = [str(f) for f in Path(".").glob("junit-*.xml")]
    
    print(f"Processing {len(junit_files)} JUnit files...")
    
    # Load test results
    generator.load_junit_results(junit_files)
    
    # Extract performance metrics
    generator.extract_performance_metrics()
    
    # Collect system metrics
    generator.collect_system_metrics()
    
    # Generate reports
    generator.generate_html_report(args.output)
    
    if args.json_output:
        generator.generate_json_summary(args.json_output)
    
    print("Performance report generation completed!")


if __name__ == "__main__":
    main()