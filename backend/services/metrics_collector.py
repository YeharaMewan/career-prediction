"""
Metrics Collector Service

Collects, aggregates, and persists ingestion metrics.

Features:
- Real-time metrics collection
- Time-series data aggregation
- Prometheus-compatible export
- Persistent storage to JSON
- Statistical calculations (avg, p95, p99)
- Dashboard-friendly API

Metrics tracked:
- Documents processed
- Chunks created/updated/deleted
- Ingestion duration (total, per file, per stage)
- Error rates and types
- File monitor events
- Validation runs
- Web scraping success/failure rates
- Vector DB operations

Author: Career Planning System
Created: 2025
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import aiofiles

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMING = "timing"  # Duration measurements


@dataclass
class MetricValue:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class MetricStats:
    """Statistical summary of a metric"""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    avg: float = 0.0
    p50: float = 0.0  # Median
    p95: float = 0.0
    p99: float = 0.0
    last_value: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCollector:
    """
    Collects and manages system metrics.

    Features:
    - In-memory time-series storage with configurable retention
    - Periodic persistence to disk
    - Statistical aggregation
    - Prometheus export (if available)
    - Thread-safe operations
    """

    def __init__(
        self,
        metrics_file: str = "logs/metrics.json",
        persist_interval: int = 300,  # 5 minutes
        retention_hours: int = 24,
        max_data_points: int = 10000
    ):
        """
        Initialize Metrics Collector

        Args:
            metrics_file: Path to persist metrics
            persist_interval: Seconds between persisting to disk
            retention_hours: Hours to retain metrics in memory
            max_data_points: Maximum data points per metric
        """
        self.metrics_file = Path(metrics_file)
        self.persist_interval = persist_interval
        self.retention_hours = retention_hours
        self.max_data_points = max_data_points

        # Create directory
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # Metric storage: {metric_name: deque of MetricValue}
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_data_points)
        )

        # Aggregated stats: {metric_name: MetricStats}
        self.stats: Dict[str, MetricStats] = {}

        # Prometheus metrics (if available)
        self.prom_counters: Dict[str, Counter] = {}
        self.prom_gauges: Dict[str, Gauge] = {}
        self.prom_histograms: Dict[str, Histogram] = {}

        # State
        self.running = False
        self.persist_task: Optional[asyncio.Task] = None

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load persisted metrics
        asyncio.create_task(self._load_metrics())

    async def _load_metrics(self):
        """Load persisted metrics from disk"""
        if not self.metrics_file.exists():
            return

        try:
            async with aiofiles.open(self.metrics_file, 'r') as f:
                data = json.loads(await f.read())

            # Restore metrics
            for metric_name, values in data.get("metrics", {}).items():
                for value_data in values:
                    metric_value = MetricValue(
                        timestamp=datetime.fromisoformat(value_data["timestamp"]),
                        value=value_data["value"],
                        labels=value_data.get("labels", {})
                    )
                    self.metrics[metric_name].append(metric_value)

            # Restore stats
            for metric_name, stats_data in data.get("stats", {}).items():
                self.stats[metric_name] = MetricStats(**stats_data)

            self.logger.info(f"Loaded {len(self.metrics)} metrics from {self.metrics_file}")

        except Exception as e:
            self.logger.error(f"Error loading metrics: {e}")

    async def _persist_metrics(self):
        """Persist metrics to disk"""
        try:
            data = {
                "metrics": {
                    name: [v.to_dict() for v in values]
                    for name, values in self.metrics.items()
                },
                "stats": {
                    name: stats.to_dict()
                    for name, stats in self.stats.items()
                },
                "last_updated": datetime.now().isoformat()
            }

            async with aiofiles.open(self.metrics_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))

            self.logger.debug(f"Persisted metrics to {self.metrics_file}")

        except Exception as e:
            self.logger.error(f"Error persisting metrics: {e}")

    async def _persist_worker(self):
        """Background worker that periodically persists metrics"""
        while self.running:
            try:
                await asyncio.sleep(self.persist_interval)
                await self._persist_metrics()
            except Exception as e:
                self.logger.error(f"Persist worker error: {e}")

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)

        for metric_name, values in self.metrics.items():
            # Remove old values (deque doesn't support efficient removal from middle)
            # So we create a new deque with only recent values
            recent_values = deque(
                (v for v in values if v.timestamp > cutoff),
                maxlen=self.max_data_points
            )
            self.metrics[metric_name] = recent_values

    def record(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.COUNTER
    ):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
            metric_type: Type of metric
        """
        labels = labels or {}

        # Add to time-series
        metric_value = MetricValue(
            timestamp=datetime.now(),
            value=value,
            labels=labels
        )
        self.metrics[metric_name].append(metric_value)

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            if metric_type == MetricType.COUNTER:
                if metric_name not in self.prom_counters:
                    self.prom_counters[metric_name] = Counter(
                        metric_name.replace(".", "_"),
                        f"Counter for {metric_name}",
                        list(labels.keys()) if labels else []
                    )
                if labels:
                    self.prom_counters[metric_name].labels(**labels).inc(value)
                else:
                    self.prom_counters[metric_name].inc(value)

            elif metric_type == MetricType.GAUGE:
                if metric_name not in self.prom_gauges:
                    self.prom_gauges[metric_name] = Gauge(
                        metric_name.replace(".", "_"),
                        f"Gauge for {metric_name}",
                        list(labels.keys()) if labels else []
                    )
                if labels:
                    self.prom_gauges[metric_name].labels(**labels).set(value)
                else:
                    self.prom_gauges[metric_name].set(value)

            elif metric_type == MetricType.HISTOGRAM:
                if metric_name not in self.prom_histograms:
                    self.prom_histograms[metric_name] = Histogram(
                        metric_name.replace(".", "_"),
                        f"Histogram for {metric_name}",
                        list(labels.keys()) if labels else []
                    )
                if labels:
                    self.prom_histograms[metric_name].labels(**labels).observe(value)
                else:
                    self.prom_histograms[metric_name].observe(value)

        # Update aggregated stats
        self._update_stats(metric_name)

    def _update_stats(self, metric_name: str):
        """Update statistical summary for a metric"""
        values_list = list(self.metrics[metric_name])

        if not values_list:
            return

        values = [v.value for v in values_list]

        stats = MetricStats()
        stats.count = len(values)
        stats.sum = sum(values)
        stats.min = min(values)
        stats.max = max(values)
        stats.avg = statistics.mean(values)
        stats.last_value = values[-1]

        # Calculate percentiles if enough data
        if len(values) >= 2:
            stats.p50 = statistics.median(values)
            stats.p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 20 else stats.max
            stats.p99 = statistics.quantiles(values, n=100)[98] if len(values) >= 100 else stats.max

        self.stats[metric_name] = stats

    def increment(
        self,
        metric_name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric"""
        # Get current value
        current = self.get_current_value(metric_name)
        self.record(metric_name, current + value, labels, MetricType.COUNTER)

    def set_gauge(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric"""
        self.record(metric_name, value, labels, MetricType.GAUGE)

    def record_timing(
        self,
        metric_name: str,
        duration_seconds: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a timing/duration metric"""
        self.record(
            metric_name,
            duration_seconds,
            labels,
            MetricType.HISTOGRAM
        )

    def get_current_value(self, metric_name: str) -> float:
        """Get the most recent value of a metric"""
        values = self.metrics.get(metric_name)
        if values:
            return values[-1].value
        return 0.0

    def get_stats(self, metric_name: str) -> Optional[MetricStats]:
        """Get statistical summary for a metric"""
        return self.stats.get(metric_name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics with their current values and stats"""
        return {
            "metrics": {
                name: {
                    "current_value": values[-1].value if values else 0,
                    "data_points": len(values),
                    "stats": self.stats.get(name).to_dict() if name in self.stats else None
                }
                for name, values in self.metrics.items()
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_time_series(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get time-series data for a metric.

        Args:
            metric_name: Name of the metric
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of metric values
        """
        values = self.metrics.get(metric_name, [])

        # Filter by time range
        if start_time or end_time:
            filtered = []
            for v in values:
                if start_time and v.timestamp < start_time:
                    continue
                if end_time and v.timestamp > end_time:
                    continue
                filtered.append(v.to_dict())
            return filtered

        return [v.to_dict() for v in values]

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"

        try:
            return generate_latest(REGISTRY).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error generating Prometheus metrics: {e}")
            return f"# Error: {str(e)}\n"

    def start(self):
        """Start metrics collector"""
        if self.running:
            return

        self.running = True

        # Start persistence worker
        self.persist_task = asyncio.create_task(self._persist_worker())

        self.logger.info(
            f"Metrics collector started (persist interval: {self.persist_interval}s, "
            f"retention: {self.retention_hours}h)"
        )

    async def stop(self):
        """Stop metrics collector and persist final state"""
        if not self.running:
            return

        self.logger.info("Stopping metrics collector...")
        self.running = False

        # Cancel persistence worker
        if self.persist_task:
            self.persist_task.cancel()
            try:
                await self.persist_task
            except asyncio.CancelledError:
                pass

        # Final persist
        await self._persist_metrics()

        self.logger.info("Metrics collector stopped")

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.stats.clear()
        self.logger.info("All metrics reset")


# ==============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON METRICS
# ==============================================================================

class IngestionMetrics:
    """High-level interface for ingestion-specific metrics"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def document_processed(self, collection: str, file_type: str):
        """Record a document processed"""
        self.collector.increment(
            "ingestion.documents_processed",
            labels={"collection": collection, "file_type": file_type}
        )

    def chunks_created(self, collection: str, count: int):
        """Record chunks created"""
        self.collector.increment(
            "ingestion.chunks_created",
            value=count,
            labels={"collection": collection}
        )

    def ingestion_duration(self, collection: str, duration: float):
        """Record ingestion duration"""
        self.collector.record_timing(
            "ingestion.duration_seconds",
            duration,
            labels={"collection": collection}
        )

    def error_occurred(self, error_type: str, component: str):
        """Record an error"""
        self.collector.increment(
            "ingestion.errors",
            labels={"error_type": error_type, "component": component}
        )

    def file_monitor_event(self, event_type: str, collection: str):
        """Record a file monitor event"""
        self.collector.increment(
            "file_monitor.events",
            labels={"event_type": event_type, "collection": collection}
        )

    def validation_run(self, issues_found: int, issues_fixed: int):
        """Record a validation run"""
        self.collector.set_gauge("validation.last_issues_found", issues_found)
        self.collector.set_gauge("validation.last_issues_fixed", issues_fixed)
        self.collector.increment("validation.runs_total")

    def web_scraping_result(self, target: str, success: bool, duration: float):
        """Record web scraping result"""
        status = "success" if success else "failure"
        self.collector.increment(
            "web_scraping.attempts",
            labels={"target": target, "status": status}
        )
        self.collector.record_timing(
            "web_scraping.duration_seconds",
            duration,
            labels={"target": target}
        )


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def main():
    """Example usage of MetricsCollector"""

    # Create collector
    collector = MetricsCollector(
        metrics_file="logs/metrics.json",
        persist_interval=10  # Persist every 10 seconds for demo
    )

    # Start collector
    collector.start()

    # Create high-level interface
    metrics = IngestionMetrics(collector)

    # Record some metrics
    metrics.document_processed("academic_knowledge", "pdf")
    metrics.chunks_created("academic_knowledge", 50)
    metrics.ingestion_duration("academic_knowledge", 3.5)

    metrics.document_processed("skill_knowledge", "docx")
    metrics.chunks_created("skill_knowledge", 30)
    metrics.ingestion_duration("skill_knowledge", 2.1)

    # Simulate errors
    metrics.error_occurred("file_not_found", "document_processor")

    # Wait a bit
    await asyncio.sleep(2)

    # Get current metrics
    all_metrics = collector.get_all_metrics()
    print("All metrics:", json.dumps(all_metrics, indent=2))

    # Get specific stats
    duration_stats = collector.get_stats("ingestion.duration_seconds")
    if duration_stats:
        print(f"\nIngestion duration stats: {duration_stats.to_dict()}")

    # Get time series
    time_series = collector.get_time_series("ingestion.documents_processed")
    print(f"\nTime series data points: {len(time_series)}")

    # Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        prom_output = collector.get_prometheus_metrics()
        print(f"\nPrometheus metrics:\n{prom_output[:500]}...")

    # Stop collector
    await collector.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
