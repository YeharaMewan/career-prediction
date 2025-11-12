"""
Services module for Career Planning System

Contains ingestion services:
- Ingestion Registry: Track file ingestion history
- File Monitor: Real-time file system monitoring
- Scheduled Validator: Periodic consistency checks
- Web Scraper: Automated web content extraction
- Ingestion Orchestrator: Pipeline coordination
- Metrics Collector: Statistics and monitoring
- Alerting: Notification system
- Service Manager: Service orchestration
"""

__all__ = [
    'IngestionRegistry',
    'FileMonitor',
    'ScheduledValidator',
    'WebScraper',
    'IngestionOrchestrator',
    'MetricsCollector',
    'AlertingService',
    'ServiceManager'
]
