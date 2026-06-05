#!/usr/bin/env python3
"""Performance monitoring module for diagnosing response time issues."""

import time
import threading
from datetime import datetime
from typing import Dict, List
import psutil
import os


class PerformanceMonitor:
    """Monitor system performance to diagnose response time issues."""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
        self.monitoring = False
        self.thread = None
        
    def start_monitoring(self):
        """Start performance monitoring in a background thread."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            metric = self._collect_metrics()
            with self.lock:
                self.metrics.append(metric)
                # Keep only last 100 metrics to prevent memory growth
                if len(self.metrics) > 100:
                    self.metrics = self.metrics[-100:]
            time.sleep(1)  # Monitor every second
            
    def _collect_metrics(self) -> Dict:
        """Collect current system metrics."""
        try:
            process = psutil.Process(os.getpid())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'process_cpu': process.cpu_percent(),
                'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                'active_threads': threading.active_count(),
            }
        except Exception:
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'process_cpu': 0.0,
                'process_memory_mb': 0.0,
                'active_threads': threading.active_count(),
            }
        
    def get_recent_metrics(self, last_n: int = 10) -> List[Dict]:
        """Get the last N metrics."""
        with self.lock:
            return self.metrics[-last_n:] if self.metrics else []
            
    def print_report(self):
        """Print a performance report."""
        recent_metrics = self.get_recent_metrics()
        if not recent_metrics:
            print("No performance data collected yet.")
            return
            
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_mem = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
        latest_process_cpu = recent_metrics[-1]['process_cpu']
        latest_process_mem = recent_metrics[-1]['process_memory_mb']
        
        print(f"Performance Report:")
        print(f"  System CPU: {avg_cpu:.1f}% (avg), Memory: {avg_mem:.1f}% (avg)")
        print(f"  Process CPU: {latest_process_cpu:.1f}%, Memory: {latest_process_mem:.1f}MB")
        print(f"  Active Threads: {recent_metrics[-1]['active_threads']}")
        

# Global performance monitor instance
perf_monitor = PerformanceMonitor()


def start_performance_monitoring():
    """Start the global performance monitor."""
    perf_monitor.start_monitoring()
    
    
def stop_performance_monitoring():
    """Stop the global performance monitor."""
    perf_monitor.stop_monitoring()
    
    
def get_performance_report():
    """Get the current performance report."""
    perf_monitor.print_report()
