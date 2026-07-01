"""
reasoning_dashboard.py - Agent Reasoning Visualization Dashboard

Creates a comprehensive dashboard to visualize:
- Agent reasoning chains
- Decision-making processes
- Confidence scores and metrics
- Cross-agent communication
- Performance analytics
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from .orchestrator import AIOrchestrator

class ReasoningDashboard:
    """Dashboard for visualizing agent reasoning"""
    
    def __init__(self, orchestrator: AIOrchestrator):
        self.orchestrator = orchestrator
        self.dashboard_data = {}
        
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        # Get orchestration metrics
        orchestration_metrics = await self.orchestrator.get_orchestration_metrics()
        
        # Get reasoning visualization data
        reasoning_data = await self.orchestrator.get_reasoning_visualization()
        
        # Get agent status
        agent_status = await self.orchestrator.get_agent_status()
        
        # Process data for visualization
        dashboard_data = {
            'overview': self._generate_overview_data(orchestration_metrics),
            'agent_performance': self._generate_agent_performance_data(agent_status),
            'reasoning_chains': self._generate_reasoning_chains_data(reasoning_data),
            'workflow_analytics': self._generate_workflow_analytics(orchestration_metrics),
            'confidence_trends': self._generate_confidence_trends(reasoning_data),
            'agent_communication': self._generate_communication_data(reasoning_data),
            'performance_metrics': self._generate_performance_metrics(orchestration_metrics)
        }
        
        self.dashboard_data = dashboard_data
        return dashboard_data
    
    def _generate_overview_data(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overview statistics"""
        
        orch_metrics = metrics['orchestration_metrics']
        
        return {
            'total_workflows': orch_metrics['workflows_executed'],
            'success_rate': (
                orch_metrics['workflows_completed'] / max(orch_metrics['workflows_executed'], 1) * 100
            ),
            'avg_execution_time': orch_metrics['total_execution_time_ms'] / max(orch_metrics['workflows_executed'], 1),
            'active_agents': len([agent for agent in orch_metrics['agent_usage'].values() if agent > 0]),
            'total_reasoning_steps': sum(
                agent.get('metrics', {}).get('metrics', {}).get('total_reasoning_steps', 0)
                for agent in metrics['agent_status'].values()
            )
        }
    
    def _generate_agent_performance_data(self, agent_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent performance data"""
        
        performance_data = {}
        
        for agent_name, agent_info in agent_status.items():
            metrics_outer = agent_info['metrics']
            # The inner 'metrics' dict contains actual counters
            metrics = metrics_outer.get('metrics', metrics_outer)
            
            performance_data[agent_name] = {
                'tasks_completed': metrics.get('tasks_completed', 0),
                'tasks_failed': metrics.get('tasks_failed', 0),
                'success_rate': (
                    metrics.get('tasks_completed', 0) / max(metrics.get('tasks_completed', 0) + metrics.get('tasks_failed', 0), 1) * 100
                ),
                'avg_confidence': metrics.get('avg_confidence', 0.0),
                'total_processing_time': metrics.get('total_processing_time_ms', 0),
                'avg_processing_time': (
                    metrics.get('total_processing_time_ms', 0) / max(metrics.get('tasks_completed', 1), 1)
                ),
                'reasoning_steps': metrics.get('total_reasoning_steps', 0)
            }
            
            # Add agent-specific metrics
            if 'qa_metrics' in agent_info:
                performance_data[agent_name]['qa_metrics'] = agent_info['qa_metrics']
            elif 'analysis_metrics' in agent_info:
                performance_data[agent_name]['analysis_metrics'] = agent_info['analysis_metrics']
            elif 'question_metrics' in agent_info:
                performance_data[agent_name]['question_metrics'] = agent_info['question_metrics']
        
        return performance_data
    
    def _generate_reasoning_chains_data(self, reasoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning chains visualization data"""
        
        chains_data = {
            'total_chains': 0,
            'total_steps': 0,
            'avg_steps_per_chain': 0,
            'step_types': {},
            'confidence_distribution': [],
            'duration_distribution': []
        }
        
        if 'workflows' in reasoning_data:
            for workflow in reasoning_data['workflows']:
                for agent in workflow['agents']:
                    chains_data['total_chains'] += 1
                    chains_data['total_steps'] += agent['step_count']
                    
                    # Analyze step types
                    for step in agent['reasoning_steps']:
                        step_type = step.get('step_type', 'unknown')
                        chains_data['step_types'][step_type] = chains_data['step_types'].get(step_type, 0) + 1
                        
                        # Collect confidence and duration data
                        confidence = step.get('confidence', 0)
                        duration = step.get('duration_ms', 0)
                        
                        chains_data['confidence_distribution'].append(confidence)
                        chains_data['duration_distribution'].append(duration)
        
        # Calculate averages
        if chains_data['total_chains'] > 0:
            chains_data['avg_steps_per_chain'] = chains_data['total_steps'] / chains_data['total_chains']
        
        return chains_data
    
    def _generate_workflow_analytics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate workflow analytics data"""
        
        orch_metrics = metrics['orchestration_metrics']
        
        return {
            'workflow_types': orch_metrics['workflow_types'],
            'agent_usage': orch_metrics['agent_usage'],
            'total_execution_time': orch_metrics['total_execution_time_ms'],
            'avg_execution_time': orch_metrics['total_execution_time_ms'] / max(orch_metrics['workflows_executed'], 1)
        }
    
    def _generate_confidence_trends(self, reasoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence trends data"""
        
        trends_data = {
            'overall_confidence': [],
            'agent_confidence': {},
            'confidence_by_step_type': {}
        }
        
        if 'workflows' in reasoning_data:
            for workflow in reasoning_data['workflows']:
                for agent in workflow['agents']:
                    agent_name = agent['agent_name']
                    
                    # Initialize agent data
                    if agent_name not in trends_data['agent_confidence']:
                        trends_data['agent_confidence'][agent_name] = []
                    
                    # Process reasoning steps
                    for step in agent['reasoning_steps']:
                        confidence = step.get('confidence', 0)
                        step_type = step.get('step_type', 'unknown')
                        
                        trends_data['overall_confidence'].append(confidence)
                        trends_data['agent_confidence'][agent_name].append(confidence)
                        
                        # Initialize step type data
                        if step_type not in trends_data['confidence_by_step_type']:
                            trends_data['confidence_by_step_type'][step_type] = []
                        
                        trends_data['confidence_by_step_type'][step_type].append(confidence)
        
        return trends_data
    
    def _generate_communication_data(self, reasoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent communication data"""
        
        communication_data = {
            'workflow_flows': [],
            'agent_interactions': {},
            'dependency_chains': []
        }
        
        if 'workflows' in reasoning_data:
            for workflow in reasoning_data['workflows']:
                flow_data = {
                    'workflow_id': workflow['workflow_id'],
                    'workflow_type': workflow['workflow_type'],
                    'agents': [agent['agent_name'] for agent in workflow['agents']],
                    'execution_order': list(range(len(workflow['agents'])))
                }
                communication_data['workflow_flows'].append(flow_data)
                
                # Track agent interactions
                for i, agent in enumerate(workflow['agents']):
                    agent_name = agent['agent_name']
                    
                    if agent_name not in communication_data['agent_interactions']:
                        communication_data['agent_interactions'][agent_name] = {
                            'total_workflows': 0,
                            'positions': [],
                            'avg_confidence': 0
                        }
                    
                    communication_data['agent_interactions'][agent_name]['total_workflows'] += 1
                    communication_data['agent_interactions'][agent_name]['positions'].append(i)
                    communication_data['agent_interactions'][agent_name]['avg_confidence'] = agent['avg_confidence']
        
        return communication_data
    
    def _generate_performance_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance metrics data"""
        
        orch_metrics = metrics['orchestration_metrics']
        
        return {
            'throughput': orch_metrics['workflows_executed'],
            'error_rate': (
                orch_metrics['workflows_failed'] / max(orch_metrics['workflows_executed'], 1) * 100
            ),
            'avg_response_time': orch_metrics['total_execution_time_ms'] / max(orch_metrics['workflows_executed'], 1),
            'agent_utilization': {
                agent: usage / max(orch_metrics['workflows_executed'], 1) * 100
                for agent, usage in orch_metrics['agent_usage'].items()
            }
        }
    
    def create_overview_chart(self) -> go.Figure:
        """Create overview chart"""
        
        if not self.dashboard_data:
            return go.Figure()
        
        overview = self.dashboard_data['overview']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Workflows Executed', 'Success Rate', 'Avg Execution Time', 'Active Agents'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Workflows executed
        fig.add_trace(
            go.Bar(x=['Total'], y=[overview['total_workflows']], name='Workflows'),
            row=1, col=1
        )
        
        # Success rate
        fig.add_trace(
            go.Pie(
                labels=['Success', 'Failure'],
                values=[overview['success_rate'], 100 - overview['success_rate']],
                name='Success Rate'
            ),
            row=1, col=2
        )
        
        # Avg execution time
        fig.add_trace(
            go.Bar(x=['Avg Time'], y=[overview['avg_execution_time']/1000], name='Time (s)'),
            row=2, col=1
        )
        
        # Active agents
        fig.add_trace(
            go.Bar(x=['Active'], y=[overview['active_agents']], name='Agents'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="AI Orchestration Overview",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_agent_performance_chart(self) -> go.Figure:
        """Create agent performance chart"""
        
        if not self.dashboard_data:
            return go.Figure()
        
        performance = self.dashboard_data['agent_performance']
        
        agents = list(performance.keys())
        success_rates = [performance[agent]['success_rate'] for agent in agents]
        avg_confidence = [performance[agent]['avg_confidence'] * 100 for agent in agents]
        tasks_completed = [performance[agent]['tasks_completed'] for agent in agents]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate (%)', 'Avg Confidence (%)', 'Tasks Completed', 'Processing Time (ms)'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Success rate
        fig.add_trace(
            go.Bar(x=agents, y=success_rates, name='Success Rate'),
            row=1, col=1
        )
        
        # Avg confidence
        fig.add_trace(
            go.Bar(x=agents, y=avg_confidence, name='Confidence'),
            row=1, col=2
        )
        
        # Tasks completed
        fig.add_trace(
            go.Bar(x=agents, y=tasks_completed, name='Tasks'),
            row=2, col=1
        )
        
        # Processing time
        processing_times = [performance[agent]['avg_processing_time'] for agent in agents]
        fig.add_trace(
            go.Bar(x=agents, y=processing_times, name='Processing Time'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Agent Performance Metrics",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_reasoning_flow_chart(self) -> go.Figure:
        """Create reasoning flow visualization"""
        
        if not self.dashboard_data:
            return go.Figure()
        
        reasoning = self.dashboard_data['reasoning_chains']
        
        # Create sankey-style flow diagram
        step_types = list(reasoning['step_types'].keys())
        step_counts = list(reasoning['step_types'].values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=step_types,
                y=step_counts,
                name='Reasoning Steps'
            )
        ])
        
        fig.update_layout(
            title="Reasoning Step Distribution",
            xaxis_title="Step Type",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_confidence_trend_chart(self) -> go.Figure:
        """Create confidence trend chart"""
        
        if not self.dashboard_data:
            return go.Figure()
        
        trends = self.dashboard_data['confidence_trends']
        
        fig = go.Figure()
        
        # Overall confidence trend
        if trends['overall_confidence']:
            fig.add_trace(
                go.Scatter(
                    y=trends['overall_confidence'],
                    mode='lines+markers',
                    name='Overall Confidence',
                    line=dict(color='blue', width=2)
                )
            )
        
        # Agent-specific confidence trends
        colors = ['red', 'green', 'orange']
        for i, (agent, confidences) in enumerate(trends['agent_confidence'].items()):
            if confidences:
                fig.add_trace(
                    go.Scatter(
                        y=confidences,
                        mode='lines+markers',
                        name=f'{agent} Confidence',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                    )
                )
        
        fig.update_layout(
            title="Confidence Trends Over Time",
            xaxis_title="Step Number",
            yaxis_title="Confidence Score",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def export_dashboard_html(self, output_path: str = "reasoning_dashboard.html"):
        """Export dashboard as HTML file"""
        
        # Generate data
        asyncio.run(self.generate_dashboard_data())
        
        # Create charts
        overview_fig = self.create_overview_chart()
        performance_fig = self.create_agent_performance_chart()
        reasoning_fig = self.create_reasoning_flow_chart()
        confidence_fig = self.create_confidence_trend_chart()
        
        # Create HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Agent Reasoning Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ margin: 20px 0; }}
                .header {{ text-align: center; color: #333; }}
                .stats {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1 class="header">AI Agent Reasoning Dashboard</h1>
            
            <div class="stats">
                <h2>Overview Statistics</h2>
                <p>Total Workflows: {self.dashboard_data['overview']['total_workflows']}</p>
                <p>Success Rate: {self.dashboard_data['overview']['success_rate']:.1f}%</p>
                <p>Avg Execution Time: {self.dashboard_data['overview']['avg_execution_time']/1000:.1f}s</p>
                <p>Active Agents: {self.dashboard_data['overview']['active_agents']}</p>
            </div>
            
            <div class="chart-container">
                <div id="overview-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="performance-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="reasoning-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="confidence-chart"></div>
            </div>
            
            <script>
                Plotly.newPlot('overview-chart', {overview_fig.to_json()});
                Plotly.newPlot('performance-chart', {performance_fig.to_json()});
                Plotly.newPlot('reasoning-chart', {reasoning_fig.to_json()});
                Plotly.newPlot('confidence-chart', {confidence_fig.to_json()});
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Dashboard exported to {output_path}")
    
    def get_reasoning_summary_report(self) -> str:
        """Generate text summary of reasoning data"""
        
        if not self.dashboard_data:
            return "No data available"
        
        overview = self.dashboard_data['overview']
        performance = self.dashboard_data['agent_performance']
        reasoning = self.dashboard_data['reasoning_chains']
        
        report = f"""
        AI AGENT REASONING SUMMARY REPORT
        ==================================
        
        OVERVIEW:
        - Total Workflows Executed: {overview['total_workflows']}
        - Success Rate: {overview['success_rate']:.1f}%
        - Average Execution Time: {overview['avg_execution_time']/1000:.1f} seconds
        - Active Agents: {overview['active_agents']}
        - Total Reasoning Steps: {overview['total_reasoning_steps']}
        
        AGENT PERFORMANCE:
        """
        
        for agent_name, metrics in performance.items():
            report += f"""
        {agent_name.upper()}:
        - Tasks Completed: {metrics['tasks_completed']}
        - Success Rate: {metrics['success_rate']:.1f}%
        - Average Confidence: {metrics['avg_confidence']:.3f}
        - Average Processing Time: {metrics['avg_processing_time']:.1f}ms
        - Reasoning Steps: {metrics['reasoning_steps']}
            """
        
        report += f"""
        REASONING ANALYSIS:
        - Total Reasoning Chains: {reasoning['total_chains']}
        - Total Steps: {reasoning['total_steps']}
        - Average Steps per Chain: {reasoning['avg_steps_per_chain']:.1f}
        
        Step Types Distribution:
        """
        
        for step_type, count in reasoning['step_types'].items():
            report += f"        - {step_type}: {count}\n"
        
        return report
