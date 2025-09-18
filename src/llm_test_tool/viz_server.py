#!/usr/bin/env python3
"""
FastAPI server for LLM performance visualization data API.
"""

import json
import os
import re
import logging
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
import pandas as pd
import uvicorn

# Get root path from environment variable if set
root_path = os.environ.get('ROOT_PATH', '')

app = FastAPI(
    title="LLM Performance Visualization API", 
    version="1.0.0",
    root_path=root_path
)

# Get analytics configuration from environment variables
analytics_log_file = os.environ.get('ANALYTICS_LOG', 'viz_access.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(analytics_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UserAnalytics:
    """Analytics tracker for user behavior"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            log_file = os.environ.get('ANALYTICS_FILE', 'user_analytics.json')
        self.log_file = Path(log_file)
        self.sessions = {}
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing analytics data"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', {})
            except Exception as e:
                logger.error(f"Failed to load analytics data: {e}")
                self.sessions = {}
    
    def get_client_id(self, request: Request) -> str:
        """Generate or retrieve client ID from request"""
        # Try to get client ID from headers (set by client-side JS)
        client_id = request.headers.get('X-Client-ID')
        if not client_id:
            # Fallback to IP-based identification
            client_ip = request.client.host if request.client else 'unknown'
            user_agent = request.headers.get('User-Agent', 'unknown')
            client_id = f"{client_ip}_{hash(user_agent) % 10000}"
        return client_id
    
    def log_event(self, request: Request, event_type: str, data: dict = None):
        """Log user event"""
        client_id = self.get_client_id(request)
        timestamp = datetime.now().isoformat()
        
        if client_id not in self.sessions:
            self.sessions[client_id] = {
                'first_visit': timestamp,
                'last_visit': timestamp,
                'visit_count': 0,
                'events': []
            }
        
        # Update session info
        self.sessions[client_id]['last_visit'] = timestamp
        self.sessions[client_id]['visit_count'] += 1
        
        # Add event
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'data': data or {},
            'ip': request.client.host if request.client else 'unknown',
            'user_agent': request.headers.get('User-Agent', 'unknown')
        }
        
        self.sessions[client_id]['events'].append(event)
        
        # Save to file
        self.save_data()
        
        # Log to console
        logger.info(f"Analytics: {client_id} - {event_type} - {data}")
    
    def save_data(self):
        """Save analytics data to file"""
        try:
            analytics_data = {
                'last_updated': datetime.now().isoformat(),
                'total_users': len(self.sessions),
                'sessions': self.sessions
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(analytics_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save analytics data: {e}")
    
    def get_stats(self) -> dict:
        """Get analytics statistics"""
        if not self.sessions:
            return {
                'total_users': 0,
                'total_page_views': 0,
                'total_model_selections': 0,
                'total_chart_additions': 0,
                'popular_models': [],
                'popular_runtimes': [],
                'popular_instances': [],
                'recent_chart_additions': []
            }
        
        # Count only meaningful events
        page_view_count = 0
        model_selection_count = 0
        chart_addition_count = 0
        model_counts = {}
        runtime_counts = {}
        instance_counts = {}
        recent_chart_additions = []
        
        for session in self.sessions.values():
            for event in session['events']:
                # Count page visits
                if event['type'] == 'page_visit':
                    page_view_count += 1
                
                # Count model selections
                elif event['type'] == 'model_selected':
                    model_selection_count += 1
                    
                    model = event['data'].get('model_name', 'unknown')
                    runtime = event['data'].get('runtime', 'unknown')
                    instance = event['data'].get('instance_type', 'unknown')
                    
                    model_counts[model] = model_counts.get(model, 0) + 1
                    runtime_counts[runtime] = runtime_counts.get(runtime, 0) + 1
                    instance_counts[instance] = instance_counts.get(instance, 0) + 1
                
                # Count chart additions (more meaningful metric)
                elif event['type'] in ['chart_added_successfully', 'model_added_to_comparison']:
                    chart_addition_count += 1
                    recent_chart_additions.append(event)
        
        # Sort by popularity
        popular_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        popular_runtimes = sorted(runtime_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        popular_instances = sorted(instance_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Get recent chart additions (last 20)
        recent_chart_additions.sort(key=lambda x: x['timestamp'], reverse=True)
        recent_chart_additions = recent_chart_additions[:20]
        
        return {
            'total_users': len(self.sessions),
            'total_page_views': page_view_count,
            'total_model_selections': model_selection_count,
            'total_chart_additions': chart_addition_count,
            'popular_models': popular_models,
            'popular_runtimes': popular_runtimes,
            'popular_instances': popular_instances,
            'recent_chart_additions': recent_chart_additions
        }

# Initialize analytics
analytics = UserAnalytics()

# Record server start time for uptime calculation
start_time = time.time()


class PriceProvider:
    """Provider for instance pricing information"""
    
    def __init__(self, price_config_path: str = "instance_prices.json"):
        self.price_config_path = Path(price_config_path)
        self.prices = {}
        self.load_prices()
    
    def load_prices(self):
        """Load instance prices from configuration file"""
        try:
            if self.price_config_path.exists():
                with open(self.price_config_path, 'r') as f:
                    config = json.load(f)
                    self.prices = config.get('aws_ec2_prices', {})
                    print(f"Loaded pricing for {len(self.prices)} instance types")
            else:
                print(f"Price config file not found: {self.price_config_path}")
                self.prices = {}
        except Exception as e:
            print(f"Error loading price config: {e}")
            self.prices = {}
    
    def get_price(self, instance_type: str) -> float:
        """Get price for a specific instance type"""
        return self.prices.get(instance_type, 0.0)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all available instance prices"""
        return self.prices.copy()


class ResultsDataProvider:
    """Data provider for LLM performance test results"""
    
    def __init__(self, results_dir: str = "archive_results"):
        self.results_dir = Path(results_dir)
        self.data = []
        self.df = None
        self.load_all_results()
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """Parse test result filename to extract parameters"""
        pattern = r'test_in:(\d+)_out:(\d+)_proc:(\d+)_rand:(\d+)\.json'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'input_tokens': int(match.group(1)),
                'output_tokens': int(match.group(2)),
                'processes': int(match.group(3)),
                'random_tokens': int(match.group(4))
            }
        return None
    
    def parse_directory_name(self, dirname: str) -> Optional[Dict[str, str]]:
        """Parse directory name to extract runtime, instance type, and model"""
        parts = dirname.split('--')
        if len(parts) >= 3:
            model_name = '--'.join(parts[2:])
            # Strip .yaml suffix if present
            if model_name.endswith('.yaml'):
                model_name = model_name[:-5]
            
            return {
                'runtime': parts[0],
                'instance_type': parts[1],
                'model_name': model_name
            }
        return None
    
    def load_all_results(self):
        """Load all test results from the archive directory"""
        print(f"Loading results from {self.results_dir}...")
        self.data = []
        for result_dir in self.results_dir.iterdir():
            if not result_dir.is_dir():
                continue
            
            dir_info = self.parse_directory_name(result_dir.name)
            if not dir_info:
                continue
            
            for result_file in result_dir.glob("test_*.json"):
                file_info = self.parse_filename(result_file.name)
                if not file_info:
                    continue
                
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    
                    stats = result_data.get('statistics', {})
                    metadata = result_data.get('metadata', {})
                    
                    record = {
                        **dir_info,
                        **file_info,
                        'first_token_latency_mean': stats.get('first_token_latency', {}).get('mean', 0),
                        'first_token_latency_p50': stats.get('first_token_latency', {}).get('p50', 0),
                        'first_token_latency_p90': stats.get('first_token_latency', {}).get('p90', 0),
                        'first_token_latency_min': stats.get('first_token_latency', {}).get('min', 0),
                        'first_token_latency_max': stats.get('first_token_latency', {}).get('max', 0),
                        'end_to_end_latency_mean': stats.get('end_to_end_latency', {}).get('mean', 0),
                        'end_to_end_latency_p50': stats.get('end_to_end_latency', {}).get('p50', 0),
                        'end_to_end_latency_p90': stats.get('end_to_end_latency', {}).get('p90', 0),
                        'output_tokens_per_second_mean': stats.get('output_tokens_per_second', {}).get('mean', 0),
                        'output_tokens_per_second_p50': stats.get('output_tokens_per_second', {}).get('p50', 0),
                        'output_tokens_per_second_p90': stats.get('output_tokens_per_second', {}).get('p90', 0),
                        'output_tokens_per_second_min': stats.get('output_tokens_per_second', {}).get('min', 0),
                        'output_tokens_per_second_max': stats.get('output_tokens_per_second', {}).get('max', 0),
                        'success_rate': stats.get('success_rate', 0),
                        'requests_per_second': metadata.get('requests_per_second', 0),
                        'total_requests': metadata.get('total_requests', 0),
                        'successful_requests': stats.get('successful_requests', 0),
                        'failed_requests': stats.get('failed_requests', 0),
                        'total_tokens_mean': stats.get('token_usage', {}).get('total_tokens', {}).get('mean', 0),
                        'server_throughput': metadata.get('requests_per_second', 0) * stats.get('token_usage', {}).get('total_tokens', {}).get('mean', 0),
                        'file_path': str(result_file)
                    }
                    
                    self.data.append(record)
                    
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
        
        self.df = pd.DataFrame(self.data)
        print(f"Loaded {len(self.data)} test results")
    
    def get_combinations(self) -> List[Dict[str, str]]:
        """Get all available runtime-instance-model combinations"""
        if self.df is None or self.df.empty:
            return []
        
        combinations = self.df[['runtime', 'instance_type', 'model_name']].drop_duplicates()
        return [
            {
                'runtime': row['runtime'],
                'instance_type': row['instance_type'],
                'model_name': row['model_name'],
                'id': f"{row['runtime']}--{row['instance_type']}--{row['model_name']}"
            }
            for _, row in combinations.iterrows()
        ]
    
    def get_test_parameters(self, runtime: str, instance_type: str, model_name: str) -> Dict[str, List]:
        """Get available test parameters for a specific combination"""
        if self.df is None or self.df.empty:
            return {}
        print(runtime, instance_type, model_name)
        filtered = self.df[
            (self.df['runtime'] == runtime) & 
            (self.df['instance_type'] == instance_type) & 
            (self.df['model_name'] == model_name)
        ]
        
        return {
            'input_tokens': sorted(filtered['input_tokens'].unique().tolist(), key=int),
            'output_tokens': sorted(filtered['output_tokens'].unique().tolist(), key=int),
            'random_tokens': sorted(filtered['random_tokens'].unique().tolist(), key=int)
        }
    
    def get_performance_data(self, filters: Dict) -> List[Dict]:
        """Get performance data based on filters"""
        if self.df is None or self.df.empty:
            return []
        
        filtered_df = self.df.copy()
        
        # Apply filters
        for key, value in filters.items():
            if key in filtered_df.columns and value is not None:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
        
        # Sort by processes for proper line plotting
        filtered_df = filtered_df.sort_values('processes')
        
        return filtered_df.to_dict('records')


# Pydantic models for request/response
class ComparisonRequest(BaseModel):
    combinations: List[Dict]

class CombinationInfo(BaseModel):
    runtime: str
    instance_type: str
    model_name: str
    id: str

class AnalyticsEvent(BaseModel):
    event_type: str
    data: Optional[Dict] = None

# Initialize data and price providers
# Get results directory from environment variable if set
results_dir = os.environ.get('RESULTS_DIR', 'archive_results')
data_provider = ResultsDataProvider(results_dir)
price_provider = PriceProvider()


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    try:
        # Check if data provider is working
        data_status = "ok" if data_provider.df is not None else "no_data"
        
        # Check analytics system
        analytics_status = "ok" if analytics.log_file.parent.exists() else "error"
        
        # Check if we can write to analytics file
        try:
            analytics.log_file.parent.mkdir(parents=True, exist_ok=True)
            analytics_writable = True
        except Exception:
            analytics_writable = False
        
        # Overall health status
        overall_status = "healthy" if (
            data_status in ["ok", "no_data"] and 
            analytics_status == "ok" and 
            analytics_writable
        ) else "unhealthy"
        
        health_data = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "data_provider": {
                    "status": data_status,
                    "total_results": len(data_provider.data) if data_provider.data else 0,
                    "results_directory": str(data_provider.results_dir)
                },
                "analytics": {
                    "status": analytics_status,
                    "writable": analytics_writable,
                    "file_path": str(analytics.log_file),
                    "total_users": len(analytics.sessions)
                },
                "price_provider": {
                    "status": "ok" if price_provider.prices else "no_prices",
                    "available_instances": len(price_provider.prices)
                }
            },
            "uptime_seconds": time.time() - start_time
        }
        
        # Return 200 for healthy, 503 for unhealthy
        status_code = 200 if overall_status == "healthy" else 503
        
        return JSONResponse(
            content=health_data,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            status_code=503
        )

@app.get("/health/ready")
async def readiness_check():
    """Simple readiness check for Kubernetes/Docker"""
    try:
        # Basic checks for readiness
        data_ready = data_provider is not None
        analytics_ready = analytics is not None
        
        if data_ready and analytics_ready:
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        else:
            return JSONResponse(
                content={"status": "not_ready", "timestamp": datetime.now().isoformat()},
                status_code=503
            )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()},
            status_code=503
        )

@app.get("/health/live")
async def liveness_check():
    """Simple liveness check for Kubernetes/Docker"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def index(request: Request):
    """Serve the main HTML page with root path injection"""
    # Log page visit
    analytics.log_event(request, 'page_visit', {'page': 'index'})
    
    # Get the directory where this script is located
    current_dir = Path(__file__).parent
    html_file = current_dir / 'index.html'
    
    # Read the HTML file and inject the root path
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Generate client ID for this session
    client_id = analytics.get_client_id(request)
    
    # Inject the root path and client ID as JavaScript variables
    root_path_script = f"""
    <script>
        window.API_BASE_PATH = '{root_path}';
        window.CLIENT_ID = '{client_id}';
    </script>
    """
    
    # Insert the script before the closing </head> tag
    html_content = html_content.replace('</head>', f'{root_path_script}</head>')
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/analytics")
async def analytics_dashboard(request: Request):
    """Serve the analytics dashboard page"""
    # Log analytics dashboard access
    analytics.log_event(request, 'analytics_dashboard_visit')
    
    # Get the directory where this script is located
    current_dir = Path(__file__).parent
    html_file = current_dir / 'analytics_dashboard.html'
    
    # Read the HTML file and inject the root path
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Inject the root path as a JavaScript variable
    root_path_script = f"""
    <script>
        window.API_BASE_PATH = '{root_path}';
    </script>
    """
    
    # Insert the script before the closing </head> tag
    html_content = html_content.replace('</head>', f'{root_path_script}</head>')
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


@app.post("/api/analytics")
async def log_analytics_event(request: Request, event: AnalyticsEvent):
    """Log analytics event from client"""
    analytics.log_event(request, event.event_type, event.data)
    return {"status": "logged"}

@app.get("/api/analytics/stats")
async def get_analytics_stats(request: Request):
    """Get analytics statistics (admin only)"""
    # Log admin access
    analytics.log_event(request, 'admin_stats_access')
    return analytics.get_stats()

@app.get("/api/combinations", response_model=List[CombinationInfo])
async def get_combinations(request: Request):
    """Get all available runtime-instance-model combinations"""
    analytics.log_event(request, 'api_combinations_access')
    combinations = data_provider.get_combinations()
    return combinations


@app.get("/api/parameters")
async def get_parameters(
    request: Request,
    runtime: str = Query(..., description="Runtime name"),
    instance_type: str = Query(..., description="Instance type"),
    model_name: str = Query(..., description="Model name")
):
    """Get available test parameters for a specific combination"""
    analytics.log_event(request, 'model_selected', {
        'runtime': runtime,
        'instance_type': instance_type,
        'model_name': model_name
    })
    parameters = data_provider.get_test_parameters(runtime, instance_type, model_name)
    return parameters


@app.get("/api/instance-prices")
async def get_instance_prices():
    """Get all available instance prices"""
    return {
        "prices": price_provider.get_all_prices(),
        "config_file": str(price_provider.price_config_path)
    }


@app.get("/api/performance-data")
async def get_performance_data(
    runtime: Optional[str] = Query(None),
    instance_type: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    input_tokens: Optional[int] = Query(None),
    output_tokens: Optional[int] = Query(None),
    random_tokens: Optional[int] = Query(None)
):
    """Get performance data based on filters"""
    filters = {}
    
    # Build filters from query parameters
    if runtime:
        filters['runtime'] = runtime
    if instance_type:
        filters['instance_type'] = instance_type
    if model_name:
        filters['model_name'] = model_name
    if input_tokens is not None:
        filters['input_tokens'] = input_tokens
    if output_tokens is not None:
        filters['output_tokens'] = output_tokens
    if random_tokens is not None:
        filters['random_tokens'] = random_tokens
    
    data = data_provider.get_performance_data(filters)
    return data


class ComparisonRequestWithPrice(BaseModel):
    combinations: List[Dict]
    instance_price: Optional[float] = 0.0

@app.post("/api/comparison-data")
async def get_comparison_data(http_request: Request, request: ComparisonRequest):
    """Get performance data for multiple combinations for comparison"""
    try:
        # Log comparison request
        analytics.log_event(http_request, 'comparison_generated', {
            'combinations_count': len(request.combinations),
            'combinations': [
                f"{combo.get('runtime', 'unknown')}-{combo.get('instance_type', 'unknown')}-{combo.get('model_name', 'unknown')}"
                for combo in request.combinations
            ]
        })
        
        result = []
        for combo in request.combinations:
            data = data_provider.get_performance_data(combo)
            
            # Get instance price from config
            instance_type = combo.get('instance_type', '')
            instance_price = price_provider.get_price(instance_type)
            
            # Calculate cost metrics and additional throughput metrics
            for record in data:
                # Get basic metrics
                first_token_latency = record.get('first_token_latency_mean', 0)
                end_to_end_latency = record.get('end_to_end_latency_mean', 0)
                input_tokens = record.get('input_tokens', 0)
                output_tokens = record.get('output_tokens', 0)
                processes = record.get('processes', 1)  # Get concurrency level
                requests_per_second = record.get('requests_per_second', 0)
                server_throughput = record.get('server_throughput', 0)
                
                # Calculate input throughput (tokens/sec)
                # Input throughput = (input_tokens * processes) / first_token_latency
                if first_token_latency > 0 and input_tokens > 0:
                    input_throughput = (input_tokens * processes) / first_token_latency * (requests_per_second / (processes / end_to_end_latency))
                    record['input_throughput'] = input_throughput
                else:
                    record['input_throughput'] = 0
                
                # Calculate output throughput (tokens/sec) 
                # Output latency = end_to_end_latency - first_token_latency
                # Output throughput = (output_tokens * processes) / output_latency
                output_latency = end_to_end_latency - first_token_latency
                if output_latency > 0 and output_tokens > 0:
                    output_throughput = (output_tokens * processes) / output_latency * (requests_per_second / (processes / end_to_end_latency))
                    record['output_throughput'] = output_throughput
                else:
                    record['output_throughput'] = 0
                
                # Calculate cost metrics if we have a price
                if instance_price and instance_price > 0:
                    # Existing cost calculations
                    if server_throughput > 0:
                        cost_per_million_tokens = (instance_price / server_throughput) * 1000000 / 3600
                        record['cost_per_million_tokens'] = cost_per_million_tokens
                    else:
                        record['cost_per_million_tokens'] = 0
                    
                    if requests_per_second > 0:
                        cost_per_1k_requests = (instance_price / requests_per_second) * 1000 / 3600
                        record['cost_per_1k_requests'] = cost_per_1k_requests
                    else:
                        record['cost_per_1k_requests'] = 0
                    
                    # Calculate input token pricing
                    # Cost per million input tokens = instance_price_per_hour / (input_throughput_tokens_per_second * 3600) * 1,000,000
                    input_throughput_val = record.get('input_throughput', 0)
                    if input_throughput_val > 0:
                        cost_per_million_input_tokens = (instance_price / input_throughput_val) * 1000000 / 3600
                        record['cost_per_million_input_tokens'] = cost_per_million_input_tokens
                    else:
                        record['cost_per_million_input_tokens'] = 0
                    
                    # Calculate output token pricing
                    # Cost per million output tokens = instance_price_per_hour / (output_throughput_tokens_per_second * 3600) * 1,000,000
                    output_throughput_val = record.get('output_throughput', 0)
                    if output_throughput_val > 0:
                        cost_per_million_output_tokens = (instance_price / output_throughput_val) * 1000000 / 3600
                        record['cost_per_million_output_tokens'] = cost_per_million_output_tokens
                    else:
                        record['cost_per_million_output_tokens'] = 0
                    
                    record['instance_price_used'] = instance_price
                else:
                    # Set cost to 0 if no price available
                    record['cost_per_million_tokens'] = 0
                    record['cost_per_1k_requests'] = 0
                    record['cost_per_million_input_tokens'] = 0
                    record['cost_per_million_output_tokens'] = 0
                    record['instance_price_used'] = 0
            
            result.append({
                'combination': combo,
                'data': data
            })
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tree-structure")
async def get_tree_structure(request: Request, reload: bool = Query(False, description="Whether to reload data from disk")):
    """Get hierarchical tree structure of Runtime -> Instance Type -> Model"""
    # Log tree structure access
    analytics.log_event(request, 'tree_structure_access', {'reload': reload})
    
    # Only reload all results from disk if explicitly requested (e.g., by refresh button)
    if reload:
        data_provider.load_all_results()
    
    if data_provider.df is None or data_provider.df.empty:
        return {"tree": []}
    
    df = data_provider.df
    tree = {}
    
    # Build hierarchical structure
    for _, row in df.iterrows():
        runtime = row['runtime']
        instance_type = row['instance_type']
        model_name = row['model_name']
        
        if runtime not in tree:
            tree[runtime] = {}
        
        if instance_type not in tree[runtime]:
            tree[runtime][instance_type] = set()
        
        tree[runtime][instance_type].add(model_name)
    
    # Convert to list format with counts
    result = []
    for runtime in sorted(tree.keys()):
        runtime_node = {
            'id': runtime,
            'label': runtime,
            'type': 'runtime',
            'count': len(df[df['runtime'] == runtime]),
            'children': []
        }
        
        for instance_type in sorted(tree[runtime].keys()):
            instance_node = {
                'id': f"{runtime}--{instance_type}",
                'label': instance_type,
                'type': 'instance_type',
                'count': len(df[(df['runtime'] == runtime) & (df['instance_type'] == instance_type)]),
                'children': []
            }
            
            for model_name in sorted(tree[runtime][instance_type]):
                model_node = {
                    'id': f"{runtime}--{instance_type}--{model_name}",
                    'label': model_name,
                    'type': 'model',
                    'count': len(df[(df['runtime'] == runtime) & 
                                   (df['instance_type'] == instance_type) & 
                                   (df['model_name'] == model_name)]),
                    'runtime': runtime,
                    'instance_type': instance_type,
                    'model_name': model_name
                }
                instance_node['children'].append(model_node)
            
            runtime_node['children'].append(instance_node)
        
        result.append(runtime_node)
    
    return {"tree": result}


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics about the dataset"""
    if data_provider.df is None or data_provider.df.empty:
        raise HTTPException(status_code=404, detail="No data available")
    
    df = data_provider.df
    
    stats = {
        'total_tests': len(df),
        'unique_combinations': len(df[['runtime', 'instance_type', 'model_name']].drop_duplicates()),
        'runtimes': sorted(df['runtime'].unique().tolist()),
        'instance_types': sorted(df['instance_type'].unique().tolist()),
        'models': sorted(df['model_name'].unique().tolist()),
        'input_token_range': [int(df['input_tokens'].min()), int(df['input_tokens'].max())],
        'output_token_range': [int(df['output_tokens'].min()), int(df['output_tokens'].max())],
        'process_range': [int(df['processes'].min()), int(df['processes'].max())],
        'performance_summary': {
            'avg_first_token_latency': float(df['first_token_latency_mean'].mean()),
            'avg_throughput': float(df['output_tokens_per_second_mean'].mean()),
            'avg_server_throughput': float(df['server_throughput'].mean()) if 'server_throughput' in df.columns else 0,
            'avg_success_rate': float(df['success_rate'].mean())
        }
    }
    
    return stats


@app.post("/api/export-csv")
async def export_csv(http_request: Request, request: ComparisonRequest):
    """Export performance data as CSV file - ALB compatible"""
    try:
        # Log export request
        analytics.log_event(http_request, 'server_export_requested', {
            'combinations_count': len(request.combinations),
            'combinations': [
                f"{combo.get('runtime', 'unknown')}-{combo.get('instance_type', 'unknown')}-{combo.get('model_name', 'unknown')}"
                for combo in request.combinations
            ]
        })
        
        # Get the data using the same logic as comparison-data
        result = []
        for combo in request.combinations:
            data = data_provider.get_performance_data(combo)
            
            # Get instance price from config
            instance_type = combo.get('instance_type', '')
            instance_price = price_provider.get_price(instance_type)
            
            # Calculate cost metrics and additional throughput metrics
            for record in data:
                # Get basic metrics
                first_token_latency = record.get('first_token_latency_mean', 0)
                end_to_end_latency = record.get('end_to_end_latency_mean', 0)
                input_tokens = record.get('input_tokens', 0)
                output_tokens = record.get('output_tokens', 0)
                processes = record.get('processes', 1)
                requests_per_second = record.get('requests_per_second', 0)
                server_throughput = record.get('server_throughput', 0)
                
                # Calculate input throughput (tokens/sec)
                if first_token_latency > 0 and input_tokens > 0:
                    input_throughput = (input_tokens * processes) / first_token_latency * (requests_per_second / (processes / end_to_end_latency))
                    record['input_throughput'] = input_throughput
                else:
                    record['input_throughput'] = 0
                
                # Calculate output throughput (tokens/sec) 
                output_latency = end_to_end_latency - first_token_latency
                if output_latency > 0 and output_tokens > 0:
                    output_throughput = (output_tokens * processes) / output_latency * (requests_per_second / (processes / end_to_end_latency))
                    record['output_throughput'] = output_throughput
                else:
                    record['output_throughput'] = 0
                
                # Calculate cost metrics if we have a price
                if instance_price and instance_price > 0:
                    if server_throughput > 0:
                        cost_per_million_tokens = (instance_price / server_throughput) * 1000000 / 3600
                        record['cost_per_million_tokens'] = cost_per_million_tokens
                    else:
                        record['cost_per_million_tokens'] = 0
                    
                    if requests_per_second > 0:
                        cost_per_1k_requests = (instance_price / requests_per_second) * 1000 / 3600
                        record['cost_per_1k_requests'] = cost_per_1k_requests
                    else:
                        record['cost_per_1k_requests'] = 0
                    
                    input_throughput_val = record.get('input_throughput', 0)
                    if input_throughput_val > 0:
                        cost_per_million_input_tokens = (instance_price / input_throughput_val) * 1000000 / 3600
                        record['cost_per_million_input_tokens'] = cost_per_million_input_tokens
                    else:
                        record['cost_per_million_input_tokens'] = 0
                    
                    output_throughput_val = record.get('output_throughput', 0)
                    if output_throughput_val > 0:
                        cost_per_million_output_tokens = (instance_price / output_throughput_val) * 1000000 / 3600
                        record['cost_per_million_output_tokens'] = cost_per_million_output_tokens
                    else:
                        record['cost_per_million_output_tokens'] = 0
                    
                    record['instance_price_used'] = instance_price
                else:
                    record['cost_per_million_tokens'] = 0
                    record['cost_per_1k_requests'] = 0
                    record['cost_per_million_input_tokens'] = 0
                    record['cost_per_million_output_tokens'] = 0
                    record['instance_price_used'] = 0
            
            result.append({
                'combination': combo,
                'data': data
            })
        
        # Generate CSV content
        csv_content = generate_csv_content(result)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"llm-performance-data-{timestamp}.csv"
        
        # Log successful export
        analytics.log_event(http_request, 'server_export_completed', {
            'combinations_count': len(request.combinations),
            'total_data_points': sum(len(item['data']) for item in result),
            'filename': filename
        })
        
        # Return CSV as downloadable response
        from fastapi.responses import Response
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv; charset=utf-8",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


def generate_csv_content(data):
    """Generate CSV content from performance data"""
    csv_lines = []
    
    # Add export metadata
    csv_lines.append(f"Export Date: {datetime.now().isoformat()}")
    csv_lines.append("")
    
    # CSV Headers
    headers = [
        'Runtime',
        'Instance Type', 
        'Model Name',
        'Input Tokens',
        'Output Tokens',
        'Random Tokens',
        'Processes',
        'First Token Latency Mean',
        'First Token Latency P50',
        'First Token Latency P90',
        'End to End Latency Mean',
        'End to End Latency P50',
        'End to End Latency P90',
        'Output Tokens Per Second Mean',
        'Output Tokens Per Second P50',
        'Output Tokens Per Second P90',
        'Success Rate (%)',
        'Requests Per Second',
        'Total Requests',
        'Successful Requests',
        'Failed Requests',
        'Input Throughput (tokens/sec)',
        'Output Throughput (tokens/sec)',
        'Server Throughput (tokens/sec)',
        'Cost Per Million Tokens ($)',
        'Cost Per 1K Requests ($)',
        'Cost Per Million Input Tokens ($)',
        'Cost Per Million Output Tokens ($)',
        'Instance Price Used ($/hour)'
    ]
    
    csv_lines.append(','.join(headers))
    
    # Add data rows
    for item in data:
        combo = item['combination']
        
        # Sort data by processes for consistent ordering
        sorted_data = sorted(item['data'], key=lambda x: x.get('processes', 0))
        
        for record in sorted_data:
            row = [
                combo.get('runtime', ''),
                combo.get('instance_type', ''),
                combo.get('model_name', ''),
                combo.get('input_tokens', 0),
                combo.get('output_tokens', 0),
                combo.get('random_tokens', 0),
                record.get('processes', 0),
                record.get('first_token_latency_mean', 0),
                record.get('first_token_latency_p50', 0),
                record.get('first_token_latency_p90', 0),
                record.get('end_to_end_latency_mean', 0),
                record.get('end_to_end_latency_p50', 0),
                record.get('end_to_end_latency_p90', 0),
                record.get('output_tokens_per_second_mean', 0),
                record.get('output_tokens_per_second_p50', 0),
                record.get('output_tokens_per_second_p90', 0),
                (record.get('success_rate', 0) * 100),
                record.get('requests_per_second', 0),
                record.get('total_requests', 0),
                record.get('successful_requests', 0),
                record.get('failed_requests', 0),
                record.get('input_throughput', 0),
                record.get('output_throughput', 0),
                record.get('server_throughput', 0),
                record.get('cost_per_million_tokens', 0),
                record.get('cost_per_1k_requests', 0),
                record.get('cost_per_million_input_tokens', 0),
                record.get('cost_per_million_output_tokens', 0),
                record.get('instance_price_used', 0)
            ]
            
            # Escape any commas in the data and wrap in quotes if needed
            escaped_row = []
            for value in row:
                string_value = str(value)
                if ',' in string_value or '"' in string_value or '\n' in string_value:
                    escaped_row.append(f'"{string_value.replace('"', '""')}"')
                else:
                    escaped_row.append(string_value)
            
            csv_lines.append(','.join(escaped_row))
    
    return '\n'.join(csv_lines)


def main():
    """Main entry point for the visualization server"""
    print("Starting LLM Performance Visualization Server...")
    print("Access the visualization at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()