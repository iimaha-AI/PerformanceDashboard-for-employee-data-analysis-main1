#!/usr/bin/env python3
"""
Employee Events Dashboard - FastHTML Application
A comprehensive dashboard for monitoring employee performance and recruitment risk.
"""

import sqlite3
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import io
import base64
from typing import Optional, Dict, List, Any

from fasthtml.common import *
from fasthtml import FastHTML

# Initialize FastHTML app
app = FastHTML(
    hdrs=[
        Link(rel="stylesheet", href="/assets/css/style.css"),
        Link(rel="stylesheet", href="/assets/css/bootstrap-icons.min.css"),
        Script(src="/assets/js/bootstrap.bundle.min.js"),
        Script(src="/assets/js/htmx.min.js")
    ],
    static_path="assets"
)

# Database and model paths
DB_PATH = Path("employee_events/employee_events.db")
MODEL_PATH = Path("assets/model.pkl")

# Load ML model
def load_model():
    """Load the trained logistic regression model."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Database helper functions
def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

def get_employees():
    """Get all employees from database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT e.id, e.name, t.name as team_name, t.shift, t.manager
            FROM employee e
            JOIN team t ON e.team_id = t.id
            ORDER BY e.name
        """, conn)
        return df.to_dict('records')
    finally:
        conn.close()

def get_teams():
    """Get all teams from database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM team ORDER BY name", conn)
        return df.to_dict('records')
    finally:
        conn.close()

def get_employee_events(employee_id: int, days: int = 30):
    """Get employee events for the last N days."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT 
                event_date,
                positive_events,
                negative_events,
                (positive_events - negative_events) as net_events
            FROM employee_events
            WHERE employee_id = ? 
            AND event_date >= date('now', '-{} days')
            ORDER BY event_date
        """.format(days), conn, params=(employee_id,))
        return df
    finally:
        conn.close()

def get_employee_summary(employee_id: int):
    """Get employee summary statistics."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT 
                e.name,
                t.name as team_name,
                t.shift,
                t.manager,
                SUM(ev.positive_events) as total_positive,
                SUM(ev.negative_events) as total_negative,
                AVG(ev.positive_events) as avg_positive,
                AVG(ev.negative_events) as avg_negative
            FROM employee e
            JOIN team t ON e.team_id = t.id
            JOIN employee_events ev ON e.id = ev.employee_id
            WHERE e.id = ?
            GROUP BY e.id, e.name, t.name, t.shift, t.manager
        """, conn, params=(employee_id,))
        return df.iloc[0] if not df.empty else None
    finally:
        conn.close()

def get_employee_notes(employee_id: int):
    """Get employee notes."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT note, created_at
            FROM notes
            WHERE employee_id = ?
            ORDER BY created_at DESC
        """, conn, params=(employee_id,))
        return df.to_dict('records')
    finally:
        conn.close()

def predict_recruitment_risk(employee_id: int):
    """Predict recruitment risk for an employee."""
    model = load_model()
    if not model:
        return None
    
    summary = get_employee_summary(employee_id)
    if not summary:
        return None
    
    # Prepare features as a DataFrame to preserve feature names
    features = pd.DataFrame({
        'total_positive': [summary['total_positive']],
        'total_negative': [summary['total_negative']]
    })
    
    # Get prediction probability
    risk_prob = model.predict_proba(features)[0][1]
    return risk_prob

def create_performance_chart(employee_id: int, days: int = 30):
    """Create a performance chart for an employee."""
    events_df = get_employee_events(employee_id, days)
    
    if events_df.empty:
        return None
    
    # Create matplotlib figure with elegant styling
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Convert date strings to datetime
    events_df['event_date'] = pd.to_datetime(events_df['event_date'])
    
    # Define gradient colors
    positive_color = '#4facfe'  # Blue gradient
    negative_color = '#fa709a'  # Pink gradient
    net_color = '#43e97b'       # Green gradient
    
    # Plot with enhanced styling
    ax.plot(events_df['event_date'], events_df['positive_events'], 
            color=positive_color, marker='o', label='Positive Events', 
            linewidth=3, markersize=8, markerfacecolor='white', 
            markeredgecolor=positive_color, markeredgewidth=2)
    ax.plot(events_df['event_date'], events_df['negative_events'], 
            color=negative_color, marker='s', label='Negative Events', 
            linewidth=3, markersize=8, markerfacecolor='white', 
            markeredgecolor=negative_color, markeredgewidth=2)
    ax.plot(events_df['event_date'], events_df['net_events'], 
            color=net_color, marker='^', label='Net Events', 
            linewidth=3, markersize=8, markerfacecolor='white', 
            markeredgecolor=net_color, markeredgewidth=2)
    
    # Enhanced styling
    ax.set_xlabel('Date', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Events', color='white', fontsize=14, fontweight='bold')
    ax.set_title(f'Performance Analytics - Last {days} Days', 
                 color='white', fontsize=18, fontweight='bold', pad=20)
    
    # Style the legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.9, fontsize=12)
    legend.get_frame().set_facecolor('#2a2a2a')
    legend.get_frame().set_edgecolor('white')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Style the axes
    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    # Set background with gradient effect
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Add subtle gradient background
    ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], 
               alpha=0.1, color='white', zorder=0)
    
    # Convert to base64 string with higher DPI for better quality
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', 
                facecolor='#1a1a1a', dpi=150, edgecolor='none')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_str

# Route handlers
@app.get("/")
def home():
    """Main dashboard page."""
    employees = get_employees()
    teams = get_teams()
    
    return Html(
        Head(
            Title("Employee Performance Analytics"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1")
        ),
        Body(
            # Navigation
            Nav(
                Div(
                    A(I(cls="bi bi-graph-up-arrow me-2"), "Employee Performance Analytics", href="/", 
                      cls="navbar-brand"),
                    Div(
                        A(I(cls="bi bi-bell"), href="#", cls="nav-link"),
                        A(I(cls="bi bi-gear"), href="#", cls="nav-link"),
                        cls="navbar-nav ms-auto"
                    ),
                    cls="container"
                ),
                cls="navbar navbar-expand-lg navbar-dark fixed-top"
            ),
            
            # Main content
            Div(
                # Hero Section
                Div(
                    Div(
                        I(cls="bi bi-people-fill me-3"),
                        "Performance Analytics Dashboard",
                        cls="hero-title"
                    ),
                    Div(
                        "Advanced AI-powered employee performance monitoring and recruitment risk analysis",
                        cls="hero-subtitle"
                    ),
                    Div(
                        I(cls="bi bi-cpu-fill", style="font-size: 3rem; color: rgba(255,255,255,0.8);"),
                        cls="pulse-animation"
                    ),
                    cls="hero-section"
                ),
                
                # Controls Section
                Div(
                    Div(
                        H4(I(cls="bi bi-person-badge me-2"), "Select Employee", cls="text-gradient mb-4"),
                        Form(
                            Select(
                                Option("Choose an employee to analyze...", value="", selected=True),
                                *[Option(f"{emp['name']} ({emp['team_name']})", value=str(emp['id']))
                                  for emp in employees],
                                name="employee_id",
                                cls="form-select mb-3",
                                hx_post="/employee_dashboard",
                                hx_target="#dashboard-content",
                                hx_trigger="change"
                            ),
                            cls="mb-4"
                        ),
                        cls="col-lg-6 floating-card"
                    ),
                    Div(
                        H4(I(cls="bi bi-diagram-3 me-2"), "Team Overview", cls="text-gradient mb-4"),
                        Div(
                            *[Div(
                                H6(team['name'], cls="mb-2"),
                                P(I(cls="bi bi-clock me-1"), f"{team['shift']}", cls="small mb-1"),
                                P(I(cls="bi bi-person-check me-1"), f"{team['manager']}", cls="small mb-0"),
                                cls="team-card"
                            ) for team in teams[:3]],
                            cls="row"
                        ),
                        cls="col-lg-6 floating-card"
                    ),
                    cls="row mb-5"
                ),
                
                # Dashboard content (will be populated by HTMX)
                Div(
                    Div(
                        I(cls="bi bi-person-plus-fill me-2"),
                        Strong("Ready to Analyze!"), " Select an employee to view their comprehensive performance dashboard",
                        cls="alert alert-gradient text-center"
                    ),
                    id="dashboard-content"
                ),
                
                cls="container", style="margin-top: 100px;"
            ),
            
            # Footer
            Footer(
                Div(
                    Div(
                        P(I(cls="bi bi-shield-check me-2"), "Employee Performance Analytics © 2025", cls="mb-0"),
                        cls="col-md-6"
                    ),
                    Div(
                        P(I(cls="bi bi-cpu me-2"), "Powered by Advanced AI", cls="mb-0"),
                        cls="col-md-6 text-end"
                    ),
                    cls="row"
                ),
                cls="footer"
            )
        )
    )

@app.post("/employee_dashboard")
def employee_dashboard(employee_id: int):
    """Load employee dashboard content."""
    if not employee_id:
        return Div("Please select an employee", cls="alert alert-gradient")
    
    # Get employee data
    summary = get_employee_summary(employee_id)
    if not summary:
        return Div("Employee not found", cls="alert alert-gradient")
    
    notes = get_employee_notes(employee_id)
    risk_prob = predict_recruitment_risk(employee_id)
    chart_img = create_performance_chart(employee_id)
    
    # Risk assessment
    risk_level = "Low"
    risk_color = "success"
    if risk_prob:
        if risk_prob > 0.7:
            risk_level = "High"
            risk_color = "danger"
        elif risk_prob > 0.4:
            risk_level = "Medium"
            risk_color = "warning"
    
    return Div(
        # Employee header
        Div(
            Div(
                Div(
                    Div(
                        I(cls="bi bi-person-fill", style="font-size: 2rem;"),
                        cls="avatar-gradient me-3"
                    ),
                    Div(
                        H2(f"{summary['name']}", cls="mb-1 text-gradient"),
                        P(
                            I(cls="bi bi-building me-2"), f"{summary['team_name']} | ",
                            I(cls="bi bi-clock me-2"), f"{summary['shift']} | ",
                            I(cls="bi bi-person-check me-2"), f"{summary['manager']}",
                            cls="text-light mb-0"
                        )
                    ),
                    cls="d-flex align-items-center mb-3"
                ),
                cls="col-lg-8"
            ),
            Div(
                Div(
                    H5(I(cls="bi bi-shield-exclamation me-2"), "Recruitment Risk Analysis", cls="mb-3"),
                    Div(f"{risk_level}", cls=f"risk-badge-{risk_color} mb-2"),
                    P(I(cls="bi bi-graph-up me-1"), f"{risk_prob:.1%}" if risk_prob else "N/A", cls="small text-light"),
                    cls="risk-assessment text-center"
                ),
                cls="col-lg-4"
            ),
            cls="row align-items-center mb-4 floating-card"
        ),
        
        # Performance metrics
        Div(
            Div(
                Div(
                    I(cls="bi bi-arrow-up-circle-fill"),
                    H3(f"{int(summary['total_positive'])}", cls="metric-number"),
                    P("Total Positive Events", cls="metric-label"),
                    cls="metric-content"
                ),
                cls="col-lg-3 col-md-6 metric-card metric-success"
            ),
            Div(
                Div(
                    I(cls="bi bi-arrow-down-circle-fill"),
                    H3(f"{int(summary['total_negative'])}", cls="metric-number"),
                    P("Total Negative Events", cls="metric-label"),
                    cls="metric-content"
                ),
                cls="col-lg-3 col-md-6 metric-card metric-danger"
            ),
            Div(
                Div(
                    I(cls="bi bi-graph-up"),
                    H3(f"{summary['avg_positive']:.1f}", cls="metric-number"),
                    P("Avg Positive/Day", cls="metric-label"),
                    cls="metric-content"
                ),
                cls="col-lg-3 col-md-6 metric-card metric-info"
            ),
            Div(
                Div(
                    I(cls="bi bi-graph-down"),
                    H3(f"{summary['avg_negative']:.1f}", cls="metric-number"),
                    P("Avg Negative/Day", cls="metric-label"),
                    cls="metric-content"
                ),
                cls="col-lg-3 col-md-6 metric-card metric-warning"
            ),
            cls="row g-4 mb-5"
        ),
        
        # Performance chart
        Div(
            Div(
                H4(I(cls="bi bi-graph-up-arrow me-2"), "Performance Trends Analysis", cls="text-gradient mb-0"),
                cls="chart-header"
            ),
            Div(
                Img(src=f"data:image/png;base64,{chart_img}", cls="img-fluid rounded") if chart_img else 
                Div(
                    I(cls="bi bi-bar-chart-line", style="font-size: 3rem; opacity: 0.5;"),
                    P("No chart data available"),
                    cls="no-data-message"
                ),
                cls="chart-container"
            ),
            cls="floating-card mb-4"
        ),
        
        # Notes section
        Div(
            Div(
                H4(I(cls="bi bi-journal-text me-2"), "Manager Notes & Comments", cls="text-gradient mb-4"),
                cls="notes-header"
            ),
            Div(
                *[Div(
                    Div(
                        P(note['note'], cls="mb-2"),
                        Small(I(cls="bi bi-calendar-event me-1"), f"Added: {note['created_at']}", cls="note-timestamp"),
                        cls="note-content"
                    ),
                    cls="note-item"
                ) for note in notes[:5]] if notes else 
                Div(
                    I(cls="bi bi-journal-plus", style="font-size: 2rem; opacity: 0.5;"),
                    P("No notes available for this employee"),
                    cls="no-notes-message"
                ),
                cls="notes-content"
            ),
            cls="floating-card"
        )
    )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)