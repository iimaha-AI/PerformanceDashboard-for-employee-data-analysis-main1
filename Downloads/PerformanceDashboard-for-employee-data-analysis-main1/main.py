#!/usr/bin/env python3
"""
Employee Events Dashboard - Main Application Entry Point
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Use Flask for compatibility
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import sqlite3
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from datetime import datetime

# Initialize Flask app with custom static folder
app = Flask(__name__, static_folder='assets', static_url_path='/assets')

# Database and model paths
DB_PATH = "python-package/employee_events/employee_events.db"
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
        return df.iloc[0].to_dict() if not df.empty else None
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
    if summary is None:
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

# HTML Templates
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Performance Analytics</title>
    <link rel="stylesheet" href="/assets/css/style.css">
    <link rel="stylesheet" href="/assets/css/bootstrap.min.css">
    <link rel="stylesheet" href="/assets/css/bootstrap-icons.min.css">
    <script src="/assets/js/bootstrap.bundle.min.js"></script>
    <script src="/assets/js/htmx.min.js"></script>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="bi bi-graph-up-arrow me-2"></i>
                Employee Performance Analytics
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="#"><i class="bi bi-bell"></i></a>
                <a class="nav-link" href="#"><i class="bi bi-gear"></i></a>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container" style="margin-top: 100px;">
        <!-- Hero Section -->
        <div class="hero-section">
            <div class="hero-title">
                <i class="bi bi-people-fill me-3"></i>
                Performance Analytics Dashboard
            </div>
            <div class="hero-subtitle"style="color:white;">
                Advanced AI-powered employee performance monitoring and recruitment risk analysis
            </div>
            <div class="pulse-animation">
                <i class="bi bi-cpu-fill" style="font-size: 3rem; color: rgba(255,255,255,0.8);"></i>
            </div>
        </div>

        <!-- Controls Section -->
        <div class="row mb-5">
            <div class="col-lg-6">
                <div class="floating-card">
                    <h4 class="text-gradient mb-4">
                        <i class="bi bi-person-badge me-2"></i>
                        Select Employee
                    </h4>
                    <form>
                        <select name="employee_id" class="form-select mb-3" 
                                hx-get="/employee_dashboard" 
                                hx-target="#dashboard-content" 
                                hx-trigger="change">
                            <option value="">Choose an employee to analyze...</option>
                            {% for emp in employees %}
                            <option value="{{ emp.id }}">{{ emp.name }} ({{ emp.team_name }})</option>
                            {% endfor %}
                        </select>
                    </form>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="floating-card">
                    <h4 class="text-gradient mb-4">
                        <i class="bi bi-diagram-3 me-2"></i>
                        Team Overview
                    </h4>
                    <div class="row">
                        {% for team in teams[:3] %}
                        <div class="col-md-4 mb-3">
                            <div class="team-card">
                                <h6 class="mb-2">{{ team.name }}</h6>
                                <p class="small mb-1">
                                    <i class="bi bi-clock me-1"></i>
                                    {{ team.shift }}
                                </p>
                                <p class="small mb-0">
                                    <i class="bi bi-person-check me-1"></i>
                                    {{ team.manager }}
                                </p>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>

        <!-- Dashboard Content -->
        <div id="dashboard-content">
            <div class="alert alert-gradient text-center">
                <i class="bi bi-person-plus-fill me-2"></i>
                <strong>Ready to Analyze!</strong> Select an employee to view their comprehensive performance dashboard
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <p class="mb-0"style="color:white;">
                        <i class="bi bi-shield-check me-2"></i>
                        Employee Performance Analytics © 2025
                    </p>
                </div>
                <div class="col-md-6 text-end">
                    <p class="mb-0"style="color:white;">
                        <i class="bi bi-cpu me-2"></i>
                        Powered by Advanced AI
                    </p>
                </div>
            </div>
        </div>
    </footer>
</body>
</html>
"""

EMPLOYEE_DASHBOARD_TEMPLATE = """
{% if not employee_id %}
    <div class="alert alert-gradient">Please select an employee</div>
{% elif summary is none %}
    <div class="alert alert-gradient">Employee not found</div>
{% else %}
    <!-- Employee Header -->
    <div class="floating-card mb-4">
        <div class="row align-items-center">
            <div class="col-lg-8">
                <div class="d-flex align-items-center mb-3">
                    <div class="avatar-gradient me-3">
                        <i class="bi bi-person-fill" style="font-size: 2rem;"></i>
                    </div>
                    <div>
                        <h2 class="mb-1 text-gradient">{{ summary.name }}</h2>
                        <p class="text-light mb-0">
                            <i class="bi bi-building me-2"></i>{{ summary.team_name }} | 
                            <i class="bi bi-clock me-2"></i>{{ summary.shift }} | 
                            <i class="bi bi-person-check me-2"></i>{{ summary.manager }}
                        </p>
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="risk-assessment text-center">
                    <h5 class="mb-3" style="color:white;">
                        <i class="bi bi-shield-exclamation me-2"></i>
                        Recruitment Risk Analysis
                    </h5>
                    <div class="risk-badge-{{ risk_color }} mb-2">
                        {{ risk_level }}
                    </div>
                    <p class="small text-light">
                        <i class="bi bi-graph-up me-1"></i>
                        {{ "%.1f%%" % (risk_prob * 100) if risk_prob else "N/A" }} Probability
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Performance Metrics -->
    <div class="row g-4 mb-5">
        <div class="col-lg-3 col-md-6">
            <div class="metric-card metric-success">
                <div class="metric-icon">
                    <i class="bi bi-arrow-up-circle-fill"></i>
                </div>
                <div class="metric-content">
                    <h3 class="metric-number">{{ "%.0f" % summary.total_positive }}</h3>
                    <p class="metric-label">Total Positive Events</p>
                </div>
            </div>
        </div>
        <div class="col-lg-3 col-md-6">
            <div class="metric-card metric-danger">
                <div class="metric-icon">
                    <i class="bi bi-arrow-down-circle-fill"></i>
                </div>
                <div class="metric-content">
                    <h3 class="metric-number">{{ "%.0f" % summary.total_negative }}</h3>
                    <p class="metric-label">Total Negative Events</p>
                </div>
            </div>
        </div>
        <div class="col-lg-3 col-md-6">
            <div class="metric-card metric-info">
                <div class="metric-icon">
                    <i class="bi bi-graph-up"></i>
                </div>
                <div class="metric-content">
                    <h3 class="metric-number">{{ "%.1f" % summary.avg_positive }}</h3>
                    <p class="metric-label">Avg Positive/Day</p>
                </div>
            </div>
        </div>
        <div class="col-lg-3 col-md-6">
            <div class="metric-card metric-warning">
                <div class="metric-icon">
                    <i class="bi bi-graph-down"></i>
                </div>
                <div class="metric-content">
                    <h3 class="metric-number">{{ "%.1f" % summary.avg_negative }}</h3>
                    <p class="metric-label">Avg Negative/Day</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Performance Chart -->
    <div class="floating-card mb-4">
        <div class="chart-header">
            <h4 class="text-gradient mb-0">
                <i class="bi bi-graph-up-arrow me-2"></i>
                Performance Trends Analysis
            </h4>
        </div>
        <div class="chart-container">
            {% if chart_img %}
                <img src="data:image/png;base64,{{ chart_img }}" class="img-fluid rounded">
            {% else %}
                <div class="no-data-message">
                    <i class="bi bi-bar-chart-line" style="font-size: 3rem; opacity: 0.5;"></i>
                    <p>No chart data available</p>
                </div>
            {% endif %}
        </div>
    </div>

    <!-- Notes Section -->
    <div class="floating-card">
        <div class="notes-header">
            <h4 class="text-gradient mb-4">
                <i class="bi bi-journal-text me-2"></i>
                Manager Notes & Comments
            </h4>
        </div>
        <div class="notes-content">
            {% if notes %}
                {% for note in notes[:5] %}
                <div class="note-item">
                    <div class="note-content">
                        <p class="mb-2">{{ note.note }}</p>
                        <small class="note-timestamp">
                            <i class="bi bi-calendar-event me-1"></i>
                            Added: {{ note.created_at }}
                        </small>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="no-notes-message">
                    <i class="bi bi-journal-plus" style="font-size: 2rem; opacity: 0.5;"></i>
                    <p>No notes available for this employee</p>
                </div>
            {% endif %}
        </div>
    </div>
{% endif %}
"""

# Routes
@app.route('/')
def home():
    """Main dashboard page."""
    employees = get_employees()
    teams = get_teams()
    return render_template_string(MAIN_TEMPLATE, employees=employees, teams=teams)

@app.route('/employee_dashboard')
def employee_dashboard():
    """Load employee dashboard content."""
    employee_id = request.args.get('employee_id')
    
    if not employee_id:
        return render_template_string(EMPLOYEE_DASHBOARD_TEMPLATE, employee_id=None)
    
    try:
        employee_id = int(employee_id)
    except ValueError:
        return render_template_string(EMPLOYEE_DASHBOARD_TEMPLATE, employee_id=None)
    
    # Get employee data
    summary = get_employee_summary(employee_id)
    if summary is None:
        return render_template_string(EMPLOYEE_DASHBOARD_TEMPLATE, employee_id=employee_id, summary=None)
    
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
    
    return render_template_string(EMPLOYEE_DASHBOARD_TEMPLATE, 
                                  employee_id=employee_id, 
                                  summary=summary,
                                  notes=notes,
                                  risk_prob=risk_prob,
                                  chart_img=chart_img,
                                  risk_level=risk_level,
                                  risk_color=risk_color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

# For gunicorn compatibility
application = app