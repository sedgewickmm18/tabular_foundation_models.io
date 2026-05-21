import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import matplotlib.font_manager as fm
import numpy as np
import os

# Create assets/images directory if it doesn't exist
os.makedirs('assets/images', exist_ok=True)

# Find and set Segoe UI Emoji font explicitly for emoji support
emoji_font = None
for font in fm.fontManager.ttflist:
    if 'Segoe UI Emoji' in font.name or 'Segoe UI Symbol' in font.name:
        emoji_font = fm.FontProperties(fname=font.fname)
        break

# If Segoe UI Emoji not found, try other emoji fonts
if emoji_font is None:
    for font in fm.fontManager.ttflist:
        if 'Emoji' in font.name or 'Symbol' in font.name:
            emoji_font = fm.FontProperties(fname=font.fname)
            break

# Set default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']

def generate_workflow_ai_agent_diagram():
    """Generate a circular/radial diagram showing 5 workflow and AI agent use cases"""
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6, 6)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Define colors
    hub_color = '#4A148C'  # Deep Purple
    chat_box_color = '#E8E8E8'  # Light gray for chat inbox
    default_box_color = 'white'  # White for other boxes
    
    # Central Hub
    hub_radius = 1.2
    hub_circle = Circle((0, 0), hub_radius, color=hub_color, ec='#311B92', linewidth=3, zorder=10)
    ax.add_patch(hub_circle)
    
    # Hub label with line breaks
    ax.text(0, 0.15, 'AI Agent +', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white', zorder=11)
    ax.text(0, -0.15, 'Workflow', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white', zorder=11)
    ax.text(0, -0.55, 'Integration', ha='center', va='center', 
            fontsize=11, fontweight='normal', color='white', zorder=11, style='italic')
    
    # Define use cases with positions (angle in degrees)
    # Pentagram layout: 5 points at 72° intervals, starting from top (90°)
    use_cases = [
        {
            'angle': 90,  # Top
            'title': 'Chat with Task Inbox',
            'description': 'Search & handle tasks\nthrough conversation',
            'icon': '💬',
            'icon_fallback': '◉',
            'box_color': chat_box_color,
            'key': 'chat'
        },
        {
            'angle': 18,  # Upper right (90 - 72)
            'title': 'Agent Tasks',
            'description': 'Consult AI agent\nfrom workflow',
            'icon': '🤖',
            'icon_fallback': '◈',
            'box_color': default_box_color,
            'key': 'consult'
        },
        {
            'angle': 306,  # Lower right (18 - 72)
            'title': 'Agent Starts Workflow',
            'description': 'AI initiates & triggers\nworkflow execution',
            'icon': '▶️',
            'icon_fallback': '▶',
            'box_color': default_box_color,
            'key': 'trigger'
        },
        {
            'angle': 234,  # Lower left (306 - 72)
            'title': 'Create BPMN Template',
            'description': 'Generate workflow\nfrom chat',
            'icon': '📋',
            'icon_fallback': '◫',
            'box_color': default_box_color,
            'key': 'create'
        },
        {
            'angle': 162,  # Upper left (234 - 72)
            'title': 'Dynamic Exception Handling',
            'description': 'Handle errors & search\nmissing data automatically',
            'icon': '⚡',
            'icon_fallback': '⚡',
            'box_color': default_box_color,
            'key': 'exception'
        }
    ]
    
    # Distance from center for use case nodes
    distance = 4.0
    box_width = 3.5  # Wider boxes for better text fit
    box_height = 1.2  # Less high since we're removing icons
    
    # Database box configuration (for external data sources)
    db_distance = 6.5  # Further out from the exception handling box
    db_angle = 162  # Same angle as Dynamic Exception Handling
    db_width = 2.0
    db_height = 1.5
    
    # Draw use cases and arrows
    for uc in use_cases:
        angle_rad = np.radians(uc['angle'])
        
        # Calculate position
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        
        # Draw use case box
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=uc['box_color'],
            edgecolor='#212121',
            linewidth=2.5,
            zorder=5
        )
        ax.add_patch(box)
        
        # Determine text color based on background
        text_color = '#212121' if uc['box_color'] != hub_color else 'white'
        
        # Icons removed - not displaying properly
        
        # Add title (centered vertically since no icon)
        ax.text(x, y + 0.2, uc['title'], ha='center', va='center',
                fontsize=11, fontweight='bold', color=text_color, zorder=6)
        
        # Add description
        ax.text(x, y - 0.25, uc['description'], ha='center', va='center',
                fontsize=8.5, color=text_color, zorder=6, style='italic',
                linespacing=1.3)
        
        # Add greenish ellipse with "partially covered" text for Chat with Task Inbox
        if uc['key'] == 'chat':
            ellipse = Ellipse((x + 1.6, y + 0.7), width=1.2, height=0.5,
                            facecolor='#C8E6C9', edgecolor='#4CAF50',
                            linewidth=2, alpha=0.7, zorder=6)
            ax.add_patch(ellipse)
            ax.text(x + 1.6, y + 0.7, 'partially\ncovered', ha='center', va='center',
                    fontsize=8, color='#2E7D32', fontweight='bold', zorder=7,
                    linespacing=1.1)
        
        # Calculate arrow start and end points
        # From hub edge to use case box edge
        hub_edge_x = hub_radius * np.cos(angle_rad)
        hub_edge_y = hub_radius * np.sin(angle_rad)
        
        # Calculate box edge point (toward center) - improved for pentagram angles
        # Use angle to determine the proper edge point
        box_edge_x = x - (box_width/2) * np.cos(angle_rad)
        box_edge_y = y - (box_height/2) * np.sin(angle_rad)
        
        # Draw bidirectional arrows with slight offset for visual clarity
        # Use gray color for arrows instead of use case colors
        arrow_color = '#666666'
        
        # Calculate perpendicular offset for bidirectional arrows
        offset = 0.15
        perp_angle = angle_rad + np.pi/2
        offset_x = offset * np.cos(perp_angle)
        offset_y = offset * np.sin(perp_angle)
        
        # Arrow 1: Hub to Use Case (solid line)
        arrow1 = FancyArrowPatch(
            (hub_edge_x + offset_x, hub_edge_y + offset_y),
            (box_edge_x + offset_x, box_edge_y + offset_y),
            arrowstyle='->', mutation_scale=25, linewidth=2.5,
            color=arrow_color, zorder=4, alpha=0.8
        )
        ax.add_patch(arrow1)
        
        # Arrow 2: Use Case to Hub (dashed line)
        arrow2 = FancyArrowPatch(
            (box_edge_x - offset_x, box_edge_y - offset_y),
            (hub_edge_x - offset_x, hub_edge_y - offset_y),
            arrowstyle='->', mutation_scale=25, linewidth=2,
            color=arrow_color, linestyle='--', zorder=4, alpha=0.6
        )
        ax.add_patch(arrow2)
    
    # Draw pentagon connecting the 5 use cases
    # Connect each use case to the next one in sequence
    pentagon_color = '#9E9E9E'  # Gray color for pentagon edges
    for i, uc in enumerate(use_cases):
        next_uc = use_cases[(i + 1) % len(use_cases)]
        
        # Calculate positions
        x1 = distance * np.cos(np.radians(uc['angle']))
        y1 = distance * np.sin(np.radians(uc['angle']))
        x2 = distance * np.cos(np.radians(next_uc['angle']))
        y2 = distance * np.sin(np.radians(next_uc['angle']))
        
        # Calculate edge points on boxes
        angle_between = np.arctan2(y2 - y1, x2 - x1)
        edge1_x = x1 + (box_width/2) * np.cos(angle_between)
        edge1_y = y1 + (box_height/2) * np.sin(angle_between)
        edge2_x = x2 - (box_width/2) * np.cos(angle_between)
        edge2_y = y2 - (box_height/2) * np.sin(angle_between)
        
        # Draw pentagon edge
        pentagon_arrow = FancyArrowPatch(
            (edge1_x, edge1_y),
            (edge2_x, edge2_y),
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color=pentagon_color, linestyle=':', zorder=3, alpha=0.5
        )
        ax.add_patch(pentagon_arrow)
    
    # Draw database box for external data sources
    db_x = db_distance * np.cos(np.radians(db_angle))
    db_y = db_distance * np.sin(np.radians(db_angle))
    
    # Create database cylinder shape (simplified as rounded rectangle with lines)
    db_box = FancyBboxPatch(
        (db_x - db_width/2, db_y - db_height/2),
        db_width, db_height,
        boxstyle="round,pad=0.1",
        facecolor='#E3F2FD',  # Light blue for database
        edgecolor='#1976D2',
        linewidth=2.5,
        zorder=5
    )
    ax.add_patch(db_box)
    
    # Add horizontal lines to make it look more like a database
    line_y1 = db_y + db_height/2 - 0.3
    line_y2 = db_y - db_height/2 + 0.3
    ax.plot([db_x - db_width/2 + 0.2, db_x + db_width/2 - 0.2], [line_y1, line_y1],
            color='#1976D2', linewidth=2, zorder=6)
    ax.plot([db_x - db_width/2 + 0.2, db_x + db_width/2 - 0.2], [line_y2, line_y2],
            color='#1976D2', linewidth=2, zorder=6)
    
    # Add database label
    ax.text(db_x, db_y + 0.25, 'External', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1565C0', zorder=6)
    ax.text(db_x, db_y, 'Data', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1565C0', zorder=6)
    ax.text(db_x, db_y - 0.25, 'Sources', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1565C0', zorder=6)
    
    # Connect database to Dynamic Exception Handling use case
    exception_uc_x = distance * np.cos(np.radians(162))
    exception_uc_y = distance * np.sin(np.radians(162))
    
    # Calculate connection points
    db_edge_x = db_x + (db_width/2) * np.cos(np.radians(162 + 180))
    db_edge_y = db_y + (db_height/2) * np.sin(np.radians(162 + 180))
    exception_edge_x = exception_uc_x + (box_width/2) * np.cos(np.radians(162))
    exception_edge_y = exception_uc_y + (box_height/2) * np.sin(np.radians(162))
    
    # Draw arrow from database to exception handling
    db_arrow = FancyArrowPatch(
        (db_edge_x, db_edge_y),
        (exception_edge_x, exception_edge_y),
        arrowstyle='->', mutation_scale=25, linewidth=2.5,
        color='#1976D2', zorder=4, linestyle='--', alpha=0.7
    )
    ax.add_patch(db_arrow)
    
    # Add label for database connection
    mid_x = (db_edge_x + exception_edge_x) / 2
    mid_y = (db_edge_y + exception_edge_y) / 2
    ax.text(mid_x - 0.3, mid_y + 0.3, 'queries', ha='center', va='center',
            fontsize=8, style='italic', color='#1976D2',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8),
            zorder=7)
    
    # Add title
    ax.text(0, 5.7, 'Workflow & AI Agent Integration Use Cases',
            ha='center', fontsize=16, fontweight='bold', color='#212121')
    
    # Add subtitle
    ax.text(0, 5.3, 'Bidirectional interactions between AI agents and workflow systems', 
            ha='center', fontsize=11, style='italic', color='#666666')
    
    # Add legend for arrow types in upper right quadrant
    legend_x = 3.5
    legend_y = 4.5
    ax.text(legend_x, legend_y + 0.5, 'Legend:', fontsize=10, fontweight='bold', color='#212121')
    
    # Solid arrow
    arrow_legend_1 = FancyArrowPatch(
        (legend_x, legend_y), (legend_x + 1, legend_y),
        arrowstyle='->', mutation_scale=20, linewidth=2.5,
        color='#666666', zorder=4
    )
    ax.add_patch(arrow_legend_1)
    ax.text(legend_x + 1.2, legend_y, 'Outbound (Hub → Use Case)',
            ha='left', va='center', fontsize=9, color='#212121')
    
    # Dashed arrow
    arrow_legend_2 = FancyArrowPatch(
        (legend_x, legend_y - 0.5), (legend_x + 1, legend_y - 0.5),
        arrowstyle='->', mutation_scale=20, linewidth=2,
        color='#666666', linestyle='--', zorder=4
    )
    ax.add_patch(arrow_legend_2)
    ax.text(legend_x + 1.2, legend_y - 0.5, 'Inbound (Use Case → Hub)',
            ha='left', va='center', fontsize=9, color='#212121')
    
    # Add footer note with bold text
    ax.text(0, -5.9, 'Each use case demonstrates seamless integration between conversational AI and workflow automation',
            ha='center', fontsize=8, fontweight='bold', color='#666666')
    
    # Save as SVG
    plt.savefig('assets/images/workflow_ai_agent_use_cases.svg',
                format='svg', bbox_inches='tight', dpi=150,
                facecolor='white', edgecolor='none')
    print("✓ Saved: assets/images/workflow_ai_agent_use_cases.svg")
    
    # Also save as PNG for broader compatibility
    plt.savefig('assets/images/workflow_ai_agent_use_cases.png',
                format='png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    print("✓ Saved: assets/images/workflow_ai_agent_use_cases.png")
    
    plt.close()

if __name__ == '__main__':
    generate_workflow_ai_agent_diagram()
    print("\n✓ Workflow & AI Agent diagram generated successfully!")
    print("  - SVG format for scalability")
    print("  - PNG format for compatibility")
    print("  - Circular/radial layout with bidirectional interactions")
    print("  - 4 use cases clearly visualized around central hub")

# Made with Bob
