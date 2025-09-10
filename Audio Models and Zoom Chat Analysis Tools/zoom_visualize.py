import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def visualize_engagement(csv_path):
    df = pd.read_csv(csv_path)

    # Create a time label for x-axis
    df['time_label'] = df['time_window_start'] + " - " + df['time_window_end']

    # Plot engagement stats
    plt.figure(figsize=(14, 6))
    plt.plot(df['time_label'], df['total_messages'], label='Total Messages', marker='o')
    plt.plot(df['time_label'], df['unique_users'], label='Unique Users', marker='s')
    plt.plot(df['time_label'], df['questions_asked'], label='Questions Asked', marker='^')
    plt.plot(df['time_label'], df['responses_after_questions'], label='Responses', marker='x')

    plt.xticks(rotation=45)
    plt.xlabel("Time Window")
    plt.ylabel("Count")
    plt.title("Zoom Chat Engagement Over Time")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    plt.savefig("chat_engagement_timeline.png")
    plt.show()

def compute_engagement_score(df):
    # Weighted score: tune the weights as needed
    df['engagement_score'] = (
        0.4 * df['total_messages'] +
        0.3 * df['unique_users'] +
        0.2 * df['questions_asked'] +
        0.1 * df['responses_after_questions']
    )
    return df

def label_engagement_level(score):
    if score >= 7:
        return 'High'
    elif score >= 4:
        return 'Medium'
    else:
        return 'Low'

def visualize_engagement_plotly(csv_path):
    df = pd.read_csv(csv_path)
    df['time_label'] = df['time_window_start'] + " - " + df['time_window_end']
    df = compute_engagement_score(df)
    df['engagement_level'] = df['engagement_score'].apply(label_engagement_level)

    # Color map for engagement level
    level_color_map = {"High": "green", "Medium": "orange", "Low": "red"}

    fig = go.Figure()

    # Add lines for core metrics
    fig.add_trace(go.Scatter(x=df['time_label'], y=df['total_messages'],
                             mode='lines+markers', name='Total Messages'))
    fig.add_trace(go.Scatter(x=df['time_label'], y=df['unique_users'],
                             mode='lines+markers', name='Unique Users'))
    fig.add_trace(go.Scatter(x=df['time_label'], y=df['questions_asked'],
                             mode='lines+markers', name='Questions Asked'))
    fig.add_trace(go.Scatter(x=df['time_label'], y=df['responses_after_questions'],
                             mode='lines+markers', name='Responses'))

    # Add engagement score with color-coded points
    fig.add_trace(go.Scatter(
        x=df['time_label'],
        y=df['engagement_score'],
        mode='markers+lines',
        name='Engagement Score',
        marker=dict(
            size=12,
            color=[level_color_map[level] for level in df['engagement_level']],
            line=dict(width=1, color='DarkSlateGrey')
        )
    ))

    fig.update_layout(
        title="📊 Zoom Chat Engagement Timeline (Interactive)",
        xaxis_title="Time Window",
        yaxis_title="Count / Score",
        legend_title="Metrics",
        xaxis_tickangle=-45,
        template="plotly_white",
        height=600
    )

    fig.show()

if __name__ == "__main__":
    visualize_engagement_plotly("chat_engagement_stats.csv")
    csv_path = "chat_engagement_stats.csv"  # Make sure this file exists
    visualize_engagement(csv_path)
    print(f"Engagement visualization saved as 'chat_engagement_timeline.png'")
