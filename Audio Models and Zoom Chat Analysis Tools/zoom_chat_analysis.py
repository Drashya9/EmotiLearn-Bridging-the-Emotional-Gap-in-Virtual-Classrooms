import re
import pandas as pd
from datetime import datetime, timedelta

def parse_zoom_chat(file_path):
    chat_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\[(\d+:\d+:\d+)\] (.*?): (.*)", line)
            if match:
                time_str, user, message = match.groups()
                time_obj = datetime.strptime(time_str, "%H:%M:%S")
                chat_data.append({
                    "timestamp": time_obj,
                    "user": user.strip(),
                    "message": message.strip()
                })
    return pd.DataFrame(chat_data)

def aggregate_stats(df, window_seconds=30):
    if df.empty:
        return pd.DataFrame()

    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    delta = timedelta(seconds=window_seconds)

    stats = []

    while start_time <= end_time:
        window_end = start_time + delta
        window_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] < window_end)]

        questions = window_df[window_df['message'].str.endswith("?")]
        question_askers = questions['user'].nunique()
        responses = window_df[~window_df['message'].str.endswith("?")]

        stats.append({
            "time_window_start": start_time.time().strftime("%H:%M:%S"),
            "time_window_end": window_end.time().strftime("%H:%M:%S"),
            "total_messages": len(window_df),
            "unique_users": window_df['user'].nunique(),
            "questions_asked": len(questions),
            "question_askers": question_askers,
            "responses_after_questions": len(responses),
            "responding_users": responses['user'].nunique()
        })

        start_time = window_end

    return pd.DataFrame(stats)

if __name__ == "__main__":
    chat_file = "sample_zoom_chat.txt"  # Change this to your file path
    df_chat = parse_zoom_chat(chat_file)
    df_stats = aggregate_stats(df_chat, window_seconds=30)
    
    output_file = "chat_engagement_stats.csv"
    df_stats.to_csv(output_file, index=False)
    print(f"Engagement stats saved to: {output_file}")
