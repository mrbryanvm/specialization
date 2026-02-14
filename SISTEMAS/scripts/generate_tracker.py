import os
import datetime
import calendar

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # SISTEMAS/
TEMPLATES_DIR = os.path.join(BASE_DIR, "scripts", "templates")
TRACKING_DIR = os.path.join(BASE_DIR, "TRACKING")

def get_date_input(prompt_text="Enter date (YYYY-MM-DD) or press Enter for today: "):
    date_str = input(prompt_text).strip().lower()
    if not date_str:
        return datetime.date.today()
    if date_str in ["y", "yesterday", "ayer"]:
        return datetime.date.today() - datetime.timedelta(days=1)
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid format. Using today's date.")
        return datetime.date.today()

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_daily():
    date = get_date_input()
    year = str(date.year)
    month_name = f"{date.month:02d}_{calendar.month_name[date.month]}"
    
    # Path: SISTEMAS/TRACKING/2026/02_February/
    output_dir = os.path.join(TRACKING_DIR, year, month_name)
    create_directory(output_dir)
    
    filename = f"{date.strftime('%Y-%m-%d')}_Daily.md"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return

    with open(os.path.join(TEMPLATES_DIR, "daily.md"), "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("{{date}}", date.strftime("%A, %B %d, %Y"))
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Created: {filepath}")

def generate_weekly():
    date = get_date_input("Enter any date within the week (YYYY-MM-DD): ")
    # Find start (Monday) and end (Sunday) of the week
    start_of_week = date - datetime.timedelta(days=date.weekday())
    end_of_week = start_of_week + datetime.timedelta(days=6)
    week_num = date.isocalendar()[1]
    
    year = str(date.year)
    month_name = f"{date.month:02d}_{calendar.month_name[date.month]}"
    
    output_dir = os.path.join(TRACKING_DIR, year, month_name)
    create_directory(output_dir)
    
    filename = f"{end_of_week.strftime('%Y-%m-%d')}_Weekly_Review.md"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return

    with open(os.path.join(TEMPLATES_DIR, "weekly.md"), "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("{{week_num}}", str(week_num))
    content = content.replace("{{date_start}}", start_of_week.strftime("%b %d"))
    content = content.replace("{{date_end}}", end_of_week.strftime("%b %d"))
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Created: {filepath}")

def generate_monthly():
    date = get_date_input("Enter any date within the month (YYYY-MM-DD): ")
    year = str(date.year)
    month_name = f"{date.month:02d}_{calendar.month_name[date.month]}"
    
    output_dir = os.path.join(TRACKING_DIR, year, month_name)
    create_directory(output_dir)
    
    # Last day of month
    last_day = calendar.monthrange(date.year, date.month)[1]
    filename = f"{date.year}-{date.month:02d}-{last_day}_Monthly_Review.md"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return

    with open(os.path.join(TEMPLATES_DIR, "monthly.md"), "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("{{month_name}}", calendar.month_name[date.month])
    content = content.replace("{{year}}", year)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Created: {filepath}")

def main():
    print("--- Bryan's ML Journey Tracker ---")
    print("1. Daily Log")
    print("2. Weekly Review")
    print("3. Monthly Review")
    
    choice = input("Select option (1-3): ")
    
    if choice == "1":
        generate_daily()
    elif choice == "2":
        generate_weekly()
    elif choice == "3":
        generate_monthly()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
