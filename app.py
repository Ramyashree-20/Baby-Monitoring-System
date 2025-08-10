# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from threading import Thread
from datetime import datetime, timedelta
from flask import send_from_directory
import os
import image_to_timestamp_logger
import baby_sleep_analyzer
import chatbot

app = Flask(__name__)
app.secret_key = 'your_secret_key'
notifications = []

@app.route('/service-worker.js')
def service_worker():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'service-worker.js')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['logged_in'] = True
        return redirect(url_for('baby_details'))
    return render_template('login.html')

@app.route('/baby_details', methods=['GET', 'POST'])
def baby_details():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        session['baby_details'] = {
            'name': request.form['name'],
            'age': request.form['age'],
            'weight': request.form['weight'],
            'dob': request.form['dob'],
            'gender': request.form['gender']
        }
        return redirect(url_for('dashboard'))
    return render_template('baby_details.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in') or not session.get('baby_details'):
        return redirect(url_for('login'))

    vaccination_reminders = get_vaccination_reminders(
        session['baby_details']['dob'],
        session['baby_details']['gender']
    )

    graph_html = baby_sleep_analyzer.generate_sleep_graph()

    return render_template('dashboard.html',
                           baby_name=session['baby_details']['name'],
                           notifications=notifications,
                           vaccination_reminders=vaccination_reminders,
                           sleep_graph=graph_html)

@app.route('/add_notification', methods=['POST'])
def add_notification():
    alert = request.json
    timestamp = datetime.now().strftime('%H:%M:%S')
    notifications.insert(0, f"[{timestamp}] {alert['message']}")
    if len(notifications) > 10: # Keep the list to a reasonable size
        notifications.pop()
    return jsonify({'status': 'success'})

@app.route('/get_notifications')
def get_notifications():
    return jsonify(notifications)

@app.route('/ask_chatbot', methods=['POST'])
def ask_chatbot():
    user_message = request.json['message']
    bot_response = chatbot.get_response(user_message)
    return jsonify({'response': bot_response})

def get_vaccination_reminders(dob_str, gender_str):
    """
    Calculates a detailed list of upcoming vaccination reminders based on the
    baby's date of birth and gender, using a comprehensive schedule.
    """
    try:
        dob = datetime.strptime(dob_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        return ["Invalid Date of Birth format. Please use YYYY-MM-DD."]

    VACCINE_SCHEDULE = {
        # Weeks to Days (weeks * 7)
        42: ("BCG, Hep B1, OPV", "all"),
        70: ("DTwP/DTaP1, Hib-1, IPV-1, Hep B2, PCV 1, Rota-1", "all"),
        98: ("DTwP/DTaP2, Hib-2, IPV-2, Hep B3, PCV 2, Rota-2", "all"),
        # Months to Days (months * 30.4 average)
        int(6 * 30.4): ("Influenza-1", "all"),
        int(7 * 30.4): ("Influenza-2", "all"),
        int(6 * 30.4): ("Typhoid Conjugate Vaccine (due between 6-9 months)", "all"), # Start of range
        int(9 * 30.4): ("MMR 1 (Mumps, Measles, Rubella)", "all"),
        int(12 * 30.4): ("Hepatitis A-1", "all"),
        int(12 * 30.4): ("PCV Booster (due between 12-15 months)", "all"), # Start of range
        int(15 * 30.4): ("MMR 2, Varicella", "all"),
        int(16 * 30.4): ("DTwP/DTaP, Hib, IPV (due between 16-18 months)", "all"), # Start of range
        int(18 * 30.4): ("Hepatitis A-2, Varicella 2", "all"),
        # Years to Days (years * 365.25 to account for leap years)
        int(4 * 365.25): ("DTwP/DTaP, IPV, MMR 3 (due between 4-6 years)", "all"), # Start of range
        int(9 * 365.25): ("HPV (2 doses, for Girls)", "Girls"),
        int(10 * 365.25): ("Tdap/Td (due between 10-12 years)", "all"), # Start of range
    }

    today = datetime.now()
    reminders_with_dates = []

    # Process the main schedule
    for due_in_days, (vaccine_name, required_gender) in VACCINE_SCHEDULE.items():
        # Skip if the vaccine is gender-specific and doesn't match
        if required_gender.lower() != 'all' and required_gender.lower() != gender_str.lower():
            continue

        due_date = dob + timedelta(days=due_in_days)

        # Only show reminders for vaccines that haven't been given yet
        if due_date > today:
            days_until_due = (due_date - today).days
            # Differentiate between "Due Soon" and "Upcoming"
            if days_until_due <= 30:
                formatted_string = f"‼️ DUE SOON (in {days_until_due} days): {vaccine_name}"
            else:
                formatted_string = f"🗓️ Upcoming: {vaccine_name} on {due_date.strftime('%b %d, %Y')}"
            reminders_with_dates.append((due_date, formatted_string))

    # Special handling for the Annual Influenza Vaccine (after 6 months old)
    age_in_days = (today - dob).days
    if age_in_days > 182: # If baby is older than 6 months
        next_flu_date = datetime(today.year, 10, 1)
        if today > next_flu_date: # If this year's flu season has already started
            next_flu_date = datetime(today.year + 1, 10, 1) # The reminder is for next year

        # add the reminder if it's for a date in the future
        if next_flu_date > today:
             reminders_with_dates.append((next_flu_date, f"🗓️ Upcoming: Annual Influenza Vaccine around {next_flu_date.strftime('%B %Y')}"))

    # Sort reminders chronologically
    reminders_with_dates.sort()
    final_reminders = [item[1] for item in reminders_with_dates]

    if not final_reminders:
        return ["All scheduled vaccinations are up to date for now!"]

    return final_reminders

if __name__ == '__main__':
    logger_thread = Thread(target=image_to_timestamp_logger.start_logging, daemon=True)
    logger_thread.start()
    app.run(debug=True, use_reloader=False)