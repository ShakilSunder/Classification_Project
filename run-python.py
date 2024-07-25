import os
import pandas as pd
import joblib
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import mysql.connector
from mysql.connector import Error

# Establish MySQL connection
def connect_to_database():
    try:
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Shakil2831",
            database="email_processing_db"
        )
        cursor = db_connection.cursor()
        return db_connection, cursor
    except Error as e:
        print(f"Failed to connect to MySQL: {e}")
        return None, None

# Load your machine learning model and vectorizer
def load_model_and_vectorizer():
    try:
        model_path = 'model/random_forest_model.joblib'
        vectorizer_path = 'model/tfidf_vectorizer.joblib'
        feature_names_path = 'model/feature_names.joblib'

        rf_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        feature_names = joblib.load(feature_names_path)

        return rf_model, vectorizer, feature_names
    except Exception as e:
        print(f"Failed to load model or vectorizer: {e}")
        return None, None, None

def classify_email(content, vectorizer, rf_model, feature_names):
    # Function to classify email content as 'PO' or 'NO'
    if pd.isna(content):
        content = ""
    email_vectorized = vectorizer.transform([content])
    email_df = pd.DataFrame(email_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    
    aligned_email_df = pd.DataFrame(columns=feature_names)
    for feature in feature_names:
        if feature in email_df.columns:
            aligned_email_df[feature] = email_df[feature]
        else:
            aligned_email_df[feature] = 0

    aligned_email_df = aligned_email_df.fillna(0)
    aligned_email_array = aligned_email_df.to_numpy()
    prediction = rf_model.predict(aligned_email_array)
    predicted_category = 'NO' if prediction[0] == 0 else 'PO'
    
    return predicted_category

def send_email_notification(to_email, from_email, subject, body, attachment_paths=None):
    # Function to send email notification
    from_email = 'shakilsunder@gmail.com'
    password = 'wjgp lxtv piay pzam'  # Sender's email password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Add additional message to the email body
    body += "\n\nCSR Team, Please Check and Take action."
    msg.attach(MIMEText(str(body), 'plain'))

    # Attach the files if attachment_paths are provided
    if attachment_paths:
        for attachment_path in attachment_paths:
            if os.path.exists(attachment_path):
                part = MIMEBase('application', 'octet-stream')
                with open(attachment_path, "rb") as attachment:
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(attachment_path)}")
                msg.attach(part)

    try:
        # Connect to SMTP server and send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print(f"Email sent to {to_email}")

        # Delete the attachment files after sending
        if attachment_paths:
            for attachment_path in attachment_paths:
                if os.path.exists(attachment_path):
                    os.remove(attachment_path)
                    print(f"Attachment deleted: {attachment_path}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def process_emails(db_connection, cursor, vectorizer, rf_model, feature_names):
    # Function to process emails from MySQL and send notifications
    try:
        # Read data from MySQL table
        query = "SELECT id, sender_email, body, attach_content, body_attach, attachments FROM emails WHERE processed = 0"
        cursor.execute(query)
        emails_data = cursor.fetchall()

        for email in emails_data:
            email_id, sender_email, body, attach_content, body_attach, attachments = email

            # Classify the email based on body, attach_content, and body_attach
            body_prediction = classify_email(body, vectorizer, rf_model, feature_names)
            attach_content_prediction = classify_email(attach_content, vectorizer, rf_model, feature_names)
            body_attach_prediction = classify_email(body_attach, vectorizer, rf_model, feature_names)

            predictions = [body_prediction, attach_content_prediction, body_attach_prediction]
            final_prediction = max(set(predictions), key=predictions.count)
            print(f"Email ID {email_id}: Predictions - {predictions}, Final Prediction - {final_prediction}")

            # Handle multiple attachments
            attachment_paths = []
            if attachments and isinstance(attachments, str):
                attachment_names_list = attachments.split(',')
                for attachment_name in attachment_names_list:
                    attachment_path = os.path.join('attachments', attachment_name.strip())
                    if os.path.exists(attachment_path):
                        attachment_paths.append(attachment_path)

            # Determine email subject and recipient based on final prediction
            if final_prediction == 'PO':
                subject = 'Purchase Order Mail'
                recipient = 'reachmeout2k3@gmail.com'
            else:
                subject = 'Non Purchase Order Mail'
                recipient = 'reachmeout2k3@gmail.com'

            # Send notification based on final classification
            send_email_notification(recipient, sender_email, subject, body, attachment_paths)

            # Update processed status in the database
            update_query = "UPDATE emails SET processed = 1 WHERE id = %s"
            cursor.execute(update_query, (email_id,))
            db_connection.commit()

    except Error as e:
        print(f"Failed to process emails: {e}")

# Main script execution
if __name__ == "__main__":
    while True:
        db_connection, cursor = connect_to_database()
        if db_connection and cursor:
            rf_model, vectorizer, feature_names = load_model_and_vectorizer()
            if rf_model is not None and vectorizer is not None and feature_names is not None:
                try:
                    process_emails(db_connection, cursor, vectorizer, rf_model, feature_names)
                except KeyboardInterrupt:
                    print("Process interrupted.")
                except Error as e:
                    print(f"Error during processing: {e}")
            else:
                print("Failed to load model, vectorizer, or feature names.")
        else:
            print("Failed to establish database connection.")

        if cursor:
            cursor.close()
        if db_connection:
            db_connection.close()

        time.sleep(30)  # Wait for 30 seconds before the next iteration
