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
import fitz
import re

# Establish MySQL connection
def connect_to_database():
    try:
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
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

def weighted_prediction(predictions):
    # Function to compute the weighted prediction
    weights = {'body': 0.40, 'sub': 0.15, 'attach_content': 0.25, 'body_sub': 0.20}
    weighted_score = sum(weights[key] if pred == 'PO' else 0 for key, pred in predictions.items())
    return 'PO' if weighted_score >= 0.5 else 'NO'

def send_email_notification(to_email, from_email, subject, body, attachment_paths=None):
    # Function to send email notification
    from_email = 'mail_id'
    password = 'app_password'  # Sender's email password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Add additional message to the email body
    body += ""
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

def extract_full_address_from_attachment(file_path, search_name="Veolia"):
    try:
        # Check if the file is a PDF
        if file_path.lower().endswith('.pdf'):
            # Open the PDF file
            pdf_document = fitz.open(file_path)
            # Iterate through all pages
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            pdf_document.close()
        else:
            # Handle text files or other formats
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

        # Split text into lines and find lines containing search_name
        lines = text.split('\n')
        address_lines = []
        for i, line in enumerate(lines):
            if search_name.lower() in line.lower():
                # Collect lines following the search_name line
                address_lines = lines[i+1:i+7]  # Adjust the range if needed
                break

        # If address_lines are found, process and print address lines
        if address_lines:
            full_address = " ".join(address_lines).strip()
            print(f"Address extracted from {file_path}:")
            print(full_address)

            # Remove any extraneous text (like emails) from the full address
            cleaned_text = re.sub(r"Email:\s*\S+@\S+", "", full_address).strip()
            cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)  # Replace multiple spaces with a single space

            print(f"Cleaned address text: {cleaned_text}")

            return cleaned_text
        else:
            print(f"No address found for {search_name} in {file_path}.")
            return None

    except Exception as e:
        print(f"Failed to read or extract address from {file_path}: {e}")
        return None

def compare_address_with_database(full_address):
    """Compare the extracted full address with the database."""
    if not full_address:
        return None

    try:
        db_connection, cursor = connect_to_database()
        if db_connection and cursor:
            query = """
                SELECT province, zipcode, country, email_id FROM locate_mail
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            full_address_lower = full_address.lower()

            matching_email_ids = []
            for row in rows:
                province = row[0].lower()  # Tuple index for province
                zipcode = row[1]          # Tuple index for zipcode
                country = row[2].lower()  # Tuple index for country
                
                if province in full_address_lower and zipcode in full_address_lower and country in full_address_lower:
                    matching_email_ids.append(row[3])  # Tuple index for email_id

            if matching_email_ids:
                return matching_email_ids
            else:
                print("No matching email IDs found.")
                return None

        else:
            print("Failed to establish database connection.")
            return None

    except Error as e:
        print(f"Failed to query database: {e}")
        return None


def process_emails(db_connection, cursor, vectorizer, rf_model, feature_names):
    # Function to process emails from MySQL and send notifications
    try:
        # Read data from MySQL table
        query = "SELECT id, sender_email, body, sub, attach_content, body_sub, attachments FROM emails WHERE processed = 0"
        cursor.execute(query)
        emails_data = cursor.fetchall()

        for email in emails_data:
            email_id, sender_email, body, sub, attach_content, body_sub, attachments = email

            # Classify the email based on body, sub, attach_content, and body_sub
            predictions = {
                'body': classify_email(body, vectorizer, rf_model, feature_names),
                'sub': classify_email(sub, vectorizer, rf_model, feature_names),
                'attach_content': classify_email(attach_content, vectorizer, rf_model, feature_names),
                'body_sub': classify_email(body_sub, vectorizer, rf_model, feature_names)
            }

            final_prediction = weighted_prediction(predictions)
            print(f"Email ID {email_id}: Predictions - {predictions}, Final Prediction - {final_prediction}")

            # Handle multiple attachments
            attachment_paths = []
            extracted_addresses = []
            if attachments and isinstance(attachments, str):
                attachment_names_list = attachments.split(',')
                for attachment_name in attachment_names_list:
                    attachment_path = os.path.join('attachments', attachment_name.strip())
                    if os.path.exists(attachment_path):
                        attachment_paths.append(attachment_path)
                        # Extract full address from the attachment
                        full_address = extract_full_address_from_attachment(attachment_path)
                        if full_address:
                            extracted_addresses.append(full_address)
                            # Compare full address with database
                            matching_email_ids = compare_address_with_database(full_address)
                            if matching_email_ids:
                                print(f"Matching Email IDs: {matching_email_ids}")

            # Determine email subject and recipient based on final prediction
            if final_prediction == 'PO':
                subject = 'Purchase Order Mail'
                # Use the matching email IDs for recipient if any found
                recipient = matching_email_ids[0] if matching_email_ids else 'mail_id'

                # Send notification based on final classification
                send_email_notification(recipient, sender_email, subject, body, attachment_paths)
            else:
                subject = 'Non Purchase Order Mail'
                recipient = 'mail_id'
                # Send notification without attachments
                send_email_notification(recipient, sender_email, subject, body)

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
