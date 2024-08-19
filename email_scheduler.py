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
import docx
import textract
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import schedule
import chardet
import imaplib
import email
import uuid

# MySQL connection details
db_config = {
    'user': 'root',
    'password': 'Shakil2831',
    'host': 'localhost',
    'database': 'email_processing_db'
}

# Folder to save attachments
attachment_folder = 'attachments/'
stop_file = "stopfile.txt"

# Function to perform OCR on an image
def ocr_image(image):
    print("Hello")
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Failed to OCR image: {e}")
        return ""

# Function to check if a PDF is scanned or normal
def is_scanned_pdf(file_path):
    try:
        document = fitz.open(file_path)
        text_present = False
        image_count = 0
        
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text("text").strip()
            
            if text:
                text_present = True
                break  # If text is found, it's a normal PDF

            # Count image objects on the page
            image_list = page.get_images(full=True)
            image_count += len(image_list)
        
        if text_present:
            return False  # It's a normal PDF
        elif image_count > 0:
            return True  # It's a scanned PDF
        else:
            return "Unknown"  # No text or images found, could be an empty or unsupported format
    except Exception as e:
        print(f"Error checking PDF type: {e}")
        return "Error"

# Function to read PDF using PyMuPDF
def read_pdf(file_path):
    try:
        document = fitz.open(file_path)
        content = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                content += text
            else:
                # If no text is found, perform OCR on the page image
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                content += ocr_image(img)
        return content
    except Exception as e:
        print(f"Failed to read PDF file {file_path}: {e}")
        return ""

# Function to read DOCX file
def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        content = ""
        for para in doc.paragraphs:
            content += para.text + "\n"
        return content
    except Exception as e:
        print(f"Failed to read DOCX file {file_path}: {e}")
        return ""

# Function to read TXT file
def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Failed to read TXT file {file_path}: {e}")
        return ""

# Function to read XLSX file
def read_xlsx(file_path):
    try:
        df = pd.read_excel(file_path)
        content = df.to_string()
        return content
    except Exception as e:
        print(f"Failed to read XLSX file {file_path}: {e}")
        return ""

# Function to read attachments
def read_attachment(file_path):
    try:
        if file_path.endswith('.pdf'):
            if is_scanned_pdf(file_path):
                print("Scanned PDF detected")
                return read_pdf(file_path)  # Treat it as a scanned PDF and perform OCR
            else:
                print("Normal PDF detected")
                return read_pdf(file_path)  # Treat it as a normal PDF
        elif file_path.endswith('.docx'):
            return read_docx(file_path)
        elif file_path.endswith('.txt'):
            return read_txt(file_path)
        elif file_path.endswith('.xlsx'):
            return read_xlsx(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            return ocr_image(Image.open(file_path))
        else:
            return textract.process(file_path).decode('utf-8')
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return ""

# Function to extract email content and attachments
def get_email_content(email_message):
    body = ""
    attachments = []
    for part in email_message.walk():
        if part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                detected_encoding = chardet.detect(payload)['encoding']
                if detected_encoding:
                    try:
                        body = payload.decode(detected_encoding)
                    except UnicodeDecodeError:
                        body = payload.decode(detected_encoding, errors='replace')
                else:
                    body = payload.decode(errors='replace')
        elif part.get_content_disposition() == 'attachment':
            filename = part.get_filename()
            if filename:
                attachment_id = str(uuid.uuid4())  # Generate a unique ID for the attachment
                attachment_path = os.path.join(attachment_folder, attachment_id + '_' + filename)
                with open(attachment_path, 'wb') as f:
                    f.write(part.get_payload(decode=True))
                attachment_content = read_attachment(attachment_path)
                attachments.append((attachment_id, filename, attachment_path, attachment_content))
    return body, attachments

# Function to fetch emails
def fetch_emails():
    imap_host = 'imap.gmail.com'
    imap_user = 'mail_id'
    imap_pass = 'app_password'

    mail = imaplib.IMAP4_SSL(imap_host)
    mail.login(imap_user, imap_pass)
    mail.select('inbox')

    status, email_ids = mail.search(None, 'UNSEEN')
    email_ids = email_ids[0].split()

    # Connect to MySQL database
    db_connection = mysql.connector.connect(**db_config)
    cursor = db_connection.cursor()

    for e_id in email_ids:
        status, data = mail.fetch(e_id, '(RFC822)')
        email_msg = email.message_from_bytes(data[0][1])

        # Generate a unique ID for the email entry
        email_entry_id = str(uuid.uuid4())

        # Extract the email sender
        from_ = email_msg['From']

        # Extract the email body and attachments
        email_content, attachments = get_email_content(email_msg)

        # Initialize attachment list string
        attachment_list_str = ""

        # Ensure email_content is not None
        if email_content is None:
            email_content = ""

        # Initialize attach_content
        attach_content = ""

        # Process attachments and add their content to the email body and attach_content
        for attachment_id, filename, attachment_path, attachment_content in attachments:
            email_content += f""
            attach_content += attachment_content or "[Attachment content could not be read]"
            attachment_list_str += f"{attachment_id}_{filename},"

        # Remove the trailing comma from the attachment list string
        attachment_list_str = attachment_list_str.rstrip(',')

        # Combine body and attachment content
        body_attach_content = email_content + "\n\n" + attach_content

        # Add the email details to the MySQL table
        sql = "INSERT INTO emails (id, sender_email, body, attachments, attach_content, body_attach, processed) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (email_entry_id, from_, email_content, attachment_list_str, attach_content, body_attach_content, False)
        cursor.execute(sql, val)

    # Commit the transaction
    db_connection.commit()

    # Close cursor and connection
    cursor.close()
    db_connection.close()

    mail.logout()

# Schedule the fetch_emails function to run every 2 minutes
schedule.every(0.5).minutes.do(fetch_emails)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)

    # Check for stop file
    if os.path.exists(stop_file):
        print("Stop file detected. Exiting.")
        break
