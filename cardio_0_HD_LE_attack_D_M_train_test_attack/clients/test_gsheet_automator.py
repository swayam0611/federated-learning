import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_name("sheets_api_credentials.json", scope)
client = gspread.authorize(creds)

sheet = client.open("fl_test").sheet1
print("connected to google sheet")

