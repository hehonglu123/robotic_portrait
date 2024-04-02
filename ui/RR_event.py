import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import numpy as np
import traceback, time

minimal_create_interface="""
service experimental.minimal_create

object create_obj
    event response_received()
    event stop_robot()
end object
"""

class create_impl(object):
    def __init__(self):   
        self.response_received=RR.EventHook()

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
credential_directory = "credentials/"

def main():
    with RR.ServerNodeSetup("experimental.minimal_create", 52222):
        #Register the service type
        RRN.RegisterServiceType(minimal_create_interface)

        create_inst=create_impl()
        
        #Register the service
        RRN.RegisterService("Create","experimental.minimal_create.create_obj",create_inst)
        try:
            creds = Credentials.from_authorized_user_file(credential_directory+"token.json", SCOPES)
        except:
            creds = None
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credential_directory+"credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(credential_directory+"token.json", "w") as token:
                token.write(creds.to_json())

        try:
            service = build("sheets", "v4", credentials=creds)

            last_response_count = -1
            while True:

                # Call the Sheets API
                sheet = service.spreadsheets()
                result = (
                    sheet.values()
                    .get(spreadsheetId="11IqWz5_Tk3X2UoyoLDZkzID8jikq28N9HT0LmdJ6EcA", range="Form Responses 1")
                    .execute()
                )
                values = result.get("values", [])
                if last_response_count == -1:
                    last_response_count = len(values)
                if len(values) > last_response_count:
                    print("New Rsponse Received")
                    last_response_count = len(values)
                    create_inst.response_received.fire()
                
                time.sleep(1)
        
        except:
            traceback.print_exc()


if __name__ == "__main__":
    main()