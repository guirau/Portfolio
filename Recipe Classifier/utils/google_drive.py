"""Google Drive utils for listing, downloading, and deleting files."""

import io
import json
import os

import backoff
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import BatchHttpRequest, MediaIoBaseDownload
from loguru import logger

# Load environment variables from .env file
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDrive:

    def __init__(self, scopes=None):
        """Initializes the GoogleDrive class with authentication details."""
        self.service_account_info = self.get_service_account_info()
        self.scopes = scopes if scopes else SCOPES
        self.client = self.authenticate()

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter
    )
    def authenticate(self):
        """Authenticates Google Drive using a service account."""
        creds = Credentials.from_service_account_info(
            self.service_account_info, scopes=self.scopes
        )
        return build("drive", "v3", credentials=creds)

    def get_service_account_info(self):
        """Retrieves the service account JSON content from the environment variable and loads it into a dictionary."""
        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT")
        if not service_account_json:
            logger.error("The GOOGLE_SERVICE_ACCOUNT environment variable is not set.")
            raise

        try:
            service_account_info = json.loads(service_account_json)
            logger.info("GOOGLE_SERVICE_ACCOUNT loaded from .env")
            return service_account_info
        except json.JSONDecodeError as ex:
            logger.error(f"Error decoding service account JSON: {ex}")
            raise

    def list_files_in_folder(self, folder_id):
        """Lists the JPEG files in a Google Drive folder."""
        logger.info(f"Listing JPEG files in folder: {folder_id}")
        files = []
        page_token = None

        while True:
            results = (
                self.client.files()
                .list(
                    q=f"'{folder_id}' in parents and mimeType='image/jpeg'",
                    pageSize=1000,
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )

            items = results.get("files", [])
            files.extend(items)

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        return files

    def download_file(self, file_id, file_name, download_dir):
        """Downloads a file from Google Drive."""
        request = self.client.files().get_media(fileId=file_id)
        file_path = os.path.join(download_dir, file_name)
        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.trace(
                    f"Downloading {file_name}: {int(status.progress() * 100)}%"
                )
            return file_path

    def delete_file(self, file_id):
        """Deletes a file from Google Drive."""
        try:
            self.client.files().delete(fileId=file_id).execute()
            logger.info(f"Successfully deleted file with ID: {file_id}")
        except Exception as ex:
            logger.error(f"Failed to delete file with ID: {file_id}. Error: {ex}")

    def delete_files_batch(self, file_ids):
        """Deletes multiple files in batch from Google Drive."""
        batch = self.client.new_batch_http_request(callback=self.batch_callback)

        for file_id in file_ids:
            batch.add(self.client.files().delete(fileId=file_id))

        try:
            batch.execute()
            logger.info("Batch deletion executed successfully.")
        except Exception as ex:
            logger.error(f"Batch deletion failed: {ex}")

    def batch_callback(self, request_id, response, exception):
        """Callback function for handling responses in batch operations."""
        if exception is not None:
            logger.error(f"Error in request {request_id}: {exception}")
        else:
            logger.info(f"Request {request_id} completed successfuly.")
