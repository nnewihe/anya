"""
Optional upload of each finished reel to YouTube (unlisted by default).

One-time authorisation, on a laptop with a browser:

    python -m anya_pi youtube-auth --client-secrets client_secret.json --out youtube_token.json

then copy youtube_token.json to the Pi (default /srv/anya/state/, mode 600).

Two Google-side rules decide whether this works, and neither is visible from
here until it bites (see pi/README.md for the steps):

  * The OAuth consent screen must be "In production".  In "Testing" the refresh
    token dies after 7 days and every upload after that fails with
    invalid_grant.
  * A Google Cloud project that has not passed YouTube's API compliance audit
    has every API upload LOCKED TO PRIVATE, whatever privacyStatus was asked
    for.  `upload` reads back the privacy YouTube actually applied and reports
    `forced_private` so the job status says so instead of claiming "unlisted".

Quota: an upload costs 1600 of the default 10,000 units a day -- about six
reels a day.
"""

import http.client
import os
import random
import socket
import ssl
import time

SCOPE_UPLOAD = "https://www.googleapis.com/auth/youtube.upload"
SCOPE_MANAGE = "https://www.googleapis.com/auth/youtube"   # playlists only

CHUNK = 16 * 1024 * 1024
RETRY_STATUS = {500, 502, 503, 504}
MAX_CHUNK_RETRIES = 10
SPORTS_CATEGORY = "17"


class UploadError(RuntimeError):
    pass


def authorize(client_secrets, token_out, playlists=False, port=0):
    """Browser consent on this machine; writes the refresh token to `token_out`."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    scopes = [SCOPE_UPLOAD] + ([SCOPE_MANAGE] if playlists else [])
    flow = InstalledAppFlow.from_client_secrets_file(client_secrets, scopes)
    creds = flow.run_local_server(port=port, access_type="offline",
                                  prompt="consent")
    _write_token(token_out, creds)
    print(f"token written to {token_out} -- copy it to the Pi (keep it mode 600)")


def _write_token(path, creds):
    """Owner-only: the refresh token is a standing credential for the channel."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(creds.to_json())


def _credentials(token_path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_file(str(token_path))
    if not creds.valid:
        if not creds.refresh_token:
            raise UploadError(f"{token_path} has no refresh token; re-run youtube-auth")
        creds.refresh(Request())
        # Persist the refreshed access token; the refresh token is unchanged.
        _write_token(token_path, creds)
    return creds


def _service(token_path):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=_credentials(token_path),
                 cache_discovery=False)


def _transient(exc):
    try:
        from googleapiclient.errors import HttpError
        if isinstance(exc, HttpError):
            return int(getattr(exc.resp, "status", 0)) in RETRY_STATUS
    except ImportError:
        pass
    net = [ConnectionError, TimeoutError, socket.timeout, socket.gaierror,
           ssl.SSLError, http.client.HTTPException]
    try:
        import httplib2
        net.append(httplib2.HttpLib2Error)
    except ImportError:
        pass
    return isinstance(exc, tuple(net))


def run_resumable(request, sleep=time.sleep, log=print, max_retries=MAX_CHUNK_RETRIES):
    """Drive a resumable upload to completion, retrying transient failures.

    Split out so the retry policy is testable without the network.  A retry
    resumes from the last acknowledged chunk (the client keeps the session
    URI), so a flaky court Wi-Fi costs a chunk, not the whole reel.
    """
    response, retries, last_pct = None, 0, -10
    while response is None:
        try:
            status, response = request.next_chunk()
            retries = 0
            if status is not None:
                pct = int(status.progress() * 100)
                if pct >= last_pct + 10:
                    log(f"[youtube] {pct}%")
                    last_pct = pct
        except Exception as e:                  # noqa: BLE001 -- classified below
            if not _transient(e):
                raise
            retries += 1
            if retries > max_retries:
                raise UploadError(f"gave up after {max_retries} retries: {e}") from e
            wait = min(2 ** retries, 300) * (0.5 + random.random())
            log(f"[youtube] transient error ({e}); retry {retries} in {wait:.0f}s")
            sleep(wait)
    return response


def upload(path, title, description, token_path, privacy="unlisted",
           playlist_id="", service=None, log=print):
    """Upload `path`; returns {video_id, url, privacy, requested, forced_private}."""
    from googleapiclient.http import MediaFileUpload
    yt = service or _service(token_path)
    body = {"snippet": {"title": title[:100], "description": description[:5000],
                        "categoryId": SPORTS_CATEGORY},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=CHUNK,
                            resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = run_resumable(req, log=log)
    vid = resp["id"]
    got = (resp.get("status") or {}).get("privacyStatus", privacy)
    out = {"video_id": vid, "url": f"https://youtu.be/{vid}", "privacy": got,
           "requested": privacy, "forced_private": got != privacy and got == "private"}
    if out["forced_private"]:
        log("[youtube] WARNING: YouTube set this video PRIVATE although "
            f"{privacy!r} was requested. Unaudited API projects are locked to "
            "private; change it in YouTube Studio, or apply for the API audit.")
    if playlist_id:
        try:
            yt.playlistItems().insert(part="snippet", body={"snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": vid}}}).execute()
        except Exception as e:                  # noqa: BLE001 -- the upload stands
            log(f"[youtube] uploaded, but adding to playlist failed: {e} "
                f"(playlists need `youtube-auth --playlists`)")
    return out
