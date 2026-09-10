"""
oauth_client.example.py — template for desktop/oauth_client.py.

    cp oauth_client.example.py oauth_client.py    # then fill in the values

`oauth_client.py` is gitignored and must stay that way. Copy this, paste your
Google OAuth *Desktop* client's id and secret in, and the build picks them up.
Without it the app still runs; Google sign-in is simply not offered, and
firebase_config.is_configured() reports the build as unconfigured.

Why these are not committed, when RFC 8252 s8.5 says an installed app's client
secret is not confidential:

  Both statements are true and they are not in conflict. The secret CANNOT be
  protected once the app is on someone's machine -- anyone can unpack the
  bundle and read it, which is why PKCE rather than the secret is what secures
  the flow (see oauth_loopback.py). But this repository is PUBLIC, and putting
  it here is a different act with a different consequence: bots scrape public
  repositories within minutes, and an id and secret in someone else's hands
  let them stand up a phishing app that shows YOUR consent screen -- "Anya
  Tennis wants access to your Google Account" -- to your own users.

  "Cannot be kept secret from a determined user" is not the same as "should be
  published". GitHub's push protection blocked exactly this, and it was right.
"""

GOOGLE_CLIENT_ID = "REPLACE_ME.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "REPLACE_ME"
