# Wimbledon pilot: finishing the court setup

The court recorder (watch or phone → GoPro → Pi → YouTube raw + highlights) works. Three
things are left: reaching the Pi's page through the travel router, calibrating the court for
where the camera is now, and one YouTube link for the club.

Do them in this order: you need the page to calibrate.

```
club router ── travel router (its own Wi-Fi, NAT) ── Pi "biquet" ── USB ── GoPro
   club Wi-Fi                 192.168.x.1               192.168.x.N         172.2x.1yy.51
```

---

## 1. Reaching the recorder page

The travel router is a **router**. It gives the Pi a private address behind its own NAT, the same
way your home router hides your laptop from the internet. So:

| Where your phone is | Can it open the Pi's page? |
|---|---|
| On the **travel router's** Wi-Fi | **Yes**, at `http://<pi's address>:8080` |
| On the **club's** Wi-Fi | **No**, unless you forward a port (option C) |
| Anywhere, with **Tailscale** | **Yes** (option B) |

### Find the addresses

You need the Pi's address, and the travel router's address (for its admin page). Use one of these:

- **From the travel router's admin page.** Join its Wi-Fi with your phone. The admin address is
  printed on the router's label, usually `192.168.8.1` on GL.iNet. Open it and look at
  **Clients**: `biquet` is the Pi.
- **From the Pi itself**, if you can SSH in (on the travel router's Wi-Fi, `ssh nnewihe@biquet.local`
  usually works from the Mac even when the phone can't resolve `.local`):

  ```bash
  anya-netinfo
  ```

  This prints the Pi's address, the router it is behind (`Router: 192.168.8.1` is the travel
  router; open that in a browser for its admin page), and the recorder page's URL. Without the
  installer's link, run `bash ~/anya/pi/bin/anya-netinfo`.

### A. Join the travel router's Wi-Fi (simplest, at the court)

1. In the travel router's admin page, find **DHCP / Address reservation** (GL.iNet: *Network →
   LAN → Address Reservation*) and reserve the Pi's current address for `biquet`. Then the
   address never changes.
2. On your phone, join the travel router's Wi-Fi and bookmark `http://<that address>:8080`.
   Add it to the home screen. It then works like an app.

### B. Tailscale: from anywhere, including home (recommended for you)

It's free for personal use, needs no router changes, and works through both routers.

On the Pi, once:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

```bash
sudo tailscale up
```

Open the link it prints and sign in. Then install the Tailscale app on your phone and sign in with
the same account. The page is at **http://biquet:8080** from anywhere, on any network. `anya-netinfo`
lists the Tailscale address too. You also get `ssh nnewihe@biquet` from the Mac at home, which
makes updates and log checks possible without driving to the club.

### C. Members on the club Wi-Fi (only if they should press Start themselves)

The page has **no login**: anyone who can open it can start and stop recordings. If you want that
for members:

1. In the travel router's admin page, open **Port forwarding** and forward TCP port `8080`
   (WAN) to the Pi's reserved address, port `8080`.
2. The URL is the travel router's **WAN** address, from the admin page's *Internet* status:
   `http://<WAN address>:8080`. Ask the club to give the travel router a fixed address on
   their network, or it can change.

Alternatively, switch the travel router to **Access Point** mode. Then it adds no NAT, and the Pi
gets an address directly on the club network. The Pi's address then comes from the club's
router, so reserve it there.

Members don't need any of this to **watch** videos: that's the YouTube link (section 3).

---

## 2. Court calibration for the current camera position

anya needs to know where the court is in the picture. You click it once per camera position; every
recording after that is checked against it. A small nudge is corrected; a knocked mount stops with
**"needs court calibration"** instead of producing a wrong reel.

1. Make sure the GoPro is where it will stay, with all **four corners of the singles court in
   the picture**. Webcam mode is Linear, which crops the edges, so check a recording, not the
   GoPro's own screen.
2. **Record about 20 seconds** (watch or page), and press Stop. Wait until it appears under
   *Recent recordings*.
3. Open **http://<pi>:8080/calibrate** (or *Court calibration* at the bottom of the main page).
   Turn the phone sideways.
4. Tap the four **singles** corners, where the inner sideline meets the baseline:
   **1 near left → 2 near right → 3 far right → 4 far left**. *Near* is the baseline closest to the
   camera.
   - Hold and drag a corner to move it. A magnifier shows exactly where it sits.
   - The arrows nudge the last corner touched by one pixel.
5. Check the **yellow court**: baselines, service lines, the centre line and the doubles alleys
   should sit on the painted lines. If a service line is off, a corner is off.
6. **Save.** Recordings that were waiting for calibration are queued again automatically. The
   old calibration is kept in `/srv/anya/site.prev`.

Redo this whenever the camera is moved. The 20-second test clip uploads like any other recording;
delete it in YouTube Studio if you don't want it in the club playlist.

**When you reopen the page, it shows the saved corners on the newest recording.** If the lines
still sit on the court, the camera hasn't moved and there's nothing to do. That makes it a quick
check after a windy day.

---

## 3. One YouTube link for the club

Every raw recording and highlights video goes into **one unlisted playlist**, newest at the top.
You share that playlist's link with the club: members tap it and see every session. Nothing shows
up in YouTube search, and only people with the link can find the videos.

### Once, on the Mac

1. The existing token can upload but **can't manage playlists**. Make a new one that can, still
   signed in as **smarttenniswithandy@gmail.com**:

   ```bash
   PYTHONPATH=pi python -m anya_pi youtube-auth --client-secrets client_secret.json --out youtube_token.json --playlists
   ```

2. Create the playlist:

   ```bash
   PYTHONPATH=pi python -m anya_pi youtube-playlist "Wimbledon Tennis Club · Court Videos" --description "Every session recorded on court, newest first. Raw recording + highlights (dead time cut out)." --token youtube_token.json
   ```

   It prints the playlist link and a `playlist_id = "PL…"` line.

3. Copy the new token to the Pi, replacing the old one:

   ```bash
   scp youtube_token.json nnewihe@<pi>:/tmp/
   ```

   ```bash
   ssh nnewihe@<pi> 'sudo install -o anya -g anya -m 600 /tmp/youtube_token.json /srv/anya/state/ && rm /tmp/youtube_token.json'
   ```

4. On the Pi, put the `playlist_id` line under `[youtube]` in `/srv/anya/config.toml`, then:

   ```bash
   sudo systemctl restart anya-worker anya-recorder
   ```

The recorder page now has an **"All club videos on YouTube"** button. Videos uploaded before this
aren't in the playlist; add them in YouTube Studio (*Content → select → Add to playlist*).

### Sharing it

- Send the club the playlist link (`https://www.youtube.com/playlist?list=PL…`) for their
  newsletter, WhatsApp group or members' page.
- Print it as a **QR code** at the court, next to the camera: a phone camera opens the playlist
  directly.
- Titles say when and who: "6:30 PM · Nov 10, 2026 · Andy Session", plus "… Highlights". Members
  find theirs by date and time, so ask them to type their first name on the page or watch before
  Start.

### Privacy

- **Unlisted** means anyone *with the link* can watch. Forwarding the link forwards access. That
  is usually right for a club. Ask the club whether members, and any juniors, need to be told
  that the court is recorded. A sign at the court is the usual answer.
- **If videos come out private** (the page shows "(private)"), the Google Cloud project hasn't
  passed YouTube's API audit, and private videos won't play for members even in the playlist.
  Switch them to Unlisted in YouTube Studio, or apply for the audit (`pi/README.md`). You
  mentioned uploads already come out unlisted, so this probably doesn't apply.
- The **quota** is about six uploads a day, which is three sessions (raw + highlights). Busy days
  need a quota increase from Google.

---

## Pilot checklist

- [ ] Pi reachable: reserved address on the travel router, bookmarked on the phone (and Tailscale)
- [ ] All four singles corners visible in a recording, with the GoPro on Linear
- [ ] Court calibrated at `/calibrate`; the yellow court sits on the lines
- [ ] One full session: watch Start → play → watch Stop → raw on YouTube → highlights on YouTube
- [ ] Both videos are in the club playlist, newest first, and play from a phone that isn't signed in
- [ ] Playlist link and QR code sent to the club; recording sign at the court
