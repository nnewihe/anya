# Anya Tennis Landing Page

A simple GitHub Pages landing page for Anya Tennis.

## Files

- `index.html` — landing page and contact modal
- `about.html` — how it works: the step-by-step walkthrough, the founder's
  story, and Andy's background
- `download.html` — download page for macOS and Windows
- `style.css` — styling, shared by both pages
- `script.js` — modal behavior and mailto contact form
- `anya-poster.png` — Anya Tennis promotional poster used as the background
- `screenshots/` — the two app screenshots `about.html` shows; see the README
  in there for what each one is and what size it wants
- `CNAME` — the custom domain, `anyatennis.com`

## GitHub Pages

**This is a copy.** The site is live at <https://anyatennis.com> and is served
by GitHub Pages from the root of a separate repository,
<https://github.com/nnewihe/ANYA-WEBSITE>. These files are mirrored here so the
site's source sits beside the app it advertises; editing them here changes
nothing on its own.

To change the live site, push to ANYA-WEBSITE and copy the result back here.
Pages can only serve from `/` or `/docs` in a repository, which is why this
`www/` directory could never have been the source itself.

Note there are two distinct sites, and they are not the same thing:

| | |
|---|---|
| <https://anyatennis.com> | this directory → ANYA-WEBSITE. The public front door and the download page. |
| <https://nnewihe.github.io/anya/> | `docs/` on `main` of this repo. The tester-facing page, and where the privacy policy and terms live. |

## Downloads

`download.html` does **not** host the installers. GitHub blocks any file over
100 MB and caps a Pages site at 1 GB; the artifacts are 358 MB, 484 MB and
331 MB. They are release assets on `nnewihe/anya`, linked through
`/releases/latest/download/<name>`. Those names carry no version, so the page
keeps working across releases without being edited — see `desktop/release.sh`,
which is what uploads them.

## Contact form note

Because GitHub Pages is static hosting, the form cannot send email through a server by itself. The current implementation opens the visitor's default email application with the recipient, subject, and message pre-filled.

If you want a true browser-based "Send" button with no email app, the form can later be connected to a form backend such as Formspree or another email service.
