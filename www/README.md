# Anya Tennis Landing Page

A simple GitHub Pages landing page for Anya Tennis.

## Files

- `index.html` — landing page and contact modal
- `style.css` — styling
- `script.js` — modal behavior and mailto contact form
- `anya-poster.png` — Anya Tennis promotional poster used as the background

## GitHub Pages

1. Create a GitHub repository, for example `ANYA-WEBSITE`.
2. Upload all four files to the repository root.
3. In GitHub, open **Settings → Pages**.
4. Set the source to **Deploy from a branch**.
5. Select the `main` branch and `/ (root)`.
6. Save.
7. Add `anyatennis.com` as the custom domain.
8. At GoDaddy, configure the DNS records GitHub provides.

## Contact form note

Because GitHub Pages is static hosting, the form cannot send email through a server by itself. The current implementation opens the visitor's default email application with the recipient, subject, and message pre-filled.

If you want a true browser-based "Send" button with no email app, the form can later be connected to a form backend such as Formspree or another email service.
