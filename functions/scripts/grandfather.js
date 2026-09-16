#!/usr/bin/env node
/**
 * grandfather.js — the free-year allowlist for beta testers.
 *
 * Thirteen betas were tested by people who got nothing for it. An email on
 * this list gets a year free the moment they create an account:
 * beforeUserCreated (src/grandfather.ts) looks up sha256 of the lowercased,
 * trimmed address and writes the entitlement. Hashed rather than plaintext so
 * that a rules mistake leaks a set of digests, not a customer list.
 *
 * Plain JavaScript on purpose. The TypeScript version of this needed ts-node
 * and a build step that did not include scripts/, so it could not actually be
 * run -- which for a script used once, under time pressure, on the day you go
 * paid, is the same as not existing.
 *
 * TWO WAYS TO USE IT.
 *
 * 1. No credentials at all (recommended for a one-off list):
 *
 *      node scripts/grandfather.js hash a@b.com c@d.com
 *      node scripts/grandfather.js hash --file testers.txt
 *
 *    Prints a document id per address. Create each one as an empty document in
 *    the `grandfathered` collection in the Firestore console. Nothing secret
 *    is involved and no key has to be downloaded.
 *
 * 2. Directly, with admin credentials:
 *
 *      export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json
 *      node scripts/grandfather.js add    a@b.com c@d.com
 *      node scripts/grandfather.js check  a@b.com
 *      node scripts/grandfather.js remove a@b.com
 *      node scripts/grandfather.js list
 *      node scripts/grandfather.js comp   a@b.com [days=365]
 *
 *    That key is a ROOT credential for the whole project. .gitignore blocks
 *    serviceAccount*.json; delete it when you are done rather than leaving it
 *    on disk.
 *
 * `comp` is the escape hatch the rollout plan calls for -- someone will need
 * comping in the first month, and doing it by hand means getting the Firestore
 * document and the custom claim to agree, which is easy to botch.
 */
const { createHash } = require("crypto");
const fs = require("fs");

const PROJECT = process.env.ANYA_FIREBASE_PROJECT || "anya-tennis-61658";
const DAY = 86400;

const hash = (email) =>
  createHash("sha256").update(email.trim().toLowerCase(), "utf8").digest("hex");

function emailsFrom(args) {
  const out = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--file") {
      const body = fs.readFileSync(args[++i], "utf8");
      // One per line; blank lines and #comments ignored so you can annotate.
      out.push(...body.split(/\r?\n/).map((l) => l.trim())
        .filter((l) => l && !l.startsWith("#")));
    } else {
      out.push(args[i]);
    }
  }
  return out;
}

function admin() {
  if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
    console.error(
      "error: this command needs admin credentials.\n" +
      "       Either set GOOGLE_APPLICATION_CREDENTIALS to a service-account\n" +
      "       key, or use `hash` and paste the ids into the Firestore console."
    );
    process.exit(1);
  }
  const { initializeApp } = require("firebase-admin/app");
  const { getFirestore } = require("firebase-admin/firestore");
  const { getAuth } = require("firebase-admin/auth");
  initializeApp({ projectId: PROJECT });
  return { db: getFirestore(), auth: getAuth() };
}

async function main() {
  const [cmd, ...args] = process.argv.slice(2);

  if (cmd === "hash") {
    const emails = emailsFrom(args);
    if (!emails.length) return usage();
    console.log(`# ${emails.length} document id(s) for the 'grandfathered' collection`);
    console.log("# create each as an empty document; the id IS the data\n");
    for (const e of emails) console.log(`${hash(e)}   # ${e}`);
    return;
  }

  if (cmd === "add" || cmd === "remove") {
    const { db } = admin();
    const emails = emailsFrom(args);
    if (!emails.length) return usage();
    const batch = db.batch();
    for (const e of emails) {
      const ref = db.doc(`grandfathered/${hash(e)}`);
      if (cmd === "add") batch.set(ref, { addedAt: Math.floor(Date.now() / 1000), note: "beta tester" });
      else batch.delete(ref);
    }
    await batch.commit();
    console.log(`${cmd === "add" ? "added" : "removed"} ${emails.length} address(es)`);
    return;
  }

  if (cmd === "check") {
    const { db } = admin();
    for (const e of emailsFrom(args)) {
      const d = await db.doc(`grandfathered/${hash(e)}`).get();
      console.log(`${e}: ${d.exists ? "on the list" : "NOT on the list"}`);
    }
    return;
  }

  if (cmd === "list") {
    const { db } = admin();
    const snap = await db.collection("grandfathered").get();
    console.log(`${snap.size} entries (digests — the addresses cannot be recovered):`);
    snap.forEach((d) => console.log(`  ${d.id}  ${JSON.stringify(d.data())}`));
    return;
  }

  if (cmd === "comp") {
    const { db, auth } = admin();
    const [email, daysRaw] = args;
    if (!email) return usage();
    const days = Number(daysRaw ?? 365);
    const user = await auth.getUserByEmail(email);
    const until = Math.floor(Date.now() / 1000) + days * DAY;
    await db.doc(`users/${user.uid}`).set(
      { entitlement: { active: true, until, source: "comp" },
        updatedAt: Math.floor(Date.now() / 1000) },
      { merge: true });
    await auth.setCustomUserClaims(user.uid, {
      ...(user.customClaims ?? {}), ent: 1, entExp: until });
    console.log(`comped ${email} (${user.uid}) for ${days} days`);
    console.log("They must relaunch, or sign out and back in, to pick up the claim.");
    return;
  }

  usage();
}

function usage() {
  console.error("commands: hash | add | remove | check | list | comp");
  console.error("  hash needs no credentials; the rest need GOOGLE_APPLICATION_CREDENTIALS");
  process.exitCode = 1;
}

main().catch((e) => { console.error(e.message || e); process.exitCode = 1; });
